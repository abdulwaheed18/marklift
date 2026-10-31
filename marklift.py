#!/usr/bin/env python3
"""
MarkLift - batch removal of the visible corner watermark (e.g. the Gemini /
Nano Banana sparkle, Veo/Flow corner mark) from images and videos.

How it works
------------
The visible mark is alpha-blended on top of the picture:

    shown = alpha * logo_colour + (1 - alpha) * original

Once the alpha map is known (you "calibrate" it once from a black image that
carries the mark), the original pixels are recovered exactly:

    original = (shown - alpha * logo_colour) / (1 - alpha)

Pixels where alpha is almost 1 (or where compression broke the maths) are
touched up with a tiny inpaint.

Commands
--------
  calibrate  Learn the watermark from black (optionally + white) samples.
  clean      Batch-clean files and/or folders of images and videos.
  masks      List calibrated masks.

Run `python marklift.py <command> -h` for options.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MASKS = SCRIPT_DIR / "masks"
CORNER_FRAC = 0.35      # watermark is searched in this bottom-right fraction
ALPHA_MAX = 0.97        # never divide by (1 - alpha) smaller than this allows
ACCEPT_RATIO = 0.8      # fit score (outline edges after/before + continuity) must be <= this
SEARCH_FRAC = 0.35      # marks are searched in a bottom-right square this big (of the short side)
MIN_SIZE_FRAC = 0.025   # smallest / largest mark searched for, as a fraction of the short side
MAX_SIZE_FRAC = 0.14
MAX_MARKS = 3           # e.g. image-to-video clips carry the image mark plus the video mark
EDGE_RESIDUE = 1.3      # outline gradient this much above its surroundings -> inpaint that thin band
CONTINUITY_W = 4.0      # weight of "logo interior matches its surroundings" in the fit score (video)
CONTINUITY_W_SINGLE = 0.5   # same, for a single image / frame (one background: weak prior)
PRESENT = 0.8           # video: presence score below this = overlay mark is in the frame
PRESENT_ACCEPT = 0.6    # video: median presence this low accepts a mark whose fit score is marginal
GATE_HALF = 6           # video: presence decisions use the median over +-6 frames
SCENE_FRAC = 0.2        # a "static" mark absent in more samples than this is scene content
TRACK_ACCEPT = 0.6      # video: stricter per-frame threshold for marks that move with the scene
OPAQUE_FLAT = 0.2       # interior texture below this fraction of the surroundings' = opaque mark
OPAQUE_STEP = 25        # flat interior still this far off its surroundings after un-blending = opaque
LEARN_MIN_FRAMES = 150 # `learn`: usable frames needed for a reliable per-pixel fit
SHAPE_MATCH = 0.6      # scene mark that can't be un-blended but clearly has the logo shape -> inpaint


# --------------------------------------------------------------------------
# small I/O helpers (unicode-safe on Windows)
# --------------------------------------------------------------------------
def imread(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 97]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, 97]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"cannot encode {path}")
    buf.tofile(str(path))


def need_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg/ffprobe not found on PATH - install ffmpeg to process videos.")


def probe_video(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,nb_frames:"
           "stream_side_data=rotation:stream_tags=rotate",
           "-of", "json", str(path)]
    info = json.loads(subprocess.check_output(cmd))["streams"][0]
    w, h = int(info["width"]), int(info["height"])
    rot = 0
    for sd in info.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int(info.get("tags", {}).get("rotate", 0))
    if abs(rot) % 180 == 90:
        w, h = h, w
    num, den = info.get("r_frame_rate", "30/1").split("/")
    fps = float(num) / float(den or 1)
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)]).strip() or 0)
    return {"w": w, "h": h, "fps": fps, "fps_str": info.get("r_frame_rate", "30/1"),
            "duration": dur}


def video_frames(path: Path, w: int, h: int):
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path),
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                            stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
    finally:
        proc.stdout.close()
        proc.wait()


def sample_frames(path: Path, info: dict, n: int = 15) -> list[np.ndarray]:
    """Grab n frames spread across the clip (cheap seeks)."""
    frames = []
    dur = max(info["duration"], 0.1)
    for i in range(n):
        t = dur * (i + 0.5) / n
        out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
                              "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                             capture_output=True).stdout
        if len(out) == info["w"] * info["h"] * 3:
            frames.append(np.frombuffer(out, np.uint8).reshape(info["h"], info["w"], 3))
    if not frames:  # fall back to the first frames
        for i, f in enumerate(video_frames(path, info["w"], info["h"])):
            frames.append(f)
            if i >= n:
                break
    return frames


def load_sample(path: Path) -> np.ndarray:
    """An image, or the per-pixel median of frames sampled from a video."""
    if path.suffix.lower() in VID_EXT:
        need_ffmpeg()
        info = probe_video(path)
        return np.median(np.stack(sample_frames(path, info, 20)), axis=0).astype(np.uint8)
    return imread(path)[..., :3]


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def estimate_mask(black: np.ndarray, white: np.ndarray | None = None) -> dict:
    H, W = black.shape[:2]
    y0, x0 = int(H * (1 - CORNER_FRAC)), int(W * (1 - CORNER_FRAC))
    b = black[y0:, x0:].astype(np.float32)

    if white is not None:
        if white.shape[:2] != black.shape[:2]:
            raise ValueError("black and white samples must be the same size")
        wt = white[y0:, x0:].astype(np.float32)
        alpha = np.clip(1.0 - (wt - b).mean(axis=2) / 255.0, 0, 1)
        bg = 0.0
    else:
        bg = float(np.median(b))                 # "black" is rarely exactly 0
        alpha = np.clip((b.mean(axis=2) - bg) / (255.0 - bg), 0, 1)

    # denoise: drop faint noise and stray specks, keep the logo component(s)
    alpha[alpha < 0.02] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats((alpha > 0.04).astype(np.uint8), 8)
    if n <= 1:
        raise ValueError("no watermark found in the bottom-right of the calibration sample")
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros_like(alpha, bool)
    for i, a in enumerate(areas, start=1):
        if a >= 0.05 * areas.max():
            keep |= lab == i
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    alpha = np.where(keep, alpha, 0)

    ys, xs = np.nonzero(alpha > 0.02)
    pad = 2
    ya, yb = max(ys.min() - pad, 0), min(ys.max() + pad + 1, alpha.shape[0])
    xa, xb = max(xs.min() - pad, 0), min(xs.max() + pad + 1, alpha.shape[1])
    crop = alpha[ya:yb, xa:xb].astype(np.float32)

    # logo colour: from black sample, colour = (pixel - bg*(1-a)) / a
    strong = crop > 0.25
    if strong.any():
        bb = b[ya:yb, xa:xb][strong]
        aa = crop[strong][:, None]
        color = np.clip((bb - bg * (1 - aa)) / aa, 0, 255).mean(axis=0)
    else:
        color = np.array([255.0, 255.0, 255.0])

    return {"alpha": crop, "x": int(x0 + xa), "y": int(y0 + ya), "W": int(W), "H": int(H),
            "color": color.astype(np.float32)}


def save_mask(m: dict, masks_dir: Path, name: str) -> Path:
    d = masks_dir / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{m['W']}x{m['H']}.npz"
    np.savez_compressed(p, alpha=m["alpha"], x=m["x"], y=m["y"], W=m["W"], H=m["H"],
                        color=m["color"])
    cv2.imwrite(str(p.with_suffix(".png")), (m["alpha"] * 255).astype(np.uint8))  # preview
    return p


def load_masks(masks_dir: Path, only: list[str] | None = None) -> list[dict]:
    out = []
    if not masks_dir.exists():
        return out
    for p in sorted(masks_dir.glob("*/*.npz")):
        if only and p.parent.name not in only:
            continue
        z = np.load(p)
        out.append({"name": p.parent.name, "alpha": z["alpha"].astype(np.float32),
                    "x": int(z["x"]), "y": int(z["y"]), "W": int(z["W"]), "H": int(z["H"]),
                    "color": z["color"].astype(np.float32), "file": p.name})
    return out


# --------------------------------------------------------------------------
# detection + removal
# --------------------------------------------------------------------------
def edge_energy(patch: np.ndarray, weight: np.ndarray) -> float:
    """Image gradient strength measured only along the logo's own edges.
    Not clipped, so an over-subtracted (out of range) result scores badly."""
    p = patch.astype(np.float32)
    g = 0.114 * p[..., 0] + 0.587 * p[..., 1] + 0.299 * p[..., 2]
    # light blur: pixel noise (which un-blending amplifies) must not outweigh the outline
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float((np.sqrt(gx * gx + gy * gy) * weight).sum()) + 1e-6


def logo_edges(a: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    w = np.sqrt(gx * gx + gy * gy)
    return w / (w.max() + 1e-6)


def unblend(patch: np.ndarray, alpha: np.ndarray, color: np.ndarray) -> np.ndarray:
    a = np.minimum(alpha, ALPHA_MAX)[..., None]
    return (patch.astype(np.float32) - a * color) / (1.0 - a)


def shape_templates(masks: list[dict]) -> list[dict]:
    """One normalised logo shape per mask set, taken from its sharpest (largest) calibration."""
    best = {}
    for m in masks:
        if m["name"] not in best or m["alpha"].size > best[m["name"]]["alpha"].size:
            best[m["name"]] = m
    out = []
    for m in best.values():
        peak, grey = float(m["alpha"].max()), float(m["color"].max())
        out.append({"name": m["name"], "shape": m["alpha"] / peak, "peak": peak, "peak0": peak,
                    "tint": m["color"] / grey, "grey": grey})
    return out


def render(mk: dict, soft: bool = True):
    """Rasterise a mark (shape at size/centre, sub-pixel) -> (alpha, x0, y0)."""
    if mk.get("exact") is not None:              # calibrated at this exact resolution
        a, x, y = mk["exact"]
        return a / a.max(), x, y
    t = mk["shape"]
    th, tw = t.shape
    s = mk["size"] / max(th, tw)
    w, h = int(np.ceil(tw * s)) + 4, int(np.ceil(th * s)) + 4
    x0, y0 = int(np.floor(mk["cx"] - w / 2)), int(np.floor(mk["cy"] - h / 2))
    pw, ph = max(2, round(tw * s)), max(2, round(th * s))
    pre = cv2.resize(t, (pw, ph), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    sx, sy = tw * s / pw, th * s / ph
    M = np.float32([[sx, 0, mk["cx"] - x0 - (pw - 1) / 2 * sx],
                    [0, sy, mk["cy"] - y0 - (ph - 1) / 2 * sy]])
    a = cv2.warpAffine(pre, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    e = mk.get("e", 0.0) if soft else 0.0
    if e > 0.05:                                 # softer edge than the template
        a = cv2.GaussianBlur(a, (0, 0), e)
    elif e < -0.05:                              # crisper edge than the template
        a = a + (-e) * (a - cv2.GaussianBlur(a, (0, 0), 1.0))
    return np.clip(a, 0, 1).astype(np.float32), x0, y0


class Prepared:
    """A fitted mark rendered once, ready to be applied to many frames."""

    def __init__(self, mk: dict):
        # imperfect fit (rescaled / re-encoded mark) -> also fill the thin outline band
        self.fix_edges = bool(mk.get("fix"))
        a, self.x, self.y = render(mk)
        self.alpha = np.clip(a * float(mk["peak"]), 0, 1).astype(np.float32)
        self.color = (mk["tint"] * mk["grey"]).astype(np.float32)
        self.wgt = logo_edges(render(mk, soft=False)[0])
        self.h, self.w = a.shape
        # interior of the logo vs a thin ring just outside it (continuity check)
        body = (a > 0.03).astype(np.uint8)
        k = max(3, (min(a.shape) // 10) | 1)
        self.core = (cv2.erode(body, np.ones((k, k), np.uint8)) > 0)   # interior, away from the rim
        if not self.core.any():
            self.core = a >= 0.7
        self.ring = (cv2.dilate(body, np.ones((7, 7), np.uint8)) > 0) & \
                    ~(cv2.dilate(body, np.ones((3, 3), np.uint8)) > 0)
        self.strength = float(self.alpha.max() * self.color.mean()) + 1.0

    def inside(self, W: int, H: int) -> bool:
        return self.x >= 0 and self.y >= 0 and self.x + self.w <= W and self.y + self.h <= H

    def ratio(self, img: np.ndarray, cont_w: float = 0.0) -> float:
        """Edge energy along the logo outline after / before removal (<1 = mark removed)."""
        p = img[self.y:self.y + self.h, self.x:self.x + self.w]
        rec = unblend(p, self.alpha, self.color)
        r = edge_energy(rec, self.wgt) / edge_energy(p, self.wgt)
        on = self.alpha > 0.05
        if on.any():                             # penalise impossible (out of range) pixels
            r += 2.0 * float((((rec < -12) | (rec > 267)).any(axis=2) & on).sum() / on.sum())
        if cont_w and self.core.any() and self.ring.any():  # interior must match its surroundings
            g = rec.mean(axis=2)
            r += cont_w * abs(float(g[self.core].mean() - g[self.ring].mean())) / self.strength
        return r

    def outline_residue(self, img: np.ndarray) -> float:
        """After un-blending: gradient on the logo outline / gradient on a ring beside it.
        About 1 = clean; clearly above 1 = a visible outline is left."""
        p = img[self.y:self.y + self.h, self.x:self.x + self.w]
        g = unblend(p, self.alpha, self.color).mean(axis=2).astype(np.float32)
        mag = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) ** 2)
        band = self.wgt > 0.3
        if not band.any() or not self.ring.any():
            return 1.0
        return float(mag[band].mean() / (mag[self.ring].mean() + 1.0))

    def presence(self, img: np.ndarray) -> float:
        """Edges after removing the mark / edges after adding it once more.
        Mark present -> well below 1 (even on busy backgrounds); absent -> about 1 or more."""
        p = img[self.y:self.y + self.h, self.x:self.x + self.w].astype(np.float32)
        a = self.alpha[..., None]
        added = a * self.color + (1 - a) * p
        return (edge_energy(unblend(p, self.alpha, self.color), self.wgt)
                / edge_energy(added, self.wgt))

    def opaque(self, img: np.ndarray) -> bool:
        """The mark hides the picture completely here (flat interior while the surroundings
        have texture, or un-blending leaves a clearly different patch): it must be filled in."""
        if not self.core.any() or not self.ring.any():
            return False
        p = img[self.y:self.y + self.h, self.x:self.x + self.w].astype(np.float32)
        g = p.mean(axis=2)
        s_in, s_ring = float(g[self.core].std()), float(g[self.ring].std())
        if s_ring > 4 and s_in < OPAQUE_FLAT * s_ring:
            return True
        rec = unblend(p, self.alpha, self.color).mean(axis=2)
        return s_in < 2.0 and abs(float(rec[self.core].mean() - g[self.ring].mean())) > OPAQUE_STEP

    def remove(self, img: np.ndarray, touchup: bool = True) -> np.ndarray:
        if touchup and self.opaque(img):
            return self.inpaint(img)
        out = apply_removal(img, self.alpha, self.x, self.y, self.color, touchup)
        if touchup and self.fix_edges:
            m = np.zeros(img.shape[:2], np.uint8)
            band = cv2.dilate((self.wgt > 0.3).astype(np.uint8), np.ones((3, 3), np.uint8))
            m[self.y:self.y + self.h, self.x:self.x + self.w] = band * 255
            ys, xs = self.y, self.x
            pad = 8
            ya, yb = max(ys - pad, 0), min(ys + self.h + pad, img.shape[0])
            xa, xb = max(xs - pad, 0), min(xs + self.w + pad, img.shape[1])
            out[ya:yb, xa:xb] = cv2.inpaint(out[ya:yb, xa:xb], m[ya:yb, xa:xb], 2, cv2.INPAINT_TELEA)
        return out

    def shape_match(self, img: np.ndarray) -> float:
        """|correlation| of the local detail with the logo shape (catches dark or
        light versions of a mark that is part of the scene)."""
        p = cv2.cvtColor(img[self.y:self.y + self.h, self.x:self.x + self.w], cv2.COLOR_BGR2GRAY)
        p = p.astype(np.float32)
        hp = p - cv2.GaussianBlur(p, (0, 0), max(2.0, self.w / 6))
        a = self.alpha - cv2.GaussianBlur(self.alpha, (0, 0), max(2.0, self.w / 6))
        hp, a = hp - hp.mean(), a - a.mean()
        return abs(float((hp * a).sum() / (np.sqrt((hp * hp).sum() * (a * a).sum()) + 1e-6)))

    def inpaint(self, img: np.ndarray) -> np.ndarray:
        m = np.zeros(img.shape[:2], np.uint8)
        m[self.y:self.y + self.h, self.x:self.x + self.w] = \
            cv2.dilate((self.alpha > 0.1 * self.alpha.max()).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
        return inpaint_mask(img, m)


def mark_ratio(frames: list[np.ndarray], mk: dict) -> float:
    p = Prepared(mk)
    H, W = frames[0].shape[:2]
    if not p.inside(W, H):
        return 9.0
    # median over frames: robust to frames (e.g. cut-aways in an edit) without the mark.
    # "interior matches surroundings" is only a reliable prior over many backgrounds
    cw = CONTINUITY_W if len(frames) >= 5 else CONTINUITY_W_SINGLE
    return float(np.median([p.ratio(f, cw) for f in frames]))


def refine(frames: list[np.ndarray], mk: dict, budget: int = 260,
           keys=("cx", "cy", "size", "peak", "grey", "e"), limits: dict | None = None):
    """Pattern search over position/size/strength/colour/edge softness."""
    H, W = frames[0].shape[:2]
    short = min(W, H)
    limits = {"peak": (mk["peak0"] * 0.5, min(0.85, mk["peak0"] * 1.6)), "grey": (140.0, 255.0),
              "e": (-3.5, 2.5), "size": (MIN_SIZE_FRAC * short, MAX_SIZE_FRAC * short),
              **(limits or {})}
    step = {"cx": 1.0, "cy": 1.0, "size": max(0.5, mk["size"] * 0.04),
            "peak": 0.04, "grey": 12.0, "e": 0.4}
    step = {k: v for k, v in step.items() if k in keys}
    best, n = mark_ratio(frames, mk), 1
    while n < budget:
        improved = False
        for k in step:
            for sgn in (1, -1):
                t = dict(mk)
                lo, hi = limits.get(k, (-1e9, 1e9))
                t[k] = min(max(mk[k] + sgn * step[k], lo), hi)
                if t[k] == mk[k]:
                    continue
                r = mark_ratio(frames, t); n += 1
                if r < best - 1e-4:
                    mk, best, improved = t, r, True
                    break
        if not improved:
            for k in step:
                step[k] *= 0.5
            if step["cx"] < 0.06:
                break
    return mk, best


def blend_from_frames(frames: list[np.ndarray], mk: dict) -> dict | None:
    """With a varying background under a static mark, fit  inside = a*C + (1-a)*outside
    across frames (least squares) -> the mark's true strength a and colour C."""
    p = Prepared(mk)
    H, W = frames[0].shape[:2]
    if not p.inside(W, H) or not p.core.any() or not p.ring.any():
        return None
    S, B = [], []
    for f in frames:
        g = f[p.y:p.y + p.h, p.x:p.x + p.w].astype(np.float32).mean(axis=2)
        ring = g[p.ring]
        if ring.std() > 12:            # busy background around the mark: unreliable
            continue
        S.append(g[p.core].mean()); B.append(ring.mean())
    if len(S) < 5 or np.ptp(B) < 40:
        return None
    k, b = np.polyfit(B, S, 1)
    a = 1.0 - k
    if not 0.08 <= a <= 0.85:
        return None
    C = b / a
    if not 120.0 <= C <= 265.0:
        return None
    core_a = float(p.alpha[p.core].mean()) / mk["peak"]        # core alpha as a fraction of peak
    return {**mk, "peak": a / core_a, "grey": float(min(C, 255.0))}


def coarse_candidates(gray: np.ndarray, tpls: list[dict], found: list[dict], masks: list[dict],
                      top: int = 24, sizes: tuple[float, float] | None = None) -> list[dict]:
    """Multi-scale template search for the logo shape in the bottom-right corner."""
    H, W = gray.shape
    side = max(64, int(SEARCH_FRAC * min(W, H)))
    x0, y0 = max(W - side, 0), max(H - side, 0)
    win = gray[y0:, x0:]
    lo = max(12.0, MIN_SIZE_FRAC * min(W, H))
    hi = max(16.0, MAX_SIZE_FRAC * min(W, H))
    if sizes:
        lo, hi = max(lo, sizes[0]), min(hi, sizes[1])
    cands = []
    for t in tpls:
        th, tw = t["shape"].shape
        size = lo
        while size <= hi:
            s = size / max(th, tw)
            a = cv2.resize(t["shape"], (max(3, round(tw * s)), max(3, round(th * s))),
                           interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            if a.shape[0] >= win.shape[0] or a.shape[1] >= win.shape[1]:
                break
            res = np.nan_to_num(cv2.matchTemplate(win, a * 255, cv2.TM_CCOEFF_NORMED), nan=-1.0)
            for _ in range(2):                   # two strongest peaks per scale
                _, mv, _, (px, py) = cv2.minMaxLoc(res)
                cands.append({**t, "size": size, "e": 0.0, "ncc": mv,
                              "cx": x0 + px + (a.shape[1] - 1) / 2, "cy": y0 + py + (a.shape[0] - 1) / 2})
                r = max(2, min(a.shape) // 2)
                res[max(py - r, 0):py + r + 1, max(px - r, 0):px + r + 1] = -1.0
            size *= 1.08
    def clash(c):
        return any(abs(c["cx"] - f["cx"]) < f["size"] * 0.4 and abs(c["cy"] - f["cy"]) < f["size"] * 0.4
                   for f in found)
    cands = [c for c in cands if not clash(c)]
    cands.sort(key=lambda c: -c["ncc"])
    cands = cands[:top]
    # plus the calibrated positions of same-aspect masks, scaled to this resolution
    by_name = {t["name"]: t for t in tpls}
    for m in masks:
        s = W / m["W"]
        if m["name"] not in by_name or abs(H / m["H"] - s) > 0.01 * s:
            continue
        ah, aw = m["alpha"].shape
        c = {**by_name[m["name"]], "size": float(max(ah, aw)) * s,
             "cx": (W - 1) - ((m["W"] - 1) - (m["x"] + (aw - 1) / 2)) * s,
             "cy": (H - 1) - ((m["H"] - 1) - (m["y"] + (ah - 1) / 2)) * s,
             "peak": float(m["alpha"].max()), "e": 0.0, "ncc": 2.0}
        if (not sizes or sizes[0] <= c["size"] <= sizes[1]) and not clash(c):
            cands.append(c)
    return cands


def needs_edge_fix(frames: list[np.ndarray], mk: dict) -> bool:
    p = Prepared(mk)
    H, W = frames[0].shape[:2]
    if not p.inside(W, H):
        return False
    return float(np.median([p.outline_residue(f) for f in frames])) > EDGE_RESIDUE


def find_marks(ref: np.ndarray, masks: list[dict], frames: list[np.ndarray] | None = None,
               force: bool = False, exclude: list[dict] = (), max_marks: int = MAX_MARKS,
               accept: float = ACCEPT_RATIO) -> list[tuple[dict, float]]:
    """Detect and fit every watermark in the corner. Returns [(mark, edge_ratio), ...]."""
    frames = [f.copy() for f in (frames or [ref])]
    work = ref.copy()
    tpls = shape_templates(masks)
    found = []
    for _ in range(max_marks):
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # marks in one file come at a similar scale (video mark + baked-in image mark)
        known = list(exclude) + [f for f, _ in found]
        sizes = (0.6 * known[0]["size"], 1.4 * known[0]["size"]) if known else None
        cands = coarse_candidates(gray, tpls, known, masks, sizes=sizes)
        if not cands:
            break
        # a further mark may be fainter than the calibrated overlay (e.g. baked into the scene)
        lim = {"peak": (0.3 * tpls[0]["peak0"], 0.85)} if known else None
        scored = sorted(((mark_ratio(frames, c), i) for i, c in enumerate(cands)))[:3]
        best, best_r = None, 9.0
        for r0, i in scored:
            if r0 > 1.5:
                continue
            mk, r = refine(frames, cands[i], limits=lim)
            if r < best_r:
                best, best_r = mk, r
        if best is not None and len(frames) >= 5:
            fit = blend_from_frames(frames, best)       # measured strength/colour (video)
            if fit is not None:
                mk, r = refine(frames, fit, budget=120, keys=("cx", "cy", "size", "e"))
                if r <= best_r * 1.15:                  # keep unless it clearly fits worse
                    best, best_r = mk, r
        # a calibrated mask used as-is (exact resolution) or simply resized by the
        # resolution ratio wins when it fits (about) as well as the free fit
        H, W = work.shape[:2]
        for m in masks:
            if best is None:
                continue
            s = W / m["W"]
            if abs(H / m["H"] - s) > 0.01 * s:          # different aspect ratio
                continue
            ah, aw = m["alpha"].shape
            a = m["alpha"] if abs(s - 1) < 1e-6 else np.clip(cv2.resize(
                m["alpha"], (max(3, round(aw * s)), max(3, round(ah * s))),
                interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC), 0, 1)
            h2, w2 = a.shape
            x2 = W - round((m["W"] - m["x"] - aw) * s) - w2
            y2 = H - round((m["H"] - m["y"] - ah) * s) - h2
            ex = {**best, "exact": (a, x2, y2), "name": m["name"],
                  "peak": float(a.max()), "grey": float(m["color"].max()),
                  "tint": m["color"] / float(m["color"].max()),
                  "cx": x2 + (w2 - 1) / 2, "cy": y2 + (h2 - 1) / 2, "size": float(max(h2, w2))}
            if any(abs(ex["cx"] - f["cx"]) < f["size"] * 0.4 and abs(ex["cy"] - f["cy"]) < f["size"] * 0.4
                   for f in list(exclude) + [f for f, _ in found]):
                continue
            exact_res = (m["W"], m["H"]) == (W, H)
            if exact_res:   # ground truth: judge only the outline, not the flat-background prior
                pe = Prepared(ex)
                r = float(np.median([pe.ratio(f) for f in frames])) if pe.inside(W, H) else 9.0
            else:
                r = mark_ratio(frames, ex)
            # a mask made at this very resolution is ground truth: use it whenever it is
            # acceptable; a merely resized one must fit about as well as the free fit
            if r <= best_r + 0.03 or (exact_res and r <= max(accept, best_r + 0.3)):
                best, best_r = ex, r
                if exact_res:
                    break
        ok = best is not None and (best_r <= accept or (force and not found))
        if best is not None and not ok and len(frames) >= 5 and best_r <= 1.6:
            # busy texture caps the fit score; with many frames the presence test decides
            p = Prepared(best)
            ok = float(np.median([p.presence(f) for f in frames])) <= PRESENT_ACCEPT
        if not ok:
            break
        best = {**best, "fix": needs_edge_fix(frames, best)}
        found.append((best, best_r))
        p = Prepared(best)
        work = p.remove(work, touchup=False)
        frames = [p.remove(f, touchup=False) for f in frames]
    return found


class Tracker:
    """Follows a mark that moves/zooms with the scene (e.g. the Gemini mark baked into
    the start image of an image-to-video clip) and removes it frame by frame."""

    def __init__(self, proto: dict, masks: list[dict], exclude: list[dict]):
        self.proto, self.exclude, self.prev, self.hits = proto, exclude, None, 0
        self.tpls = [t for t in shape_templates(masks) if t["name"] == proto["name"]]

    def step(self, f: np.ndarray, touchup: bool = True) -> np.ndarray:
        # a scene mark can slide partly out of the picture: work on a frame extended
        # (mirrored) past the right/bottom edge, then crop back
        H, W = f.shape[:2]
        B = int(self.proto["size"])
        fp = cv2.copyMakeBorder(f, 0, B, 0, B, cv2.BORDER_REFLECT)
        return self._step(fp, touchup)[:H, :W]

    def _step(self, f: np.ndarray, touchup: bool = True) -> np.ndarray:
        # a mark that is part of the scene can render brighter than the overlay, or
        # darker than the background (blend towards black): fit from both starts
        lim = {"peak": (0.05, 0.85), "grey": (0.0, 255.0)}
        keys = ("cx", "cy", "size", "peak", "grey")
        mk, r = None, 9.0
        if self.prev:                           # warm start from the last frame: cheap
            mk, r = refine([f], {**self.prev}, budget=45, keys=keys, limits=lim)
        if r > TRACK_ACCEPT:                    # lost it: search the corner again
            s = self.proto["size"]
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            cands = coarse_candidates(gray, self.tpls, self.exclude, [], top=4, sizes=(s * 0.5, s * 1.6))
            cands.insert(0, dict(self.proto))
            fixed = {k: self.proto[k] for k in ("e", "peak0", "tint", "shape", "name")}
            cands = [{**c, **fixed, "peak": c.get("peak", self.proto["peak"]),
                      "grey": c.get("grey", self.proto["grey"])} for c in cands]
            c = min(cands, key=lambda c: mark_ratio([f], c))
            for start in (c, {**c, "peak": 0.12, "grey": 20.0}):
                m2, r2 = refine([f], start, budget=110, keys=keys, limits=lim)
                if r2 < r:
                    mk, r = m2, r2
        mk = {**mk, "fix": needs_edge_fix([f], mk)} if mk else mk
        p = Prepared(mk)
        if not p.inside(*f.shape[1::-1]):
            self.prev = None
            return f
        if r <= TRACK_ACCEPT:
            self.prev, self.hits = mk, self.hits + 1
            return p.remove(f, touchup)
        if p.shape_match(f) >= SHAPE_MATCH:     # e.g. a darker copy of the logo: fill it in
            self.prev, self.hits = mk, self.hits + 1
            return p.inpaint(f)
        self.prev = None
        return f


def find_moving(frames: list[np.ndarray], masks: list[dict], static: list[dict]):
    """A mark found in >= 2 individual sample frames (outside the static ones) -> prototype."""
    hits = []
    for f in frames:
        hits += find_marks(f, masks, None, exclude=static, max_marks=1, accept=TRACK_ACCEPT)
    if len(hits) < 2:
        return None
    best = dict(min(hits, key=lambda h: h[1])[0])
    for k in ("peak", "grey", "e", "size"):
        best[k] = float(np.median([h[0][k] for h in hits]))
    return best, len(hits)


def describe(found) -> str:
    return "; ".join(f"[{mk['name']}] {mk['size']:.0f}px at ({mk['cx']:.0f},{mk['cy']:.0f}) "
                     f"alpha {mk['peak']:.2f} edge {r:.2f}" for mk, r in found)


def apply_removal(img: np.ndarray, a: np.ndarray, x: int, y: int, color: np.ndarray,
                  touchup: bool = True) -> np.ndarray:
    out = img.copy()
    h, w = a.shape
    patch = img[y:y + h, x:x + w]
    rec = unblend(patch, a, color)
    bad = (a >= 0.9) | (((rec < -10) | (rec > 265)).any(axis=2) & (a > 0.05))
    out[y:y + h, x:x + w] = np.clip(np.round(rec), 0, 255).astype(np.uint8)
    if touchup and bad.any():
        m = np.zeros(img.shape[:2], np.uint8)
        m[y:y + h, x:x + w][bad] = 255
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        pad = max(h, w)
        ya, yb = max(y - pad, 0), min(y + h + pad, img.shape[0])
        xa, xb = max(x - pad, 0), min(x + w + pad, img.shape[1])
        out[ya:yb, xa:xb] = cv2.inpaint(out[ya:yb, xa:xb], m[ya:yb, xa:xb], 3, cv2.INPAINT_TELEA)
    return out


# ---- box fallback (no calibration): find bright mark pixels in a corner box ----
def parse_box(spec: str):
    # "WxH" or "WxH+MX+MY"  (margins from the right/bottom edge)
    parts = spec.lower().replace(" ", "").split("+")
    bw, bh = (int(v) for v in parts[0].split("x"))
    mx = int(parts[1]) if len(parts) > 1 else 0
    my = int(parts[2]) if len(parts) > 2 else 0
    return bw, bh, mx, my


def box_mask(img: np.ndarray, box) -> np.ndarray:
    H, W = img.shape[:2]
    bw, bh, mx, my = box
    x1, y1 = W - mx, H - my
    x0, y0 = max(x1 - bw, 0), max(y1 - bh, 0)
    region = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    k = max(7, (min(bw, bh) // 3) | 1)
    tophat = cv2.morphologyEx(region, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    thr = max(12, int(np.percentile(tophat, 90)))
    sel = (tophat >= thr).astype(np.uint8)
    m = np.zeros((H, W), np.uint8)
    if sel.sum() < 4:
        m[y0:y1, x0:x1] = 255            # nothing obvious -> whole box
    else:
        m[y0:y1, x0:x1] = cv2.dilate(sel, np.ones((5, 5), np.uint8)) * 255
    return m


def inpaint_mask(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return img
    pad = 40
    ya, yb = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    xa, xb = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    out = img.copy()
    out[ya:yb, xa:xb] = cv2.inpaint(img[ya:yb, xa:xb], m[ya:yb, xa:xb], 5, cv2.INPAINT_TELEA)
    return out


# --------------------------------------------------------------------------
# per-file processing
# --------------------------------------------------------------------------
def process_image(src: Path, dst: Path, masks, box, force, touchup) -> str:
    raw = imread(src)
    alpha_ch = raw[..., 3] if raw.shape[2] == 4 else None
    img = np.ascontiguousarray(raw[..., :3])
    found = find_marks(img, masks, None, force) if masks else []
    if found:
        out = img
        for mk, r in found:
            out = Prepared(mk).remove(out, touchup)
        how = describe(found)
    elif box:
        out = inpaint_mask(img, box_mask(img, box))
        how = "box inpaint"
    else:
        return "SKIP  no watermark detected"
    if alpha_ch is not None:
        out = np.dstack([out, alpha_ch])
    imwrite(dst, out)
    return "OK    " + how


def process_video(src: Path, dst: Path, masks, box, force, touchup, crf: int) -> str:
    need_ffmpeg()
    info = probe_video(src)
    W, H = info["w"], info["h"]
    frames = sample_frames(src, info, 15)
    ref = np.median(np.stack(frames), axis=0).astype(np.uint8)   # static mark survives
    found = find_marks(ref, masks, frames, force) if masks else []
    # a true overlay is in (nearly) every sample; one that comes and goes is scene content
    scene = [(mk, r) for mk, r in found
             if np.mean([Prepared(mk).presence(f) > PRESENT for f in frames]) > SCENE_FRAC]
    found = [x for x in found if not any(x[0] is s[0] for s in scene)]
    moving = None
    if masks:
        rest = frames
        for mk, _ in found:
            p = Prepared(mk)
            rest = [p.remove(f, touchup=False) for f in rest]
        moving = find_moving(rest, masks, [mk for mk, _ in found + scene])
    trackers = [Tracker(mk, masks, [m for m, _ in found]) for mk, _ in scene]
    if moving:
        trackers.append(Tracker(moving[0], masks, [m for m, _ in found + scene]))
    stats = None
    if found or trackers:
        preps = [Prepared(mk) for mk, _ in found]
        stats = [0] * len(preps)

        def fn_stream(frames_iter):
            for f in static_stream(frames_iter):
                for t in trackers:
                    f = t.step(f, touchup)
                yield f

        def static_stream(frames_iter):
            # per-frame gate (median over a small window) so cut-aways without the
            # mark in an edited video are left untouched instead of getting a "hole"
            pending, ratios = [], []          # ratios: one row per frame, absolute index
            def emit():
                i = len(ratios) - len(pending)
                win = ratios[max(i - GATE_HALF, 0):i + GATE_HALF + 1]
                out = pending.pop(0)
                for k, p in enumerate(preps):
                    if np.median([r[k] for r in win]) < PRESENT:
                        out = p.remove(out, touchup); stats[k] += 1
                return out
            for f in frames_iter:
                pending.append(f); ratios.append([p.presence(f) for p in preps])
                while len(pending) > GATE_HALF:
                    yield emit()
            while pending:                    # end of clip: window shrinks
                yield emit()
        how = describe(found)
        if scene:
            how += ("; " if how else "") + "scene mark(s) tracked: " + describe(scene)
        if moving:
            mk = moving[0]
            how += ("; " if how else "") + (f"[{mk['name']}] moving mark ~{mk['size']:.0f}px "
                                            f"(seen in {moving[1]}/{len(frames)} samples)")
    elif box:
        bm = box_mask(ref, box)
        fn_stream = lambda it: (inpaint_mask(f, bm) for f in it)
        how = "box inpaint"
    else:
        return "SKIP  no watermark detected"

    dst = dst.with_suffix(".mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", info["fps_str"], "-i", "-",
         "-i", str(src), "-map", "0:v:0", "-map", "1:a?",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
         "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(dst)],
        stdin=subprocess.PIPE)
    n = 0
    try:
        for f in fn_stream(video_frames(src, W, H)):
            writer.stdin.write(f.tobytes())
            n += 1
    finally:
        writer.stdin.close()
        rc = writer.wait()
    if rc != 0:
        return f"FAIL  ffmpeg exited with {rc}"
    counts = (stats or []) + [t.hits for t in (trackers if stats is not None else [])]
    cover = "" if not counts else " (mark on " + ", ".join(f"{s}/{n}" for s in counts) + " frames)"
    return f"OK    {how}, {n} frames{cover}"


def _job(args):
    src, dst, kind, masks, box, force, touchup, crf = args
    try:
        if kind == "img":
            return src, process_image(src, dst, masks, box, force, touchup)
        return src, process_video(src, dst, masks, box, force, touchup, crf)
    except Exception as e:  # keep the batch going
        return src, f"FAIL  {e}"


def collect(inputs: list[str], out_dir: Path | None):
    jobs = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            base = out_dir or (p / "cleaned")
            for f in sorted(p.rglob("*")):
                if f.is_file() and "cleaned" not in f.relative_to(p).parts:
                    ext = f.suffix.lower()
                    if ext in IMG_EXT or ext in VID_EXT:
                        jobs.append((f, base / f.relative_to(p)))
        elif p.is_file():
            jobs.append((p, (out_dir or (p.parent / "cleaned")) / p.name))
        else:
            print(f"  not found: {s}")
    return jobs


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def cmd_calibrate(ns):
    masks_dir = Path(ns.masks)
    whites = ns.white or []
    for i, s in enumerate(ns.black):
        black = load_sample(Path(s))
        white = load_sample(Path(whites[i])) if i < len(whites) else None
        m = estimate_mask(black, white)
        p = save_mask(m, masks_dir, ns.name)
        h, w = m["alpha"].shape
        print(f"saved {p}  logo {w}x{h} at ({m['x']},{m['y']}) on {m['W']}x{m['H']}, "
              f"peak alpha {m['alpha'].max():.2f}, colour BGR {np.round(m['color']).astype(int).tolist()}")


def learn_group(items: list, masks_dir: Path, name: str) -> None:
    """Fit per-pixel alpha + colour of one mark shared by several clips (same position
    relative to the bottom-right corner) and save an exact mask for each resolution."""
    offx = float(np.median([W - 1 - mk["cx"] for _, W, H, mk in items]))
    offy = float(np.median([H - 1 - mk["cy"] for _, W, H, mk in items]))
    proto = {k: v for k, v in items[0][3].items() if k != "exact"}
    R = int(round(proto["size"] * 0.9))
    # support of the mark inside a 2R x 2R patch, generously dilated
    # mark centre in patch coordinates (patch origin = rounded centre - R + 1)
    a, rx, ry = render({**proto, "cx": R - 1 + round(offx) - offx, "cy": R - 1 + round(offy) - offy},
                       soft=False)
    sup = np.zeros((2 * R, 2 * R), np.uint8)
    ya, xa = max(ry, 0), max(rx, 0)
    yb, xb = min(ry + a.shape[0], 2 * R), min(rx + a.shape[1], 2 * R)
    sup[ya:yb, xa:xb] = (a[ya - ry:yb - ry, xa - rx:xb - rx] > 0.01)
    sup = cv2.dilate(sup, np.ones((7, 7), np.uint8))
    ring = (cv2.dilate(sup, np.ones((15, 15), np.uint8)) > 0) & (sup == 0)
    S, O = [], []
    for path, W, H, mk in items:
        cx, cy = int(round(W - 1 - offx)), int(round(H - 1 - offy))
        x0, y0 = cx - R + 1, cy - R + 1
        pr = Prepared(mk)
        used = 0
        for i, f in enumerate(video_frames(path, W, H)):
            if i % 2:
                continue
            if pr.inside(W, H) and pr.presence(f) > PRESENT:      # mark not in this frame
                continue
            p = f[y0:y0 + 2 * R, x0:x0 + 2 * R]
            g = p.astype(np.float32).mean(axis=2)
            gm = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
            rv = g[ring]
            if gm[ring].mean() > 12 or rv.min() < 1 or rv.max() > 250:
                continue                       # busy or clipped surroundings
            S.append(p.astype(np.float32))
            O.append(cv2.inpaint(p, sup * 255, 6, cv2.INPAINT_TELEA).astype(np.float32))
            used += 1
        print(f"  {path.name[:48]:48s} {W}x{H}  {used} usable frames")
    if len(S) < LEARN_MIN_FRAMES:
        print(f"  only {len(S)} usable frames (need {LEARN_MIN_FRAMES}) - add more clips; nothing saved")
        return
    S, O = np.stack(S), np.stack(O)
    Sg, Og = S.mean(axis=3), O.mean(axis=3)
    Z = np.array([s[ring].mean() for s in Sg])          # instrument: surroundings' brightness
    keep = np.ones(len(S), bool)
    for _ in range(3):                                   # per-pixel IV fit  S = k*O + b
        z = Z[keep][:, None, None] - Z[keep].mean()
        om, sm = Og[keep].mean(0), Sg[keep].mean(0)
        k = (z * (Sg[keep] - sm)).mean(0) / ((z * (Og[keep] - om)).mean(0) + 1e-3)
        b = sm - k * om
        res = Sg - (k * Og + b)
        sd = np.sqrt((res[keep] ** 2).mean(0)) + 1e-3
        keep = (np.abs(res) > 3 * sd).mean(axis=(1, 2)) <= 0.05
    alpha = np.clip(1 - k, 0, 1) * (sup > 0)
    alpha[alpha < 0.015] = 0
    core = alpha > 0.7 * alpha.max()
    color = np.float32([np.median(((S[keep][..., ch].mean(0) - (1 - alpha) * O[keep][..., ch].mean(0))[core])
                                  / alpha[core]) for ch in range(3)])
    color = np.clip(color, 0, 255)
    ys, xs = np.nonzero(alpha)
    ya, yb, xa, xb = max(ys.min() - 2, 0), ys.max() + 3, max(xs.min() - 2, 0), xs.max() + 3
    crop = alpha[ya:yb, xa:xb].astype(np.float32)
    print(f"  learned from {int(keep.sum())} frames: peak alpha {alpha.max():.2f}, "
          f"colour BGR {np.round(color).astype(int).tolist()}, fit residual {np.median(sd[sup > 0]):.1f}")
    sizes = {(W, H) for _, W, H, _ in items}
    if abs(offx - offy) < 1.0:              # same distance from right and bottom: the mark is
        sizes |= {(H, W) for W, H in sizes}  # identical in the rotated (portrait/landscape) format
    for W, H in sorted(sizes):
        x0, y0 = int(round(W - 1 - offx)) - R + 1, int(round(H - 1 - offy)) - R + 1
        p = save_mask({"alpha": crop, "x": x0 + xa, "y": y0 + ya, "W": W, "H": H, "color": color},
                      masks_dir, name)
        print(f"  saved {p}")


def cmd_learn(ns):
    masks = load_masks(Path(ns.masks))
    if not masks:
        sys.exit("need at least one calibrated mask first (it provides the logo shape)")
    files = [src for src, _ in collect(ns.inputs, None) if src.suffix.lower() in VID_EXT]
    if not files:
        sys.exit("no videos given")
    items = []
    for f in files:
        info = probe_video(f)
        fr = sample_frames(f, info, 15)
        found = find_marks(np.median(np.stack(fr), axis=0).astype(np.uint8), masks, fr, max_marks=2)
        if not found:
            print(f"  skip {f.name}: no mark found"); continue
        if len(found) > 1:
            print(f"  skip {f.name}: a second mark sits next to it"); continue
        items.append((f, info["w"], info["h"], found[0][0]))
    # group clips whose mark sits at the same offset from the corner with the same size
    groups = []
    for it in items:
        _, W, H, mk = it
        for g in groups:
            _, W2, H2, m2 = g[0]
            if (abs((W - mk["cx"]) - (W2 - m2["cx"])) < 3 and abs((H - mk["cy"]) - (H2 - m2["cy"])) < 3
                    and abs(mk["size"] - m2["size"]) < 0.06 * m2["size"]):
                g.append(it); break
        else:
            groups.append([it])
    for g in groups:
        _, W, H, mk = g[0]
        print(f"mark ~{mk['size']:.0f}px, {W - 1 - mk['cx']:.0f}px from right / {H - 1 - mk['cy']:.0f}px "
              f"from bottom: {len(g)} clip(s)")
        learn_group(g, Path(ns.masks), ns.name)


def cmd_masks(ns):
    ms = load_masks(Path(ns.masks))
    if not ms:
        print(f"no masks in {ns.masks} - run `calibrate` first")
    for m in ms:
        h, w = m["alpha"].shape
        print(f"{m['name']:>12}  {m['W']}x{m['H']}  logo {w}x{h} at ({m['x']},{m['y']})")


def cmd_clean(ns):
    masks = load_masks(Path(ns.masks), ns.mask)
    box = parse_box(ns.box) if ns.box else None
    if not masks and not box:
        sys.exit(f"No calibrated masks in {ns.masks} and no --box given.\n"
                 "Run `python marklift.py calibrate black.png` first (see README).")
    out_dir = Path(ns.output) if ns.output else None
    pairs = collect(ns.inputs, out_dir)
    if not pairs:
        sys.exit("nothing to process")
    jobs, skipped = [], 0
    for src, dst in pairs:
        kind = "vid" if src.suffix.lower() in VID_EXT else "img"
        final = dst.with_suffix(".mp4") if kind == "vid" else dst
        if final.exists() and not ns.overwrite:
            skipped += 1
            continue
        jobs.append((src, dst, kind, masks, box, ns.force, not ns.no_touchup, ns.crf))
    print(f"{len(jobs)} file(s) to process" + (f", {skipped} already done" if skipped else ""))

    imgs = [j for j in jobs if j[2] == "img"]
    vids = [j for j in jobs if j[2] == "vid"]
    results = []
    if imgs:
        workers = max(1, ns.workers)
        if workers == 1:
            for j in imgs:
                results.append(_job(j)); print(f"  {results[-1][1]}  {j[0].name}")
        else:
            with ProcessPoolExecutor(workers) as ex:
                for fut in as_completed([ex.submit(_job, j) for j in imgs]):
                    src, msg = fut.result(); results.append((src, msg))
                    print(f"  {msg}  {src.name}")
    for j in vids:  # ffmpeg is already multi-threaded
        print(f"  ... {j[0].name}", flush=True)
        results.append(_job(j)); print(f"  {results[-1][1]}  {j[0].name}")

    ok = sum(r[1].startswith("OK") for r in results)
    sk = sum(r[1].startswith("SKIP") for r in results)
    fl = sum(r[1].startswith("FAIL") for r in results)
    print(f"done: {ok} cleaned, {sk} without watermark, {fl} failed")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="marklift", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", default=str(DEFAULT_MASKS), help="mask folder (default: ./masks)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="learn the watermark from black sample(s)")
    c.add_argument("black", nargs="+", help="black image(s) or black video(s) carrying the mark, "
                                            "one per resolution you use")
    c.add_argument("--white", nargs="*", help="matching white sample(s), same order (optional, "
                                              "improves colour/alpha accuracy)")
    c.add_argument("--name", default="gemini", help="mask set name (e.g. gemini, veo)")
    c.set_defaults(fn=cmd_calibrate)

    k = sub.add_parser("clean", help="batch-remove the watermark")
    k.add_argument("inputs", nargs="+", help="files and/or folders (folders are scanned recursively)")
    k.add_argument("-o", "--output", help="output folder (default: <input folder>/cleaned)")
    k.add_argument("--mask", nargs="*", help="only use these mask set names")
    k.add_argument("--box", help="fallback without calibration: corner box WxH[+MX+MY] from "
                                 "bottom-right, e.g. 120x120+10+10")
    k.add_argument("-j", "--workers", type=int, default=4, help="parallel image workers (default 4)")
    k.add_argument("--crf", type=int, default=16, help="x264 quality for videos (lower = better)")
    k.add_argument("--force", action="store_true", help="remove even if detection is unsure")
    k.add_argument("--no-touchup", action="store_true", help="skip inpaint touch-up")
    k.add_argument("--overwrite", action="store_true", help="redo files already in output")
    k.set_defaults(fn=cmd_clean)

    n = sub.add_parser("learn", help="learn the exact mask from ordinary watermarked videos "
                                     "(no black sample needed)")
    n.add_argument("inputs", nargs="+", help="watermarked videos and/or folders; more varied clips = better")
    n.add_argument("--name", default="veo", help="mask set to save into (default veo)")
    n.set_defaults(fn=cmd_learn)

    l = sub.add_parser("masks", help="list calibrated masks")
    l.set_defaults(fn=cmd_masks)

    ns = ap.parse_args(argv)
    ns.fn(ns)


if __name__ == "__main__":
    main()
