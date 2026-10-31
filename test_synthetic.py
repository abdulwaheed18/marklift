"""Synthetic end-to-end test for MarkLift."""
import subprocess, sys
from pathlib import Path
import numpy as np, cv2

T = Path(__file__).parent / "t"
if T.exists():
    import shutil; shutil.rmtree(T)
(T / "in").mkdir(parents=True)
rng = np.random.default_rng(1)

def sparkle(size):
    y, x = np.mgrid[:size, :size].astype(np.float32)
    c = (size - 1) / 2; dx, dy = np.abs(x - c) / c, np.abs(y - c) / c
    star = (np.sqrt(dx) + np.sqrt(dy)) <= 1.0        # 4-point astroid
    a = cv2.GaussianBlur(star.astype(np.float32), (0, 0), size / 40) * 0.55
    return a

def blend(img, a, margin, color=(255, 255, 255)):
    H, W = img.shape[:2]; s = a.shape[0]
    x, y = W - margin - s, H - margin - s
    out = img.astype(np.float32)
    p = out[y:y+s, x:x+s]
    out[y:y+s, x:x+s] = a[..., None] * np.array(color, np.float32) + (1 - a[..., None]) * p
    return np.clip(np.round(out), 0, 255).astype(np.uint8), (x, y, s)

def scene(W, H, seed):
    r = np.random.default_rng(seed)
    base = cv2.resize(r.integers(0, 256, (H // 64 + 2, W // 64 + 2, 3)).astype(np.uint8),
                      (W, H), interpolation=cv2.INTER_CUBIC)
    tex = r.normal(0, 12, (H, W, 3))
    return np.clip(base + tex, 0, 255).astype(np.uint8)

def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 99 if mse == 0 else 10 * np.log10(255**2 / mse)

# calibration samples (1024 tier: 48px logo, 32px margin; big tier: 96/64)
for W, H, s, mg in [(1024, 1024, 48, 32)]:
    blk, _ = blend(np.full((H, W, 3), 2, np.uint8), sparkle(s), mg)
    wht, _ = blend(np.full((H, W, 3), 255, np.uint8), sparkle(s), mg)
    cv2.imwrite(str(T / f"black_{W}x{H}.png"), blk); cv2.imwrite(str(T / f"white_{W}x{H}.png"), wht)

truth = {}
cases = [("a.png", 1024, 1024, 48, 32), ("b.png", 1024, 1024, 48, 32), ("c.jpg", 1024, 1024, 48, 32),
         ("big.png", 2048, 2048, 96, 64), ("clean.png", 1024, 1024, None, None)]
for i, (n, W, H, s, mg) in enumerate(cases):
    orig = scene(W, H, i)
    if s:
        wm, box = blend(orig, sparkle(s), mg)
    else:
        wm, box = orig, None
    cv2.imwrite(str(T / "in" / n), wm, [cv2.IMWRITE_JPEG_QUALITY, 95])
    truth[n] = (orig, box, wm)

# video 1280x720, logo 48/32, moving content
vw, vh, s, mg = 1280, 720, 48, 32
big = scene(vw * 2, vh, 99)
writer = cv2.VideoWriter(str(T / "src.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 24, (vw, vh))
vid_orig = []
for f in range(48):
    fr = np.ascontiguousarray(big[:, f * 10: f * 10 + vw])
    vid_orig.append(fr); writer.write(blend(fr, sparkle(s), mg)[0])
writer.release()
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(T / "src.avi"), "-f", "lavfi", "-i",
                "sine=f=440:d=2", "-shortest", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
                "-c:a", "aac", str(T / "in" / "clip.mp4")], check=True)
# black video for video calibration at 1280x720
bw, _ = blend(np.full((vh, vw, 3), 3, np.uint8), sparkle(s), mg)
cv2.imwrite(str(T / "black_1280x720.png"), bw)

py = [sys.executable, str(Path(__file__).parent / "marklift.py"), "--masks", str(T / "masks")]
subprocess.run(py + ["calibrate", str(T / "black_1024x1024.png"), "--white", str(T / "white_1024x1024.png")], check=True)
subprocess.run(py + ["calibrate", str(T / "black_1280x720.png"), "--name", "veo"], check=True)
subprocess.run(py + ["masks"], check=True)
subprocess.run(py + ["clean", str(T / "in"), "-j", "3"], check=True)

print("\nresults (PSNR of the logo patch, before -> after):")
for n, (orig, box, wm) in truth.items():
    out_p = T / "in" / "cleaned" / n
    if not out_p.exists():
        print(f"  {n}: no output"); continue
    out = cv2.imread(str(out_p))
    if box:
        x, y, s = box; p = lambda im: im[y:y+s, x:x+s]
        print(f"  {n}: {psnr(p(orig), p(wm)):.1f} dB -> {psnr(p(orig), p(out)):.1f} dB")
    else:
        print(f"  {n}: unchanged={np.array_equal(out, wm)}")

cap = cv2.VideoCapture(str(T / "in" / "cleaned" / "clip.mp4")); ps_b, ps_a, k = [], [], 0
src = cv2.VideoCapture(str(T / "in" / "clip.mp4"))
while True:
    ok, fr = cap.read(); ok2, fw = src.read()
    if not ok or not ok2: break
    x, y = vw - mg - s, vh - mg - s; p = lambda im: im[y:y+s, x:x+s]
    ps_b.append(psnr(p(vid_orig[k]), p(fw))); ps_a.append(psnr(p(vid_orig[k]), p(fr))); k += 1
print(f"  clip.mp4: {k} frames, {np.mean(ps_b):.1f} dB -> {np.mean(ps_a):.1f} dB")
print(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                      str(T / "in" / "cleaned" / "clip.mp4")], capture_output=True, text=True).stdout)
