"""Benchmark: FaceSwap FPS on current GPU.

Measures three stages independently:
  1. Face detection only (buffalo_l)
  2. Face swap (detection + inswapper)
  3. Full pipeline with GFPGAN enhancement

Usage:
  python benchmark_faceswap.py [--frames 200] [--res 640x480]
  python benchmark_faceswap.py --frame-img /path/to/face.jpg
  python benchmark_faceswap.py --face-url https://example.com/face.jpg
"""

import argparse
import os
import subprocess
import time

import cv2
import numpy as np


DEFAULT_FACE_URL = (
    "https://static.toiimg.com/thumb/msid-124380235,imgsize-15708,"
    "width-400,resizemode-4/bryan-johnson-reveals-his-2-million-year-"
    "approach-to-longevity-by-avoiding-these-3-surprising-habits.jpg"
)


def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            text=True,
        ).strip()
        print(f"GPU: {out}")
    except Exception:
        print("nvidia-smi not available")


def download_face(url: str, dest: str) -> bool:
    """Download a face image from *url* to *dest*. Returns True on success."""
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r, open(dest, "wb") as f:
            f.write(r.read())
        return True
    except Exception as exc:
        print(f"Download failed: {exc}")
        return False


def load_frame(args, w: int, h: int) -> np.ndarray:
    """Return a (h, w, 3) BGR frame from --frame-img, --face-url, or default URL."""
    if args.frame_img:
        frame = cv2.imread(args.frame_img)
        if frame is None:
            raise SystemExit(f"ERROR: cannot read {args.frame_img}")
        print(f"Using provided image: {args.frame_img}")
        return cv2.resize(frame, (w, h))

    cache = "/tmp/bench_face.jpg"
    url = args.face_url or DEFAULT_FACE_URL
    if not os.path.exists(cache):
        print(f"Downloading face image...")
        if not download_face(url, cache):
            raise SystemExit("ERROR: could not download face image")
        print("Downloaded OK")

    frame = cv2.imread(cache)
    if frame is None:
        raise SystemExit(f"ERROR: downloaded file is not a valid image")
    print(f"Using face image from {cache}")
    return cv2.resize(frame, (w, h))


def benchmark_detection(swapper, frame, n_frames, warmup=20):
    for _ in range(warmup):
        swapper.face_app.get(frame)
    times = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        swapper.face_app.get(frame)
        times.append(time.perf_counter() - t0)
    n_faces = len(swapper.face_app.get(frame))
    return np.array(times), n_faces


def benchmark_swap(swapper, frame, source_face, n_frames, warmup=20):
    for _ in range(warmup):
        swapper.swap_with_source(frame, source_face)
    times = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        swapper.swap_with_source(frame, source_face)
        times.append(time.perf_counter() - t0)
    return np.array(times)


def benchmark_enhanced(swapper, frame, source_face, n_frames, warmup=20):
    swapper.enhance_enabled = True
    for _ in range(warmup):
        swapper.swap_with_source(frame, source_face)
    times = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        swapper.swap_with_source(frame, source_face)
        times.append(time.perf_counter() - t0)
    swapper.enhance_enabled = False
    return np.array(times)


def report(label, times):
    ms = times * 1000
    fps = 1.0 / times
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Frames:  {len(times)}")
    print(f"  Mean:    {ms.mean():.1f} ms  ({fps.mean():.1f} FPS)")
    print(f"  Median:  {np.median(ms):.1f} ms  ({np.median(fps):.1f} FPS)")
    print(f"  P95:     {np.percentile(ms, 95):.1f} ms")
    print(f"  P99:     {np.percentile(ms, 99):.1f} ms")
    print(f"  Min:     {ms.min():.1f} ms  ({fps.max():.1f} FPS)")
    print(f"  Max:     {ms.max():.1f} ms  ({fps.min():.1f} FPS)")
    print(f"  Std:     {ms.std():.1f} ms")


def main():
    parser = argparse.ArgumentParser(description="FaceSwap GPU Benchmark")
    parser.add_argument("--frames", type=int, default=200, help="frames per test")
    parser.add_argument("--res", type=str, default="640x480", help="WxH resolution")
    parser.add_argument("--frame-img", type=str, default=None,
                        help="path to a local image with face(s)")
    parser.add_argument("--face-url", type=str, default=None,
                        help="URL to download a face image from")
    parser.add_argument("--source", type=str, default=None,
                        help="path to source face image (default: use target frame)")
    parser.add_argument("--skip-enhance", action="store_true",
                        help="skip GFPGAN benchmark")
    args = parser.parse_args()

    w, h = map(int, args.res.split("x"))
    n = args.frames

    print("=" * 55)
    print("  FaceSwap Benchmark")
    print("=" * 55)
    get_gpu_info()
    print(f"Resolution: {w}x{h}")
    print(f"Frames per test: {n}")
    print()

    # Load model
    print("Loading FaceSwapper (CUDA)...")
    t0 = time.perf_counter()
    from core.face_swap import FaceSwapper
    swapper = FaceSwapper(device="cuda")
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # Load frame
    frame = load_frame(args, w, h)

    # Detect faces
    faces = swapper.face_app.get(frame)
    if faces:
        print(f"Detected {len(faces)} face(s) in frame")
    else:
        print("WARNING: No faces detected. Swap benchmark will be skipped.")
        print("TIP: provide a real face image with --frame-img or --face-url")

    # Source face
    if args.source:
        src_img = cv2.imread(args.source)
        if src_img is None:
            raise SystemExit(f"ERROR: cannot read source {args.source}")
        if not swapper.set_source(src_img):
            raise SystemExit("ERROR: no face detected in source image")
    elif faces:
        swapper.source_face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    else:
        swapper.source_face = None

    # --- 1. Detection only ---
    print("\n[1/3] Benchmarking face detection (buffalo_l)...")
    det_times, n_faces = benchmark_detection(swapper, frame, n)
    report(f"Face Detection ({n_faces} face(s), {w}x{h})", det_times)

    # --- 2. Detection + Swap ---
    if swapper.source_face is not None:
        print("\n[2/3] Benchmarking face swap (detection + inswapper)...")
        swap_times = benchmark_swap(swapper, frame, swapper.source_face, n)
        report(f"Face Swap ({n_faces} face(s), {w}x{h})", swap_times)
    else:
        print("\n[2/3] SKIPPED: no source face")
        swap_times = None

    # --- 3. Full pipeline with GFPGAN ---
    if not args.skip_enhance and swapper.source_face is not None:
        print("\n[3/3] Benchmarking full pipeline (swap + GFPGAN)...")
        try:
            enh_times = benchmark_enhanced(swapper, frame, swapper.source_face, n)
            report(f"Full Pipeline w/ GFPGAN ({n_faces} face(s), {w}x{h})", enh_times)
        except Exception as e:
            print(f"\n[3/3] GFPGAN benchmark failed: {e}")
            enh_times = None
    else:
        print("\n[3/3] SKIPPED")
        enh_times = None

    # Summary
    print(f"\n{'=' * 55}")
    print("  SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Detection:  {1 / det_times.mean():.1f} FPS  ({det_times.mean() * 1000:.1f} ms)")
    if swap_times is not None:
        print(f"  Swap:       {1 / swap_times.mean():.1f} FPS  ({swap_times.mean() * 1000:.1f} ms)")
    if enh_times is not None:
        print(f"  Enhanced:   {1 / enh_times.mean():.1f} FPS  ({enh_times.mean() * 1000:.1f} ms)")
    print()


if __name__ == "__main__":
    main()
