import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def find_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path not found: {input_path}")

    files: list[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(input_path.rglob(f"*{ext}"))
    return sorted(files)


def load_pixels(path: Path, resize: int, max_pixels: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if resize > 0:
        img.thumbnail((resize, resize))
    arr = np.asarray(img)
    pixels = arr.reshape(-1, 3)
    if max_pixels > 0 and len(pixels) > max_pixels:
        idx = np.random.choice(len(pixels), size=max_pixels, replace=False)
        pixels = pixels[idx]
    return pixels


def rgb_to_hsv(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = pixels.astype(np.float32) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min

    hue = np.zeros_like(c_max)
    mask = delta > 1e-6
    r_mask = (c_max == r) & mask
    g_mask = (c_max == g) & mask
    b_mask = (c_max == b) & mask

    hue[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0
    hue[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
    hue[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0
    hue = (hue / 6.0) % 1.0

    sat = np.zeros_like(c_max)
    sat[c_max > 1e-6] = delta[c_max > 1e-6] / c_max[c_max > 1e-6]

    val = c_max
    return hue, sat, val


def circular_mean(values: np.ndarray) -> float:
    angles = values * 2.0 * np.pi
    sin_sum = np.sin(angles).mean()
    cos_sum = np.cos(angles).mean()
    if sin_sum == 0 and cos_sum == 0:
        return 0.0
    angle = np.arctan2(sin_sum, cos_sum)
    if angle < 0:
        angle += 2.0 * np.pi
    return angle / (2.0 * np.pi)


def hue_tone(h: float) -> str:
    if h < 1 / 12 or h >= 11 / 12:
        return "red"
    if h < 3 / 12:
        return "orange"
    if h < 5 / 12:
        return "yellow"
    if h < 7 / 12:
        return "green"
    if h < 9 / 12:
        return "cyan"
    if h < 11 / 12:
        return "blue"
    return "red"


def color_temperature(mean_r: float, mean_b: float) -> float:
    return (mean_r - mean_b) / 255.0


def compute_metrics(pixels: np.ndarray) -> dict:
    mean_rgb = pixels.mean(axis=0)
    mean_r, mean_g, mean_b = mean_rgb.tolist()

    luma = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    luma_mean = float(luma.mean())
    luma_std = float(luma.std())

    hue, sat, val = rgb_to_hsv(pixels)
    hue_mean = float(circular_mean(hue))
    sat_mean = float(sat.mean())
    sat_std = float(sat.std())
    val_mean = float(val.mean())
    val_std = float(val.std())

    rg = pixels[:, 0] - pixels[:, 1]
    yb = 0.5 * (pixels[:, 0] + pixels[:, 1]) - pixels[:, 2]
    rg_std = rg.std()
    yb_std = yb.std()
    rg_mean = rg.mean()
    yb_mean = yb.mean()
    colorfulness = float(np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2))

    vibrance = float(np.clip(sat_mean + sat_std, 0.0, 1.0))

    temp = color_temperature(mean_r, mean_b)
    temp_label = "neutral"
    if temp > 0.08:
        temp_label = "warm"
    elif temp < -0.08:
        temp_label = "cool"

    contrast_label = "low"
    if luma_std > 50:
        contrast_label = "high"
    elif luma_std > 25:
        contrast_label = "medium"

    saturation_label = "muted"
    if sat_mean > 0.55:
        saturation_label = "vivid"
    elif sat_mean > 0.35:
        saturation_label = "moderate"

    return {
        "mean_rgb": [float(mean_r), float(mean_g), float(mean_b)],
        "mean_luma": luma_mean,
        "contrast": luma_std,
        "brightness": val_mean,
        "brightness_std": val_std,
        "saturation": sat_mean,
        "saturation_std": sat_std,
        "vibrance": vibrance,
        "colorfulness": colorfulness,
        "hue_mean": hue_mean,
        "hue_tone": hue_tone(hue_mean),
        "temperature": temp,
        "temperature_label": temp_label,
        "contrast_label": contrast_label,
        "saturation_label": saturation_label,
    }


def compute_palette(pixels: np.ndarray, k: int, seed: int) -> list[dict]:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    counts = np.bincount(labels, minlength=k)
    total = counts.sum()

    order = np.argsort(-counts)
    palette: list[dict] = []
    for i in order:
        rgb = centers[i].round().astype(int).clip(0, 255)
        hex_code = "#%02x%02x%02x" % tuple(rgb.tolist())
        palette.append(
            {
                "hex": hex_code,
                "rgb": rgb.tolist(),
                "pct": float(counts[i]) / float(total) if total else 0.0,
            }
        )
    return palette


def window_indices(total: int, window_size: int, step: int) -> list[tuple[int, int]]:
    if window_size <= 0 or total <= 0:
        return []
    if step <= 0:
        step = window_size
    windows: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + window_size, total)
        windows.append((start, end))
        if end == total:
            break
        start += step
    return windows


def extract_palette(
    input_path: Path,
    k: int,
    resize: int,
    max_images: int,
    stride: int,
    max_pixels: int,
    sample_total: int,
    window_size: int,
    window_step: int,
    fps: float,
    seed: int,
) -> dict:
    images = find_images(input_path)
    if not images:
        raise ValueError(f"No images found under: {input_path}")

    if stride > 1:
        images = images[::stride]
    if max_images and len(images) > max_images:
        images = images[:max_images]

    frame_pixels: list[np.ndarray] = []
    for path in images:
        frame_pixels.append(load_pixels(path, resize, max_pixels))

    result = {
        "input": str(input_path),
        "num_images": len(images),
        "k": k,
    }

    windows = window_indices(len(images), window_size, window_step)
    if not windows:
        pixels = np.concatenate(frame_pixels, axis=0)
        if sample_total and len(pixels) > sample_total:
            idx = np.random.choice(len(pixels), size=sample_total, replace=False)
            pixels = pixels[idx]

        result["palette"] = compute_palette(pixels, k, seed)
        result["metrics"] = compute_metrics(pixels)
        return result

    window_results: list[dict] = []
    for idx, (start, end) in enumerate(windows):
        pixels = np.concatenate(frame_pixels[start:end], axis=0)
        if sample_total and len(pixels) > sample_total:
            sample_n = min(sample_total, len(pixels))
            pick = np.random.choice(len(pixels), size=sample_n, replace=False)
            pixels = pixels[pick]

        entry = {
            "window_index": idx,
            "frame_start": start,
            "frame_end": end - 1,
            "num_images": end - start,
            "palette": compute_palette(pixels, k, seed),
            "metrics": compute_metrics(pixels),
        }

        if fps and fps > 0:
            entry["time_start_sec"] = start / fps
            entry["time_end_sec"] = (end - 1) / fps

        window_results.append(entry)

    result["windows"] = window_results
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract dominant colors from images (frames) using KMeans."
    )
    parser.add_argument("--input", required=True, help="Image file or folder of frames")
    parser.add_argument("--k", type=int, default=5, help="Number of dominant colors")
    parser.add_argument(
        "--resize",
        type=int,
        default=200,
        help="Resize longest side to this for speed (0 disables)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit number of images (0 = all)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth image (1 = all)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=5000,
        help="Max pixels sampled per image (0 = all)",
    )
    parser.add_argument(
        "--sample-total",
        type=int,
        default=200000,
        help="Downsample total pixels before KMeans (0 = all)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="Group frames into windows of N images (0 = single global palette)",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=0,
        help="Step between windows in frames (0 = same as window-size)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Optional FPS for time-based labels in output",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--out", default="", help="Optional JSON output path")

    args = parser.parse_args()
    result = extract_palette(
        input_path=Path(args.input),
        k=args.k,
        resize=args.resize,
        max_images=args.max_images,
        stride=args.stride,
        max_pixels=args.max_pixels,
        sample_total=args.sample_total,
        window_size=args.window_size,
        window_step=args.window_step,
        fps=args.fps,
        seed=args.seed,
    )

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
