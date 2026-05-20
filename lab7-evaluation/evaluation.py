# Matthew Harris
# 241ADB166
# Lab 7 Evaluation
# Matthew Harris
# 241ADB166
# Practical Assignment 7 - Evaluation of image quality and image processing algorithms
# Based on Practical Work 6 (contrast enhancement via parallel algorithm combination)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import os

# ── Image paths ──────────────────────────────────────────────────────────────
IMG1 = "/Users/Matt/Desktop/imgs/under-exposed.jpg"
IMG2 = "/Users/Matt/Desktop/imgs/hazy.jpg"
IMG3 = "/Users/Matt/Desktop/imgs/well-lit.jpg"


# ── Reused from PW6: Logarithmic correction ───────────────────────────────────
def logarithmic_correction(img: np.ndarray) -> np.ndarray:
    A = img.astype(np.float32)
    c = 255 / np.log(1 + 255)
    corrected = c * np.log(1 + A)
    return corrected.astype(np.uint8)


# ── Reused from PW6: Histogram equalization (per-channel) ────────────────────
def histogram_equalization(img: np.ndarray) -> np.ndarray:
    channels = cv2.split(img)
    equalized = []
    for ch in channels:
        hist, _ = np.histogram(ch.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        cdf_normalized = cdf_normalized.astype(np.uint8)
        equalized.append(cdf_normalized[ch])
    return cv2.merge(equalized)


# ── Reused from PW6: Transparency blend ──────────────────────────────────────
def transparency(img1: np.ndarray, img2: np.ndarray, d: float = 0.5) -> np.ndarray:
    A = img1.astype(np.float32)
    B = img2.astype(np.float32)
    result = d * A + (1.0 - d) * B
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Reused from PW6: Combined enhancement pipeline ───────────────────────────
def enhance_contrast(rgb: np.ndarray, d: float = 0.5) -> dict:
    log_result    = logarithmic_correction(rgb)
    histeq_result = histogram_equalization(rgb)
    combined      = transparency(log_result, histeq_result, d=d)
    return {
        "log":      log_result,
        "histeq":   histeq_result,
        "combined": combined,
    }


# ── NEW (PW7): Degradation functions ─────────────────────────────────────────

def add_gaussian_noise(img: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
    """
    Add Gaussian noise to simulate a degraded / low-light capture.
    sigma controls noise intensity — higher sigma = more noise.
    """
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def reduce_contrast(img: np.ndarray, factor: float = 0.4) -> np.ndarray:
    """
    Compress the dynamic range toward 128 (mid-gray).
    factor < 1 produces a flat, hazy look.
    result = factor * img + (1 - factor) * 128
    """
    compressed = factor * img.astype(np.float32) + (1.0 - factor) * 128
    return np.clip(compressed, 0, 255).astype(np.uint8)


def underexpose(img: np.ndarray, gamma: float = 2.5) -> np.ndarray:
    """
    Apply inverse gamma correction to simulate underexposure.
    gamma > 1 darkens the image.
    result = (pixel / 255) ^ gamma * 255
    """
    normalized = img.astype(np.float32) / 255.0
    darkened   = np.power(normalized, gamma) * 255.0
    return np.clip(darkened, 0, 255).astype(np.uint8)


# ── NEW (PW7): Objective quality metrics ──────────────────────────────────────

def compute_mse(reference: np.ndarray, processed: np.ndarray) -> float:
    """
    Mean Squared Error — average squared per-pixel intensity difference.
    Lower is better. 0 means the images are identical.
    """
    diff = reference.astype(np.float64) - processed.astype(np.float64)
    return float(np.mean(diff ** 2))


def compute_psnr(reference: np.ndarray, processed: np.ndarray, max_val: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio — expresses MSE on a logarithmic dB scale.
    Higher is better. Typical range: 20–50 dB; > 30 dB generally looks good.
    PSNR = 10 * log10(MAX^2 / MSE)
    Returns infinity if the images are identical (MSE = 0).
    """
    mse = compute_mse(reference, processed)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((max_val ** 2) / mse))


def compute_ssim(reference: np.ndarray, processed: np.ndarray) -> float:
    """
    Structural Similarity Index — measures luminance, contrast, and structure
    simultaneously. Range: [-1, 1], where 1 means identical images.
    Computed per channel and averaged, matching human perception better than MSE.
    channel_axis=2 tells skimage that the last axis is color channels (R, G, B).
    data_range=255 sets the expected pixel value range for the normalization step.
    """
    return float(ssim(reference, processed, channel_axis=2, data_range=255))


def compute_all_metrics(reference: np.ndarray, processed: np.ndarray) -> dict:
    return {
        "MSE":  compute_mse(reference, processed),
        "PSNR": compute_psnr(reference, processed),
        "SSIM": compute_ssim(reference, processed),
    }


# ── NEW (PW7): Per-image experiment ──────────────────────────────────────────

def run_experiment(path: str, title: str, degradation: str, d: float = 0.5) -> dict:
    """
    1. Load original (reference).
    2. Produce degraded version using the specified degradation type.
    3. Apply the PW6 combined enhancement pipeline to the degraded image.
    4. Compute metrics: degraded vs reference, enhanced vs reference.
    5. Return images + metrics for visualization.

    degradation options: "noise", "contrast", "underexpose"
    """
    reference = np.array(Image.open(path).convert("RGB"))

    # Degrade
    if degradation == "noise":
        degraded = add_gaussian_noise(reference, sigma=25)
    elif degradation == "contrast":
        degraded = reduce_contrast(reference, factor=0.4)
    elif degradation == "underexpose":
        degraded = underexpose(reference, gamma=2.5)
    else:
        raise ValueError(f"Unknown degradation type: {degradation}")

    # Enhance degraded image with PW6 pipeline
    enhanced = enhance_contrast(degraded, d=d)["combined"]

    # Metrics
    metrics_degraded  = compute_all_metrics(reference, degraded)
    metrics_enhanced  = compute_all_metrics(reference, enhanced)

    return {
        "title":             title,
        "degradation":       degradation,
        "d":                 d,
        "reference":         reference,
        "degraded":          degraded,
        "enhanced":          enhanced,
        "metrics_degraded":  metrics_degraded,
        "metrics_enhanced":  metrics_enhanced,
    }


# ── NEW (PW7): Visualization ───────────────────────────────────────────────

def visualize_experiment(result: dict) -> None:
    ref  = result["reference"]
    deg  = result["degraded"]
    enh  = result["enhanced"]
    md   = result["metrics_degraded"]
    me   = result["metrics_enhanced"]
    title = result["title"]
    d    = result["d"]
    deg_type = result["degradation"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{title}  |  degradation: {deg_type}  |  blend d={d}",
                 fontsize=13, fontweight="bold")

    panels = [
        (ref, "Reference (original)"),
        (deg, (
            f"Degraded\n"
            f"MSE={md['MSE']:.1f}  PSNR={md['PSNR']:.2f} dB\n"
            f"SSIM={md['SSIM']:.4f}"
        )),
        (enh, (
            f"Enhanced (PW6 pipeline)\n"
            f"MSE={me['MSE']:.1f}  PSNR={me['PSNR']:.2f} dB\n"
            f"SSIM={me['SSIM']:.4f}"
        )),
    ]

    for ax, (img, label) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.tight_layout()


def print_metrics_table(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print(f"{'Image':<30} {'Stage':<12} {'MSE':>10} {'PSNR (dB)':>12} {'SSIM':>8}")
    print("=" * 80)
    for r in results:
        name = r["title"][:29]
        for stage, metrics in [("Degraded", r["metrics_degraded"]),
                                ("Enhanced", r["metrics_enhanced"])]:
            psnr_str = f"{metrics['PSNR']:.2f}" if metrics['PSNR'] != float('inf') else "  inf"
            print(f"{name:<30} {stage:<12} {metrics['MSE']:>10.2f} {psnr_str:>12} {metrics['SSIM']:>8.4f}")
        print("-" * 80)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runs = [
        # (path,  title,                          degradation,   d)
        (IMG1, "Image 1 - Underexposed",        "underexpose", 0.4),
        (IMG2, "Image 2 - Hazy / low contrast", "contrast",    0.3),
        (IMG3, "Image 3 - Well-lit (control)",  "noise",       0.5),
    ]

    results = []
    for path, title, degradation, d in runs:
        if not os.path.exists(path):
            print(f"[SKIP] File not found: {path}")
            continue
        result = run_experiment(path, title, degradation, d=d)
        results.append(result)
        visualize_experiment(result)

    if results:
        print_metrics_table(results)

    plt.show()