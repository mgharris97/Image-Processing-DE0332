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
# Reference images (clean originals)
REF1 = "/Users/Matt/Desktop/imgs/well-lit.jpg"
REF2 = "/Users/Matt/Desktop/imgs/hazy.jpg"
REF3 = "/Users/Matt/Desktop/imgs/under-exposed.jpg"

# Degraded versions (prepared in Lightroom)
DEG1 = "/Users/Matt/Desktop/imgs/well-lit-grain.jpg"
DEG2 = "/Users/Matt/Desktop/imgs/hazy-underexposed.jpg"
DEG3 = "/Users/Matt/Desktop/imgs/under-exposed-decontrast.jpg"


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
def enhance_contrast(rgb: np.ndarray, d: float = 0.5) -> np.ndarray:
    log_result    = logarithmic_correction(rgb)
    histeq_result = histogram_equalization(rgb)
    combined      = transparency(log_result, histeq_result, d=d)
    return combined


# ── NEW (PW7): Objective quality metrics ─────────────────────────────────────

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
    Higher is better. > 30 dB is generally considered good quality.
    PSNR = 10 * log10(MAX^2 / MSE)
    """
    mse = compute_mse(reference, processed)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((max_val ** 2) / mse))


def compute_ssim(reference: np.ndarray, processed: np.ndarray) -> float:
    """
    Structural Similarity Index — measures luminance, contrast, and structure
    simultaneously. Range: [-1, 1], where 1 means identical images.
    channel_axis=2 tells skimage the last axis is color channels (R, G, B).
    data_range=255 sets the expected pixel value range for normalization.
    """
    return float(ssim(reference, processed, channel_axis=2, data_range=255))


def compute_metrics(reference: np.ndarray, processed: np.ndarray) -> dict:
    return {
        "MSE":  compute_mse(reference, processed),
        "PSNR": compute_psnr(reference, processed),
        "SSIM": compute_ssim(reference, processed),
    }


# ── NEW (PW7): Per-image experiment ──────────────────────────────────────────

def run_experiment(ref_path: str, deg_path: str, title: str, d: float = 0.5) -> dict:
    """
    1. Load clean original as reference.
    2. Load pre-made degraded version.
    3. Resize degraded to match reference dimensions if needed.
    4. Run PW6 enhancement pipeline on the degraded image.
    5. Compute metrics for degraded vs reference and enhanced vs reference.
    """
    reference = np.array(Image.open(ref_path).convert("RGB"))
    degraded  = np.array(Image.open(deg_path).convert("RGB"))

    # Resize degraded to match reference if Lightroom export changed dimensions
    if degraded.shape != reference.shape:
        degraded = cv2.resize(degraded, (reference.shape[1], reference.shape[0]))

    enhanced = enhance_contrast(degraded, d=d)

    return {
        "title":            title,
        "d":                d,
        "reference":        reference,
        "degraded":         degraded,
        "enhanced":         enhanced,
        "metrics_degraded": compute_metrics(reference, degraded),
        "metrics_enhanced": compute_metrics(reference, enhanced),
    }


# ── NEW (PW7): Visualization ──────────────────────────────────────────────────

def visualize(result: dict) -> None:
    ref   = result["reference"]
    deg   = result["degraded"]
    enh   = result["enhanced"]
    md    = result["metrics_degraded"]
    me    = result["metrics_enhanced"]
    title = result["title"]
    d     = result["d"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{title}  |  blend d={d}", fontsize=13, fontweight="bold")

    panels = [
        (ref, "Reference (original)"),
        (deg, f"Degraded\nMean Sq. Err. (MSE)={md['MSE']:.1f}\nPeak Signal-to-Noise Ratio={md['PSNR']:.2f} dB\nStructural Similarity Index={md['SSIM']:.4f}"),
        (enh, f"Enhanced (PW6 pipeline)\nMean Sq. Err. (MSE)={me['MSE']:.1f}\nPeak Signal-to-Noise Ratio={me['PSNR']:.2f} dB\nStructural Similarity Index={me['SSIM']:.4f}"),
    ]

    for ax, (img, label) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.tight_layout()


def print_metrics_table(results: list) -> None:
    print("\n" + "=" * 80)
    print(f"{'Image':<30} {'Stage':<12} {'MSE':>10} {'PSNR (dB)':>12} {'SSIM':>8}")
    print("=" * 80)
    for r in results:
        name = r["title"][:29]
        for stage, metrics in [("Degraded", r["metrics_degraded"]),
                                ("Enhanced", r["metrics_enhanced"])]:
            psnr_str = f"{metrics['PSNR']:.2f}" if metrics['PSNR'] != float('inf') else "inf"
            print(f"{name:<30} {stage:<12} {metrics['MSE']:>10.2f} {psnr_str:>12} {metrics['SSIM']:>8.4f}")
        print("-" * 80)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runs = [
        # (reference path, degraded path, title, blend d)
        (REF1, DEG1, "Image 1 - Well-lit (grain added)",        0.5),
        (REF2, DEG2, "Image 2 - Hazy (underexposed further)",   0.3),
        (REF3, DEG3, "Image 3 - Underexposed (contrast cut)",   0.4),
    ]

    results = []
    for ref_path, deg_path, title, d in runs:
        if not os.path.exists(ref_path) or not os.path.exists(deg_path):
            print(f"[SKIP] Missing file for: {title}")
            continue
        result = run_experiment(ref_path, deg_path, title, d=d)
        results.append(result)
        visualize(result)

    if results:
        print_metrics_table(results)

    plt.show()