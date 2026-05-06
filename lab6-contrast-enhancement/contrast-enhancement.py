# Matthew Harris
# 241ADB166
# Practical Assignment 6 - Contrast enhancement via parallel algorithm combination

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ── Image paths ──────────────────────────────────────────────────────────────
IMG1 = "/Users/Matt/Desktop/imgs/IMG1 — Underexposed.jpg"   # Underexposed
IMG2 = "/Users/Matt/Desktop/imgs/IMG2 — Low-contrast.jpg"   # Low-contrast / hazy
IMG3 = "/Users/Matt/Desktop/imgs/IMG3 — Mixed lighting.jpg"   # Mixed lighting
IMG4 = "/Users/Matt/Desktop/imgs/IMG4 — Already high contrast.jpg"   # Already high contrast (control)
IMG5 = "/Users/Matt/Desktop/imgs/IMG5 — Noisy.jpg"   # Noisy / low-light


# ── Reused from HW2: Logarithmic correction ──────────────────────────────────
# I_new = c * log(1 + I_old), where c scales output back into [0, 255]
def logarithmic_correction(img: np.ndarray) -> np.ndarray:
    A = img.astype(np.float32)
    c = 255 / np.log(1 + 255)
    corrected = c * np.log(1 + A)
    return corrected.astype(np.uint8)


# ── Reused from HW2: Histogram equalization (per-channel) ────────────────────
# 1. Compute CDF for each channel
# 2. Normalize CDF to span [0, 255]
# 3. Use it as a lookup table to remap pixel intensities
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


# ── Reused from HW1: Transparency blend ──────────────────────────────────────
# d * A + (1 - d) * B  →  d closer to 1 favors A, d closer to 0 favors B
def transparency(img1: np.ndarray, img2: np.ndarray, d: float = 0.5) -> np.ndarray:
    A = img1.astype(np.float32)
    B = img2.astype(np.float32)
    result = d * A + (1.0 - d) * B
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Combined enhancement pipeline (parallel) ─────────────────────────────────
# Both algorithms run independently on the original image, then blend.
# d controls the contribution of each:
#   d → 1.0  favors log correction  (use for noisy / already-bright images)
#   d → 0.0  favors histogram eq.   (use for flat / hazy images)
#   d = 0.5  balanced default
def enhance_contrast(rgb: np.ndarray, d: float = 0.5) -> dict:
    log_result    = logarithmic_correction(rgb)
    histeq_result = histogram_equalization(rgb)
    combined      = transparency(log_result, histeq_result, d=d)
    return {
        "log":      log_result,
        "histeq":   histeq_result,
        "combined": combined,
    }


# ── Visualization ────────────────────────────────────────────────────────────
def process(path: str, title: str, d: float = 0.5) -> None:
    rgb = np.array(Image.open(path).convert("RGB"))
    out = enhance_contrast(rgb, d=d)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"{title}  (transparency = {d})", fontsize=13, fontweight="bold")

    panels = [
        (rgb,             "Original"),
        (out["log"],      "Log correction"),
        (out["histeq"],   "Histogram equalization"),
        (out["combined"], f"Combined (blend d={d})"),
    ]
    for ax, (img, label) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.tight_layout()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Per-image blend ratios — tune these after viewing initial results,
    # then record the final values used in the report.
    runs = [
        (IMG1, "Image 1 - Underexposed",            0.4),
        (IMG2, "Image 2 - Low-contrast / hazy",     0.3),
        (IMG3, "Image 3 - Mixed lighting",          0.5),
        (IMG4, "Image 4 - Already high contrast",   0.7),
        (IMG5, "Image 5 - Noisy / low-light",       0.7),
    ]
    for path, title, d in runs:
        process(path, title, d=d)
    plt.show()