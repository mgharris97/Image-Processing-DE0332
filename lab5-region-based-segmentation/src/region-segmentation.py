# Matthew Harris
# ID: 241ADB166
# Practical 5 - Region-based segmentation, thresholding, and clustering.

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# image paths
IMAGE_1_PATH = "/Users/Matt/Desktop/images/b&w_img1.jpg"    # Grayscale with multiple objects
IMAGE_2_PATH = "/Users/Matt/Desktop/images/img2.jpg"        # Image containing a person, animal, or a car
IMAGE_3_PATH = "/Users/Matt/Desktop/images/img3.jpg"        # Freely chosen image 

# K - means parameter
K = 3 # the number of clusters 

# ── Method 1: Gaussian Adaptive Thresholding ──────────────────────────────────
def gaussian_adaptive_threshold(gray: np.ndarray, block_size: int = 35, C: int = 5) -> np.ndarray:

    if block_size % 2 == 0:
        block_size += 1  # enforce odd
 
    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,   # object → 255, background → 0
        blockSize=block_size,
        C=C
    )
 
 
# ── Method 2: K-Means Clustering ─────────────────────────────────────────────
def kmeans_segment(rgb: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    pixels = rgb.reshape(-1, 3).astype(np.float32)
 
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,    # max iterations
        0.2     # epsilon
    )
 
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        attempts=10,
        flags=cv2.KMEANS_PP_CENTERS
    )
 
    centers = np.uint8(centers)
    segmented_pixels = centers[labels.flatten()]
    segmented = segmented_pixels.reshape(rgb.shape)
 
    labels_2d = labels.reshape(rgb.shape[:2])
    return segmented, labels_2d
 
 
# ── Histogram helper ──────────────────────────────────────────────────────────
def plot_histogram(ax: plt.Axes, gray: np.ndarray, title: str = "Histogram") -> None:
    ax.hist(gray.ravel(), bins=256, range=(0, 256), color="steelblue", edgecolor="none")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Intensity", fontsize=8)
    ax.set_ylabel("Pixel count", fontsize=8)
    ax.tick_params(labelsize=7)
 
 
# ── Main processing pipeline ──────────────────────────────────────────────────
def process(path: str, title: str, k: int = K) -> None:
   
    rgb  = np.array(Image.open(path).convert("RGB"))
    gray = np.array(Image.open(path).convert("L"))
 
    # ── apply methods ──
    gauss_thresh          = gaussian_adaptive_threshold(gray)
    kmeans_seg, label_map = kmeans_segment(rgb, k=k)
 
  
    lut = np.array([
        plt.cm.tab10(i / max(k - 1, 1))[:3]   # RGB floats in [0,1]
        for i in range(k)
    ])
    label_colour = (lut[label_map] * 255).astype(np.uint8)
 
    # ── figure ──
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
 
    panels = [
        (rgb,           "Original",                         None),
        (None,          "Grayscale Histogram",               None),  # handled separately
        (gauss_thresh,  f"Gaussian Adaptive\n(block=35, C=5)", "gray"),
        (kmeans_seg,    f"K-Means Segmented\n(k={k})",       None),
        (label_colour,  f"K-Means Label Map\n(k={k})",        None),
    ]
 
    for ax, (img, label, cmap) in zip(axes, panels):
        if label == "Grayscale Histogram":
            plot_histogram(ax, gray)
        else:
            ax.imshow(img, cmap=cmap)
            ax.set_title(label, fontsize=10)
        ax.axis("off") if label != "Grayscale Histogram" else None
 
    plt.tight_layout()
    plt.show()
 
 
if __name__ == "__main__":
    process(IMAGE_1_PATH, "Image 1 – Grayscale / Multiple Objects")
    process(IMAGE_2_PATH, "Image 2 – Person / Animal / Car")
    process(IMAGE_3_PATH, "Image 3 – Free Choice")