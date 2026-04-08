"""
Practical Work 4: Edge-Based Segmentation
Student ID: 241ADB166 (ends in 6)
Methods: Canny edge detector + Roberts operator (manual implementation)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ─────────────────────────────────────────────
# IMAGE PATHS — update these before running
# ─────────────────────────────────────────────
IMAGE_1_PATH = "image1_clean.jpg"
IMAGE_2_PATH = "image2_noisy.jpg"
IMAGE_3_PATH = "image3_free.jpg"


# ─────────────────────────────────────────────
# CANNY (built-in permitted)
# ─────────────────────────────────────────────

def canny_detect(gray: np.ndarray,
                 low_threshold: int = 50,
                 high_threshold: int = 150) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.4)
    return cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=3)


# ─────────────────────────────────────────────
# ROBERTS OPERATOR (manual — no built-in functions)
# ─────────────────────────────────────────────

def roberts_manual(gray: np.ndarray, threshold: int = 30) -> dict:
    """
    Kernels:
        Mx = [[ 1,  0],    My = [[ 0,  1],
              [ 0, -1]]          [-1,  0]]

    For each 2x2 patch (a b / c d):
        Gx = a - d
        Gy = b - c
        G  = sqrt(Gx^2 + Gy^2)
    """
    gray = gray.astype(np.float64)
    h, w = gray.shape

    Gx = np.zeros((h, w), dtype=np.float64)
    Gy = np.zeros((h, w), dtype=np.float64)

    for y in range(h - 1):
        for x in range(w - 1):
            a = gray[y,     x    ]
            b = gray[y,     x + 1]
            c = gray[y + 1, x    ]
            d = gray[y + 1, x + 1]
            Gx[y, x] = a - d
            Gy[y, x] = b - c

    G = np.sqrt(Gx ** 2 + Gy ** 2)

    g_max = G.max()
    gradient_map = (G / g_max * 255).astype(np.uint8) if g_max > 0 else G.astype(np.uint8)

    binary = np.zeros_like(gradient_map)
    binary[G > threshold] = 255

    return {"gradient_map": gradient_map, "binary": binary}


# ─────────────────────────────────────────────
# VISUALISATION & MAIN
# ─────────────────────────────────────────────

def process(path: str, title: str):
    rgb  = np.array(Image.open(path).convert("RGB"))
    gray = np.array(Image.open(path).convert("L"))

    canny   = canny_detect(gray)
    roberts = roberts_manual(gray)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (img, label, cmap) in zip(axes, [
        (rgb,                      "Original",               None),
        (canny,                    "Canny",                  "gray"),
        (roberts["gradient_map"],  "Roberts - gradient map", "gray"),
        (roberts["binary"],        "Roberts - binary",       "gray"),
    ]):
        ax.imshow(img, cmap=cmap)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    process(IMAGE_1_PATH, "Image 1 - Clean")
    process(IMAGE_2_PATH, "Image 2 - Noisy")
    process(IMAGE_3_PATH, "Image 3 - Free choice")
