# Matthew Harris
# 241ADB166
# Homework Assignment 2

import cv2
import matplotlib.pyplot as plt
from contrast_correction import logarithmic_correction, histogram_equalization

underexposed = cv2.imread("/Users/Matt/Desktop/Lab_2_Images/Underexposed.jpg")
overexposed = cv2.imread("/Users/Matt/Desktop/Lab_2_Images/Overexposed.jpg")
grayish = cv2.imread("/Users/Matt/Desktop/Lab_2_Images/Grayish.jpg")

images = {
    "Underexposed": underexposed,
    "Overexposed":  overexposed,
    "Grayish":      grayish,
}

for name, img in images.items():
    log_corrected  = logarithmic_correction(img)
    hist_equalized = histogram_equalization(img)

    # ── Plot original + corrected images ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{name} — Images", fontsize=14)

    for ax, image, title in zip(axes,
                                [img, log_corrected, hist_equalized],
                                ["Original", "Logarithmic", "Histogram Eq."]):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"output/{name.lower()}_images.png")
    plt.show()

    # ── Plot histograms ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{name} — Histograms", fontsize=14)

    for ax, image, title in zip(axes,
                                [img, log_corrected, hist_equalized],
                                ["Original", "Logarithmic", "Histogram Eq."]):
        for i, color in enumerate(["b", "g", "r"]):
            hist_vals = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist_vals, color=color)
        ax.set_title(title)
        ax.set_xlim([0, 256])

    plt.tight_layout()
    plt.savefig(f"output/{name.lower()}_histograms.png")
    plt.show()