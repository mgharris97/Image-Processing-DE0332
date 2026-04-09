
# Matthew Harris
# 241ADB166
# Practical Assignment 4

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Image paths

IMAGE_1_PATH = "/Users/Matt/Desktop/imgs/clean.jpg"
IMAGE_2_PATH = "/Users/Matt/Desktop/imgs/noisy.jpg"
IMAGE_3_PATH = "/Users/Matt/Desktop/imgs/random.jpg"

# Canny (built-in permitted)
# function signature takes a grayscale image as a NumPy array and 2 threshold values 
# returns a NumPy array (the edge mask)
def canny_detect(gray: np.ndarray,
                 low_threshold: int = 50,
                 high_threshold: int = 150) -> np.ndarray:
    
    # Gaussian blur that smooths the image with a 5x5 gaussian filter before doing anything else. 
    # This is step 1 of the Canny (supress noise to avoid mistaking noise for edges later)
    # simgX of 1.4 is the standard value used in slides
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.4)
    
    # Runs the remaining three Canny steps internally 
    # aperture size = 3 uses a 3x3 sobel kernel internally
    return cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=3)



# Roberts operator (manual — no built-in functions)

def roberts_manual(gray: np.ndarray, threshold: int = 30) -> dict:
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

    return {"gradient_map": gradient_map}


# Visualization

def process(path: str, title: str):
    rgb  = np.array(Image.open(path).convert("RGB"))
    gray = np.array(Image.open(path).convert("L"))

    canny   = canny_detect(gray)
    roberts = roberts_manual(gray)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (img, label, cmap) in zip(axes, [
        (rgb,                      "Original",               None),
        (canny,                    "Canny",                  "gray"),
        (roberts["gradient_map"],  "Roberts - gradient map", "gray"),
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
