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
 
# K-means parameter
K = 3  # number of clusters
 
 
# ── Method 1: Gaussian Adaptive Thresholding ──────────────────────────────────
# Takes grayscale imaghe as numpy array, a neighborhood block size, and a constant C default = 5
def gaussian_adaptive_threshold(gray: np.ndarray, block_size: int = 35, C: int = 5) -> np.ndarray:

    # openCV requires block sizes to be odd.
    if block_size % 2 == 0:
        block_size += 1  # enforce odd
 
    return cv2.adaptiveThreshold(
        gray,                                                   # input graysclae image 
        maxValue=255,                                           # max pixel thershold
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,          # use gaussan -weighted neghborhood average
        thresholdType=cv2.THRESH_BINARY_INV,                    # inver pixels: foreground -> white background -> black
        blockSize=block_size,                                   # size of each pixel neighborhood used to compute local thershold
        C=C                                                     # subtracted from the weighted mean to fine-tune cutoff 
    )

 
 
# ── Method 2: K-Means Clustering ─────────────────────────────────────────────

# input image as numpy array, 
# k: the number of clusters to segment the image into
# Return type hint saying function returns a tuple containing two numpy arrays
def kmeans_segment(rgb: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:

    # flatten the image into a signle list of RGB triplets
    pixels = rgb.reshape(-1, 3).astype(np.float32)
 
    # when k-means should stop refining. Either after 100 iterations or when cluster centers are moving less than 0.2 between steps 
    # whichever comes first
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,   # max iterations
        0.2    # epsilon
    )
    
    # Run K-Means 
    # Group all pixels into k clusters based on color similarity
    # Each pixel gets assigned a lebl (0 to k-1) indicating its cluster 
    _, labels, centers = cv2.kmeans(
        pixels,                         # the data to cluster, flattened list of RGB triplets from above
        k,                              # how many clusters (colors) to group pixels into
        None,                           # tells open CV to handle intitialization by itself 
        criteria,                       # Stopping condition for refining defined earlier. Stop after 100 iterations or when movement drops below 0.2
        attempts=10,                    # Run 10 attempts with different starting points since a bad starting point can lead to poor results
        flags=cv2.KMEANS_PP_CENTERS     # K-Means++ picks the initial cluster centers in a smarter way than pure random
    )
 
    centers = np.uint8(centers)                         # convert cluster center colors from decimal points to standard 8-bit ints
    segmented_pixels = centers[labels.flatten()]        # Recoloring step. labels holds cluster ID for every pixel. Using those IDs as indices into centers fetches the corresponding color for each pixel.  
    segmented = segmented_pixels.reshape(rgb.shape)     # Reshapes the flattened list back into the original image dimensions (height, width, 3)
 
    labels_2d = labels.reshape(rgb.shape[:2])           # creates a 2D grid where each cell holds the cluster ID of a pixel
    return segmented, labels_2d                         # Return both outputs. The recolored simplified image and the 2D map of cluster IDs
    
 
# ── Main processing pipeline ──────────────────────────────────────────────────
def process(path: str, title: str, k: int = K) -> None:
 
    rgb  = np.array(Image.open(path).convert("RGB"))
    gray = np.array(Image.open(path).convert("L"))
 
    gauss_thresh          = gaussian_adaptive_threshold(gray)
    kmeans_seg, _         = kmeans_segment(rgb, k=k)
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
 
    panels = [
        (rgb,          "Original",                            None),
        (gauss_thresh, f"Gaussian Adaptive\n(block=35, C=5)", "gray"),
        (kmeans_seg,   f"K-Means Segmented\n(k={k})",         None),
    ]
 
    for ax, (img, label, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
 
    plt.tight_layout()
 
 
if __name__ == "__main__":
    process(IMAGE_1_PATH, "Image 1 – Grayscale / Multiple Objects")
    process(IMAGE_2_PATH, "Image 2 – Person / Animal / Car")
    process(IMAGE_3_PATH, "Image 3 – Free Choice")
    plt.show()