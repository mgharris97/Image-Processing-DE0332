# Matthew Harris
# 241ADB166
# Homework Assignment 2
# Logarithmic correction: 𝐼𝑛𝑒𝑤 = 𝑐 ⋅ log(1 + 𝐼𝑜𝑙𝑑)
# Histogram equalization:
#   1. Calculate the cumulative distribution (CDF - Cumulative Distribution Function)
#   2. Normalize the values so that they cover the range [0, 255]
#   3. Replace the original pixel values with the new ones

import numpy as np
import cv2


def logarithmic_correction(img):
    A = img.astype(np.float32)
    c = 255 / np.log(1 + 255)  # scaling constant so output stays in 0-255 range
    corrected = c * np.log(1 + A)
    return corrected.astype(np.uint8)


def histogram_equalization(img):
    # split image into 3 color channels. Creates a tuple of 2D arrays for B, G, R (H, W)
    channels = cv2.split(img)
    equalized = []

    # loops through each of the 3 color channels one by one B -> G -> R
    for ch in channels:
        # hist stores the frequency of each intensity value (0-255)
        hist, bins = np.histogram(ch.flatten(), 256, [0, 256])

        # running tally of hist[i] values that reveals where pixels are concentrated
        cdf = hist.cumsum()

        # shift curve to start at 0, then normalize so max value becomes 255
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        cdf_normalized = cdf_normalized.astype(np.uint8)

        # replace each pixel's old intensity with its new equalized intensity
        equalized.append(cdf_normalized[ch])

    # merge all three layers back into a color image
    return cv2.merge(equalized)
