# Matthew Harris
# 241ADB166
# Homework Assignment 1
# Multiply blending mode

import numpy as np

def multiply(img1, img2):
    A = img1.astype(np.float32)
    B = img2.astype(np.float32)
    result = A * B / 255.0
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result
