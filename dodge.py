# Matthew Harris
# 241ADB166
# Homework Assignment 1
# Dodge / Color blending mode

import numpy as np

def dodge(img1, img2):
    A = img1.astype(np.float32)
    B = img2.astype(np.float32)

    result = np.where(
        A == 255,
        255,
        B * 255.0 / (255.0 - A)
    )
    return np.clip(result, 0, 255).astype(np.uint8)
