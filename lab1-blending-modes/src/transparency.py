# Matthew Harris
# 241ADB166
# Homework Assignment 1
# Transparency / Opacity change blending mode

import numpy as np

def transparency(img1, img2, d=0.5):
    A = img1.astype(np.float32)
    B = img2.astype(np.float32)

    result = d * A + (1.0 - d) * B
    return np.clip(result, 0, 255).astype(np.uint8)
    
