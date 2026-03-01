import sys
import os
import cv2
import argparse
import numpy as np

from src.dodge import dodge
from src.multiply import multiply
from src.linear_burn import linear_burn
from src.transparency import transparency

import os
import argparse
import cv2
import numpy as np

# Import your blend functions
from src.multiply import multiply
from src.linear_burn import linear_burn          # (your Linear Burn)
from src.dodge import dodge
from src.transparency import transparency  # make sure it takes (img1, img2, d)

BLENDS = {
    "multiply": multiply,
    "linear_burn": linear_burn,
    "dodge": dodge,
    "transparency": transparency,
}

def load_images(path1: str, path2: str):
    img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(path2, cv2.IMREAD_COLOR)

    if img1 is None:
        raise FileNotFoundError(f"Could not read image: {path1}")
    if img2 is None:
        raise FileNotFoundError(f"Could not read image: {path2}")

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

    return img1, img2

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Image blending modes (implemented with NumPy formulas)")
    parser.add_argument("--img1", required=True, help="Path to first image (A)")
    parser.add_argument("--img2", required=True, help="Path to second image (B)")
    parser.add_argument("--mode", choices=BLENDS.keys(), default="multiply", help="Blending mode")
    parser.add_argument("--d", type=float, default=0.5, help="Transparency factor for 'transparency' mode (0..1)")
    parser.add_argument("--out", default="Results", help="Output directory")
    args = parser.parse_args()

    if not (0.0 <= args.d <= 1.0):
        raise ValueError("--d (transparency factor) must be between 0 and 1")

    img1, img2 = load_images(args.img1, args.img2)
    ensure_dir(args.out)

    if args.mode == "transparency":
        result = transparency(img1, img2, args.d)
        out_name = f"transparency_d{args.d:.2f}.png"
    else:
        result = BLENDS[args.mode](img1, img2)
        out_name = f"{args.mode}.png"

    out_path = os.path.join(args.out, out_name)
    cv2.imwrite(out_path, result)

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()