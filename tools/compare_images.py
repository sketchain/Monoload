"""Compare two images pixel by pixel.

    python tools/compare_images.py a.png b.png
"""
import sys

import numpy as np
from PIL import Image


def main(a, b):
    x = np.asarray(Image.open(a).convert("RGB")).astype(np.int16)
    y = np.asarray(Image.open(b).convert("RGB")).astype(np.int16)
    if x.shape != y.shape:
        print("DIFFERENT SIZE", x.shape, y.shape)
        return 1
    d = np.abs(x - y)
    n = int((d.max(axis=2) > 0).sum())
    print("identical" if n == 0 else "differs", "| pixels differing: {} / {} ({:.4f}%) | max abs diff {} | mean abs diff {:.5f}".format(
        n, d.shape[0] * d.shape[1], 100.0 * n / (d.shape[0] * d.shape[1]), int(d.max()), float(d.mean())))
    return 0 if n == 0 else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
