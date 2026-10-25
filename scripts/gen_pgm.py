#!/usr/bin/env python3
"""gen_pgm.py – Generate synthetic grayscale PGM images for pipeline testing.

Each image contains a mix of geometric shapes (circles, rectangles, gradients)
to produce meaningful edge-detection output.

Usage:
    python3 scripts/gen_pgm.py --count 200 --width 512 --height 512 \
                               --output data/input
"""

import argparse
import math
import os
import random
import struct
import sys


def make_image(width: int, height: int, seed: int) -> bytes:
    """Generate a synthetic grayscale image as raw bytes."""
    rng = random.Random(seed)
    pixels = bytearray(width * height)

    # Background gradient
    for y in range(height):
        for x in range(width):
            pixels[y * width + x] = int(128 * x / width + 64 * y / height) % 256

    # Draw 5-10 filled circles
    n_circles = rng.randint(5, 10)
    for _ in range(n_circles):
        cx = rng.randint(0, width - 1)
        cy = rng.randint(0, height - 1)
        r  = rng.randint(10, min(width, height) // 4)
        val = rng.randint(0, 255)
        for y in range(max(0, cy - r), min(height, cy + r + 1)):
            for x in range(max(0, cx - r), min(width, cx + r + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    pixels[y * width + x] = val

    # Draw 3-6 rectangles
    n_rects = rng.randint(3, 6)
    for _ in range(n_rects):
        x0 = rng.randint(0, width - 2)
        y0 = rng.randint(0, height - 2)
        x1 = rng.randint(x0 + 1, width - 1)
        y1 = rng.randint(y0 + 1, height - 1)
        val = rng.randint(0, 255)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                pixels[y * width + x] = val

    return bytes(pixels)


def write_pgm(path: str, pixels: bytes, width: int, height: int) -> None:
    """Write a binary PGM (P5) file."""
    header = f"P5\n{width} {height}\n255\n".encode()
    with open(path, "wb") as f:
        f.write(header)
        f.write(pixels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count",  type=int, default=200,
                        help="Number of images to generate (default: 200)")
    parser.add_argument("--width",  type=int, default=512,
                        help="Image width in pixels (default: 512)")
    parser.add_argument("--height", type=int, default=512,
                        help="Image height in pixels (default: 512)")
    parser.add_argument("--output", default="data/input",
                        help="Output directory (default: data/input)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    digits = len(str(args.count))
    for i in range(1, args.count + 1):
        filename = f"image{str(i).zfill(digits)}.pgm"
        path = os.path.join(args.output, filename)
        pixels = make_image(args.width, args.height, seed=i)
        write_pgm(path, pixels, args.width, args.height)
        if i % 20 == 0 or i == args.count:
            print(f"  Generated {i}/{args.count} images...", flush=True)

    print(f"\nDone. {args.count} PGM images written to: {args.output}")


if __name__ == "__main__":
    main()
