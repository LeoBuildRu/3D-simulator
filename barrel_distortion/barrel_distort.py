#!/usr/bin/env python3
"""Standalone barrel-distortion CLI.

Reproduces the radial warp from ``RendererUtils.barrel_distortion`` in
``src/rendering/renderer_utils.py`` (distortion factor ``1 + k1*r^2 + k2*r^4``)
as a self-contained script with no Panda3D dependency.

The warp is an inverse map: for every output pixel we compute a radius ``r``,
scale it by the distortion factor and sample the *source* image at that
radius. With positive k1/k2 this pulls the corners inward (pincushion-style
sampling that straightens barrel-bulged footage); negative values do the
opposite.

FOV handling
------------
By default the image is treated as having square pixels and the normalisation
matches the original code (``max_dist = min(cx, cy)``).

If you pass ``--hfov`` and ``--vfov`` the script works in angular (tangent)
space instead, so the radial distortion stays isotropic *in real-world angle*
even when the source frame is anamorphically squeezed. Example: a 3:4 scene
captured/stored squeezed into a 16:9 frame. Give the true horizontal and
vertical field of view of the original shot and the corners are distorted by
the correct angular amount rather than by raw pixel distance.

Usage
-----
    python barrel_distort.py in.png out.png --k1 0.15 --k2 0.35
    python barrel_distort.py in.png out.png --k1 0.15 --k2 0.35 \
        --hfov 90 --vfov 60
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("This script needs Pillow:  pip install pillow")


def barrel_distort(
    img: np.ndarray,
    k1: float,
    k2: float,
    hfov: float | None = None,
    vfov: float | None = None,
    bilinear: bool = True,
    fill=(0, 0, 0),
) -> np.ndarray:
    """Apply the radial (barrel) warp to an HxWxC uint8 array.

    Parameters
    ----------
    img : ndarray, shape (H, W, C)
    k1, k2 : float
        Radial distortion coefficients (same polynomial as the original code).
    hfov, vfov : float or None
        Horizontal / vertical field of view of the ORIGINAL shot, in degrees.
        When both are given the warp is computed in angular space so squeezed
        (non-square-pixel) frames are handled correctly. When omitted, the
        image is treated as square-pixel and the classic
        ``max_dist = min(cx, cy)`` normalisation is used.
    bilinear : bool
        Bilinear sampling (True) vs nearest neighbour (False). Use nearest for
        label / segmentation maps.
    fill : tuple
        Colour for out-of-bounds samples.
    """
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    cx = w / 2.0
    cy = h / 2.0

    # Output pixel grid.
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)

    if hfov is not None and vfov is not None:
        # Angular (tangent) space: normalise each axis by tan(fov/2). Equal
        # angles -> equal normalised distance regardless of pixel squeeze.
        tan_hx = math.tan(math.radians(hfov) / 2.0)
        tan_hy = math.tan(math.radians(vfov) / 2.0)
        nx = (xs - cx) / cx * tan_hx
        ny = (ys - cy) / cy * tan_hy

        r = np.sqrt(nx * nx + ny * ny)
        factor = 1.0 + k1 * r * r + k2 * (r ** 4)

        sx = nx * factor
        sy = ny * factor

        # Back to pixel coordinates.
        src_x = cx + sx / tan_hx * cx
        src_y = cy + sy / tan_hy * cy
    else:
        # Original square-pixel normalisation.
        max_dist = min(cx, cy)
        nx = (xs - cx) / max_dist
        ny = (ys - cy) / max_dist

        r = np.sqrt(nx * nx + ny * ny)
        factor = 1.0 + k1 * r * r + k2 * (r ** 4)

        src_x = cx + nx * factor * max_dist
        src_y = cy + ny * factor * max_dist

    out = np.empty((h, w, c), dtype=np.float64)
    fill_arr = np.asarray(fill, dtype=np.float64)[:c]

    if bilinear:
        x0 = np.floor(src_x).astype(np.int64)
        y0 = np.floor(src_y).astype(np.int64)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = src_x - x0
        wy = src_y - y0

        def gather(xi, yi):
            valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            xic = np.clip(xi, 0, w - 1)
            yic = np.clip(yi, 0, h - 1)
            vals = img[yic, xic].astype(np.float64)
            vals[~valid] = 0.0
            return vals, valid.astype(np.float64)[..., None]

        v00, m00 = gather(x0, y0)
        v10, m10 = gather(x1, y0)
        v01, m01 = gather(x0, y1)
        v11, m11 = gather(x1, y1)

        wx = wx[..., None]
        wy = wy[..., None]
        acc = (
            v00 * (1 - wx) * (1 - wy)
            + v10 * wx * (1 - wy)
            + v01 * (1 - wx) * wy
            + v11 * wx * wy
        )
        weight = (
            m00 * (1 - wx) * (1 - wy)
            + m10 * wx * (1 - wy)
            + m01 * (1 - wx) * wy
            + m11 * wx * wy
        )
        # Where the whole neighbourhood is out of bounds, use fill.
        oob = weight[..., 0] <= 1e-6
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(weight > 1e-6, acc / np.clip(weight, 1e-6, None), acc)
        out[oob] = fill_arr
    else:
        xi = np.round(src_x).astype(np.int64)
        yi = np.round(src_y).astype(np.int64)
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xic = np.clip(xi, 0, w - 1)
        yic = np.clip(yi, 0, h - 1)
        out = img[yic, xic].astype(np.float64)
        out[~valid] = fill_arr

    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Apply barrel (radial) distortion to an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Input image path")
    p.add_argument("output", help="Output image path")
    p.add_argument("--k1", type=float, default=0.15, help="Radial coefficient k1")
    p.add_argument("--k2", type=float, default=0.35, help="Radial coefficient k2")
    p.add_argument(
        "--hfov",
        type=float,
        default=None,
        help="Horizontal FOV of the original shot in degrees "
        "(enables angular / squeeze-aware mode; requires --vfov)",
    )
    p.add_argument(
        "--vfov",
        type=float,
        default=None,
        help="Vertical FOV of the original shot in degrees (requires --hfov)",
    )
    p.add_argument(
        "--nearest",
        action="store_true",
        help="Use nearest-neighbour sampling (for label/segmentation maps)",
    )
    args = p.parse_args()

    if (args.hfov is None) != (args.vfov is None):
        p.error("--hfov and --vfov must be given together")

    im = Image.open(args.input)
    mode = im.mode
    if mode not in ("RGB", "RGBA", "L"):
        im = im.convert("RGB")
        mode = "RGB"
    arr = np.asarray(im)
    if arr.ndim == 2:
        arr = arr[..., None]

    out = barrel_distort(
        arr,
        k1=args.k1,
        k2=args.k2,
        hfov=args.hfov,
        vfov=args.vfov,
        bilinear=not args.nearest,
        fill=(0,) * arr.shape[2],
    )

    if out.shape[2] == 1:
        out_im = Image.fromarray(out[..., 0], mode="L")
    else:
        out_im = Image.fromarray(out, mode=mode)
    out_im.save(args.output)

    fov_note = (
        f"  hfov={args.hfov} vfov={args.vfov} (angular mode)"
        if args.hfov is not None
        else "  (square-pixel mode)"
    )
    print(f"Wrote {args.output}  k1={args.k1} k2={args.k2}{fov_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
