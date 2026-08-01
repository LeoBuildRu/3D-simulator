# barrel_distort

Standalone CLI for applying barrel (radial) distortion to an image.

Extracted from `RendererUtils.barrel_distortion` in
`src/rendering/renderer_utils.py`, with no Panda3D dependency — it uses
`numpy` + `Pillow` and is fully vectorized.

## Install

```bash
pip install numpy pillow
```

## How it works

An inverse-map radial warp. For every output pixel a radius `r` is computed,
scaled by the distortion factor

```
factor = 1 + k1 * r^2 + k2 * r^4
```

and the *source* image is sampled at that scaled radius. Positive `k1`/`k2`
pull the corners inward (straightening barrel-bulged footage); negative values
push outward.

## Usage

```bash
python barrel_distort.py INPUT OUTPUT --k1 0.15 --k2 0.35
```

### Arguments

| Argument     | Default | Description                                                            |
|--------------|---------|------------------------------------------------------------------------|
| `input`      | —       | Input image path                                                       |
| `output`     | —       | Output image path                                                      |
| `--k1`       | `0.15`  | Radial distortion coefficient k1                                       |
| `--k2`       | `0.35`  | Radial distortion coefficient k2                                       |
| `--hfov`     | none    | Horizontal FOV of the original shot, in degrees (requires `--vfov`)    |
| `--vfov`     | none    | Vertical FOV of the original shot, in degrees (requires `--hfov`)      |
| `--nearest`  | off     | Nearest-neighbour sampling (use for label / segmentation / depth maps) |

Sampling is bilinear by default; pass `--nearest` for discrete maps so class
labels aren't blended.

## FOV / squeeze-aware mode

By default the image is treated as having **square pixels** and normalisation
matches the original code (`max_dist = min(cx, cy)`).

If you pass **both** `--hfov` and `--vfov`, the warp is computed in angular
(tangent) space instead: each axis is normalised by `tan(fov/2)`, so the radial
distortion stays isotropic **in real-world angle** even when the source frame
is anamorphically squeezed.

Example — a 3:4 scene captured/stored squeezed into a 16:9 frame. Give the true
field of view of the original shot and the corners are distorted by the correct
angular amount rather than by raw pixel distance:

```bash
python barrel_distort.py input.png output.png --k1 0.15 --k2 0.35 --hfov 90 --vfov 60
```

## Examples

```bash
# Classic square-pixel warp
python barrel_distort.py frame.png out.png --k1 0.15 --k2 0.35

# Squeeze-aware, using the original shot's FOV
python barrel_distort.py frame.png out.png --k1 0.15 --k2 0.35 --hfov 90 --vfov 60

# Segmentation mask (no blending)
python barrel_distort.py mask.png mask_out.png --k1 0.15 --k2 0.35 --nearest
```
