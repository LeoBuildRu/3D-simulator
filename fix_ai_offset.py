"""
Fix the small offset/zoom that the AI img2img step introduced.

Each `*_ai.png` was generated from its `*.png` original. The AI was told to keep
the camera and truck body fixed, but the whole frame ended up slightly shifted
and/or zoomed (~5%). We estimate a global similarity transform (translation +
uniform scale + tiny rotation) that maps the AI image back onto the original,
warp the AI image with it, and save as `*_ai_fix.png`.

The background and the pile inside the truck are completely repainted by the AI,
so they give no reliable correspondence. The *truck body* is the one structure
that stays geometrically fixed, and the dataset already tells us where it is: the
`*_seg.png` mask paints the truck body blue. We therefore restrict every step of
the estimation -- SIFT keypoints, ECC intensity alignment, and the final scoring
metric -- to a (dilated) truck-body mask. That is exactly the "distinct truck
body lines" cue, isolated from the parts the AI was free to change.

Pipeline (every stage is gated: it must measurably improve the alignment
metric or it is discarded, so the output can never be worse than the input):
  1. Global similarity (translation + zoom + rotation): grid search on masked
     gradient-NCC, SIFT+RANSAC and ECC candidates; the best scorer wins.
  2. Global affine refinement (anisotropic scale + shear) by coordinate
     descent on the same masked metric.
  3. Residual non-rigid correction: dense DIS optical flow computed only on
     trusted truck-body pixels (forward/backward-consistent, textured),
     averaged onto a coarse grid, smoothed, damped and hard-clamped to a few
     pixels. It gently bends the AI image back onto the render but is too
     smooth and too small to introduce the wavy over-warping older versions
     of this script produced.
All accepted warps are composed and applied to the AI image in a SINGLE
resampling pass (no quality loss from chained warps).
"""

import sys
import glob
import os
import cv2
import numpy as np

MASK_DILATE_FRAC = 0.06   # grow truck mask by ~6% of image to cover the offset
IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], np.float32)

# --- Correction guards -----------------------------------------------------
# Cap on how large a correction we allow. If the estimated warp would move,
# zoom or rotate the frame by more than this fraction, we do NOT correct the
# shot at all (leave the AI frame untouched). "move"  = max corner displacement
# relative to the shorter image side; "zoom" = |scale-1|; "rotate" is captured
# by the corner-displacement measure.
MAX_CORRECTION_FRAC = 0.20

# Only shots whose filesystem creation time is strictly AFTER this reference
# frame get corrected. Everything at or before it is left as-is (those shots
# came from a different AI model that did not introduce the offset). The cutoff
# is read from the reference file's own metadata, not from its name.
CUTOFF_REFERENCE = "r0006_vol0007.53_random_20260723_141356_127246_ai.png"


def _creation_time(path):
    """Filesystem creation time (birth time on Windows) as a float, or None."""
    try:
        return os.path.getctime(path)
    except OSError:
        return None


def _resolve_cutoff(any_target):
    """Locate the reference file next to a target being processed and return its
    creation time. Cached after the first successful lookup."""
    if _resolve_cutoff.value is not None:
        return _resolve_cutoff.value
    folder = any_target if os.path.isdir(any_target) else os.path.dirname(any_target)
    ref = os.path.join(folder or ".", CUTOFF_REFERENCE)
    _resolve_cutoff.value = _creation_time(ref)
    if _resolve_cutoff.value is None:
        print(f"  WARN: cutoff reference not found: {ref} -- nothing will be corrected")
    return _resolve_cutoff.value


_resolve_cutoff.value = None


def past_cutoff(path):
    """True if this shot should be corrected (its creation time is strictly
    after the cutoff reference's). Conservative (skip) if either is unknown."""
    cutoff = _resolve_cutoff(path)
    ct = _creation_time(path)
    if cutoff is None or ct is None:
        return False
    return ct > cutoff


def correction_too_large(M, flow, shape, frac=MAX_CORRECTION_FRAC):
    """True if the composed correction (global warp M + optional residual flow)
    moves, zooms or rotates the frame by more than `frac`. Such shots are left
    uncorrected rather than partially fixed."""
    h, w = shape
    s, _rot, _tx, _ty = decompose(M)
    if abs(s - 1.0) > frac:                       # zoom cap
        return True
    limit_px = frac * min(h, w)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(corners, to33(M).astype(np.float32)).reshape(-1, 2)
    if np.linalg.norm(out - corners.reshape(-1, 2), axis=1).max() > limit_px:  # move/rotate cap
        return True
    if flow is not None:                          # residual non-rigid cap
        _, mx, my = flow
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        if np.hypot(mx - xs, my - ys).max() > limit_px:
            return True
    return False


def truck_from_seg(seg_path, shape):
    """Tight binary mask (uint8 0/255) of the truck body (blue in the seg image),
    or None if no seg file / no blue found."""
    if not os.path.exists(seg_path):
        return None
    seg = cv2.imread(seg_path, cv2.IMREAD_COLOR)
    if seg is None:
        return None
    if (seg.shape[0], seg.shape[1]) != shape:
        seg = cv2.resize(seg, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    b, g, r = seg[..., 0].astype(int), seg[..., 1].astype(int), seg[..., 2].astype(int)
    blue = ((b > 120) & (b - r > 60) & (b - g > 60)).astype(np.uint8) * 255
    if blue.sum() < 255 * 500:
        return None
    return blue


def load_truck_mask(seg_path, shape):
    """Dilated truck-body mask so it still covers the truck after a few-percent
    shift. Returns None if no seg / no blue."""
    tight = truck_from_seg(seg_path, shape)
    if tight is None:
        return None
    k = int(round(MASK_DILATE_FRAC * max(shape)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(tight, ker)


def _resize(img, scale, nearest=False):
    if scale >= 1.0:
        return img
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)


def estimate_similarity(orig_gray, ai_gray, mask, max_dim=1600):
    """SIFT+RANSAC similarity mapping AI -> ORIG (full-res 2x3).
    Returns (M, n_inliers, n_matches) or (None, 0, n)."""
    h, w = orig_gray.shape
    scale = min(1.0, max_dim / max(h, w))
    osmall, asmall = _resize(orig_gray, scale), _resize(ai_gray, scale)
    msmall = _resize(mask, scale, nearest=True) if mask is not None else None

    sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.02)
    kp_o, des_o = sift.detectAndCompute(osmall, msmall)
    kp_a, des_a = sift.detectAndCompute(asmall, msmall)
    if des_o is None or des_a is None or len(kp_o) < 10 or len(kp_a) < 10:
        return None, 0, 0

    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_o, k=2)
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < 0.8 * n.distance]
    if len(good) < 8:
        return None, 0, len(good)

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_o = np.float32([kp_o[m.trainIdx].pt for m in good])
    M, inliers = cv2.estimateAffinePartial2D(
        pts_a, pts_o, method=cv2.RANSAC,
        ransacReprojThreshold=3.0, maxIters=5000, confidence=0.999)
    if M is None:
        return None, 0, len(good)
    n_in = int(inliers.sum()) if inliers is not None else 0
    if scale < 1.0:
        M = M.copy()
        M[:, 2] /= scale
    return M, n_in, len(good)


def ecc_refine(orig_gray, ai_gray, mask, M_init, max_dim=900):
    """Estimate/refine a similarity by ECC, restricted to the mask. Returns
    (M_fullres, cc) or (None, -1)."""
    h, w = orig_gray.shape
    scale = min(1.0, max_dim / max(h, w))
    osmall = cv2.GaussianBlur(_resize(orig_gray, scale), (0, 0), 2.0)
    asmall = cv2.GaussianBlur(_resize(ai_gray, scale), (0, 0), 2.0)
    msmall = _resize(mask, scale, nearest=True) if mask is not None else None

    warp = M_init.astype(np.float32).copy()
    warp[:, 2] *= scale
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(
            osmall, asmall, warp, cv2.MOTION_EUCLIDEAN, criteria, msmall, 5)
    except cv2.error:
        return None, -1.0
    warp = warp.copy()
    warp[:, 2] /= scale
    return warp, float(cc)


def align_score(orig_gray, ai_gray, mask, M, max_dim=900):
    """Masked NCC of blurred gradient magnitudes after warping AI by M.
    Only truck-body pixels count. Higher = better aligned."""
    h, w = orig_gray.shape
    warped = cv2.warpAffine(ai_gray, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    scale = min(1.0, max_dim / max(h, w))
    o = cv2.GaussianBlur(_resize(orig_gray, scale), (0, 0), 1.5).astype(np.float32)
    a = cv2.GaussianBlur(_resize(warped, scale), (0, 0), 1.5).astype(np.float32)
    go = cv2.magnitude(cv2.Sobel(o, cv2.CV_32F, 1, 0), cv2.Sobel(o, cv2.CV_32F, 0, 1))
    ga = cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
    if mask is not None:
        m = _resize(mask, scale, nearest=True) > 0
        go, ga = go[m], ga[m]
    else:
        go, ga = go.ravel(), ga.ravel()
    go = go - go.mean(); ga = ga - ga.mean()
    denom = np.sqrt((go * go).sum() * (ga * ga).sum()) + 1e-9
    return float((go * ga).sum() / denom)


def _compose(scale, rot_deg, tx, ty, cx, cy):
    """Similarity 2x3 that scales+rotates about (cx,cy) then translates by (tx,ty)."""
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return M


def search_transform(orig_gray, ai_gray, mask, max_dim=500, shift_frac=0.07):
    """Coarse-to-fine search for the similarity (AI->ORIG) that maximizes the
    masked gradient-NCC. Robust where SIFT/ECC diverge because it is a global
    grid search on the very signal we ultimately care about: truck-body edges.

    Returns (M_fullres, score) or (None, -1).
    """
    h, w = orig_gray.shape
    scale = min(1.0, max_dim / max(h, w))
    o = cv2.GaussianBlur(_resize(orig_gray, scale), (0, 0), 1.2).astype(np.float32)
    a = cv2.GaussianBlur(_resize(ai_gray, scale), (0, 0), 1.2).astype(np.float32)
    sh, sw = o.shape
    cx, cy = sw / 2.0, sh / 2.0

    def grad(img):
        return cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0),
                             cv2.Sobel(img, cv2.CV_32F, 0, 1))
    go = grad(o)
    ga = grad(a)  # warp this precomputed gradient image (fast approximation)

    if mask is not None:
        m = _resize(mask, scale, nearest=True) > 0
    else:
        m = np.ones((sh, sw), bool)
    idx = m
    go_m = go[idx]
    go_c = go_m - go_m.mean()
    go_norm = np.sqrt((go_c * go_c).sum()) + 1e-9

    def score(sc, rot, dx, dy):
        M = _compose(sc, rot, dx, dy, cx, cy)
        gw = cv2.warpAffine(ga, M, (sw, sh), flags=cv2.INTER_LINEAR)
        gm = gw[idx]
        gm = gm - gm.mean()
        return float((go_c * gm).sum() / (np.sqrt((gm * gm).sum()) + 1e-9) / go_norm)

    maxsh = shift_frac * sw
    best = (-2.0, 1.0, 0.0, 0.0, 0.0)  # score, scale, rot, dx, dy

    # Coarse grid.
    scales = [0.94, 0.97, 1.0, 1.03, 1.06]
    rots = [-2.0, 0.0, 2.0]
    n = 8
    for sc in scales:
        for rot in rots:
            for dx in np.linspace(-maxsh, maxsh, n):
                for dy in np.linspace(-maxsh, maxsh, n):
                    v = score(sc, rot, dx, dy)
                    if v > best[0]:
                        best = (v, sc, rot, dx, dy)

    # Two refinement passes shrinking the window around the current best.
    span_s, span_r, span_t = 0.03, 2.0, maxsh / 3.0
    for _ in range(2):
        _, bs, br, bx, by = best
        for sc in np.linspace(bs - span_s, bs + span_s, 5):
            for rot in np.linspace(br - span_r, br + span_r, 5):
                for dx in np.linspace(bx - span_t, bx + span_t, 5):
                    for dy in np.linspace(by - span_t, by + span_t, 5):
                        v = score(sc, rot, dx, dy)
                        if v > best[0]:
                            best = (v, sc, rot, dx, dy)
        span_s, span_r, span_t = span_s / 2, span_r / 2, span_t / 2

    bscore, bs, br, bx, by = best
    if bs <= 0:
        return None, -1.0
    # Rebuild transform in full-res coordinates.
    M = _compose(bs, br, bx / scale, by / scale, cx / scale, cy / scale)
    return M.astype(np.float32), bscore


def decompose(M):
    """(scale, rotation_deg, tx, ty) from a similarity 2x3 matrix."""
    a, b = M[0, 0], M[0, 1]
    return float(np.hypot(a, b)), float(np.degrees(np.arctan2(-b, a))), \
        float(M[0, 2]), float(M[1, 2])


def valid(M):
    if M is None:
        return False
    s, rot, _, _ = decompose(M)
    return 0.85 < s < 1.18 and abs(rot) < 8


# ---------------------------------------------------------------------------
# Generic warps (2x3 affine, 3x3 homography, or a dense flow field) so the
# pipeline can escalate from similarity -> affine -> homography -> non-rigid.
# ---------------------------------------------------------------------------

def to33(M):
    if M.shape == (3, 3):
        return M.astype(np.float32)
    return np.vstack([M, [0, 0, 1]]).astype(np.float32)


def _scale_mats(scale):
    D = np.diag([scale, scale, 1.0]).astype(np.float64)
    Di = np.diag([1.0 / scale, 1.0 / scale, 1.0]).astype(np.float64)
    return D, Di


def full_to_small(M_full, scale):
    D, Di = _scale_mats(scale)
    return (D @ to33(M_full).astype(np.float64) @ Di)


def small_to_full(M_small, scale):
    D, Di = _scale_mats(scale)
    return (Di @ to33(M_small).astype(np.float64) @ D)


def warp_gray(img, M, size, flags=cv2.INTER_LINEAR, border=cv2.BORDER_REPLICATE):
    """Warp with a 2x3 affine, a 3x3 homography, or a ('flow', mapx, mapy) tuple."""
    w, h = size
    if isinstance(M, tuple) and M[0] == "flow":
        return cv2.remap(img, M[1], M[2], flags, borderMode=border)
    if np.asarray(M).shape == (3, 3):
        return cv2.warpPerspective(img, np.asarray(M, np.float32), (w, h), flags=flags, borderMode=border)
    return cv2.warpAffine(img, np.asarray(M, np.float32), (w, h), flags=flags, borderMode=border)


def valid_generic(M, shape, max_disp_frac=0.16):
    """Accept any warp whose effect on the four image corners is a small, sane
    displacement (guards homography/affine from folding or blowing up)."""
    if M is None:
        return False
    if isinstance(M, tuple):  # dense flow, validated where it is built
        return True
    h, w = shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    M33 = to33(M).astype(np.float32)
    try:
        out = cv2.perspectiveTransform(corners, M33).reshape(-1, 2)
    except cv2.error:
        return False
    if not np.isfinite(out).all():
        return False
    disp = np.linalg.norm(out - corners.reshape(-1, 2), axis=1).max()
    return disp < max_disp_frac * np.hypot(h, w)


def refine_affine(orig_gray, ai_gray, mask, M0, max_dim=700):
    """Extend the similarity M0 to a full affine (anisotropic scale + shear) by
    coordinate descent on the SAME masked gradient-NCC the search uses -- the one
    metric that stays reliable across the AI's texture changes. The result is a
    single global affine, so it corrects 'squish/stretch' smoothly with no
    non-rigid waviness. Returns (M_fullres 2x3, score) or (None, -1)."""
    h, w = orig_gray.shape
    scale = min(1.0, max_dim / max(h, w))
    o = cv2.GaussianBlur(_resize(orig_gray, scale), (0, 0), 1.2).astype(np.float32)
    a = cv2.GaussianBlur(_resize(ai_gray, scale), (0, 0), 1.2).astype(np.float32)
    sh, sw = o.shape
    cx, cy = sw / 2.0, sh / 2.0

    def grad(img):
        return cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0),
                             cv2.Sobel(img, cv2.CV_32F, 0, 1))
    go, ga = grad(o), grad(a)
    idx = (_resize(mask, scale, nearest=True) > 0) if mask is not None \
        else np.ones((sh, sw), bool)
    go_c = go[idx] - go[idx].mean()
    go_norm = np.sqrt((go_c * go_c).sum()) + 1e-9
    M0s = full_to_small(M0, scale)
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])

    def build(p):
        sx, sy, shx, shy, tx, ty = p
        A = np.array([[sx, shx, 0], [shy, sy, 0], [0, 0, 1.0]])
        T2 = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1.0]])
        return ((T2 @ A @ T1) @ M0s)[:2, :].astype(np.float32)

    def score(p):
        gw = cv2.warpAffine(ga, build(p), (sw, sh), flags=cv2.INTER_LINEAR)
        gm = gw[idx]; gm = gm - gm.mean()
        return float((go_c * gm).sum() / (np.sqrt((gm * gm).sum()) + 1e-9) / go_norm)

    p = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    best = score(p)
    # Coordinate descent with shrinking steps. Steps: scale, shear (unitless),
    # translation (small-image px).
    steps = [0.03, 0.03, 0.02, 0.02, 4.0, 4.0]
    for _ in range(4):
        improved = False
        for k in range(6):
            for d in (steps[k], -steps[k]):
                q = p.copy(); q[k] += d
                # clamp to a sane range so a runaway never squishes wildly.
                if not (0.85 < q[0] < 1.18 and 0.85 < q[1] < 1.18
                        and abs(q[2]) < 0.12 and abs(q[3]) < 0.12):
                    continue
                v = score(q)
                if v > best + 1e-5:
                    best, p, improved = v, q, True
        if not improved:
            steps = [s / 2 for s in steps]
    M_full = small_to_full(build(p), scale)
    return M_full[:2, :].astype(np.float32), best


def ecc_model(orig_gray, ai_gray, mask, M_init, motion, max_dim=1000,
              blur=2.0, iters=200):
    """ECC alignment (AI->ORIG) with a chosen motion model, masked to the truck.
    motion is cv2.MOTION_AFFINE or cv2.MOTION_HOMOGRAPHY. Returns full-res 2x3/3x3
    or None."""
    h, w = orig_gray.shape
    scale = min(1.0, max_dim / max(h, w))
    osmall = cv2.GaussianBlur(_resize(orig_gray, scale), (0, 0), blur)
    asmall = cv2.GaussianBlur(_resize(ai_gray, scale), (0, 0), blur)
    msmall = _resize(mask, scale, nearest=True) if mask is not None else None

    M_small = full_to_small(M_init, scale)
    if motion == cv2.MOTION_HOMOGRAPHY:
        warp = M_small.astype(np.float32)
    else:
        warp = M_small[:2, :].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6)
    try:
        _, warp = cv2.findTransformECC(osmall, asmall, warp, motion, criteria, msmall, 5)
    except cv2.error:
        return None
    M_full = small_to_full(warp, scale)
    if motion == cv2.MOTION_HOMOGRAPHY:
        return M_full.astype(np.float32)
    return M_full[:2, :].astype(np.float32)


# ---------------------------------------------------------------------------
# Segmentation-silhouette score: how well the AI's edges land on the truck
# outline described by the seg mask. This is the metric we care about most,
# because it decides whether the seg mask actually fits the fixed AI image.
# ---------------------------------------------------------------------------

def _outline(mask_small):
    grad = cv2.morphologyEx(mask_small, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return grad > 0


def seg_score(orig_gray, ai_gray, tight_mask, M, max_dim=1000):
    """Negative mean chamfer distance (px, in working scale) from the truck
    outline to the nearest AI edge after warping AI by M. Higher = better."""
    h, w = orig_gray.shape
    scale = min(1.0, max_dim / max(h, w))
    warped = warp_gray(ai_gray, M, (w, h))
    a = cv2.GaussianBlur(_resize(warped, scale), (0, 0), 1.2)
    ms = _resize(tight_mask, scale, nearest=True)
    outline = _outline(ms)
    if outline.sum() < 20:
        return -1e9
    edges = cv2.Canny(a, 50, 150)
    # distanceTransform gives distance to nearest zero pixel -> make edges zero.
    dt = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return -float(dt[outline].mean())


def _fill_nan_grid(c):
    """Fill NaN cells of a small 2D grid by repeated neighbor averaging, so the
    flow field extrapolates smoothly outside the trusted (truck-body) area."""
    for _ in range(c.size):
        nan = np.isnan(c)
        if not nan.any():
            break
        vs = cv2.blur(np.where(nan, 0, c).astype(np.float32), (3, 3))
        ws = cv2.blur((~nan).astype(np.float32), (3, 3))
        fill = nan & (ws > 1e-6)
        c[fill] = vs[fill] / ws[fill]
    return c


def _grad_image(gray):
    """Contrast-normalized gradient magnitude as uint8. Optical flow between
    the render and the AI repaint is only reliable on edge structure, not on
    raw intensity (the AI changes colors/lighting freely)."""
    g = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0),
                        cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return np.clip(mag / (np.percentile(mag, 95) + 1e-6) * 255, 0, 255).astype(np.uint8)


def flow_refine(orig_gray, aligned_gray, tight_mask, max_dim=1100,
                damp=0.85, max_disp_px=15.0, cell_px=40, fb_thresh=1.2):
    """Residual non-rigid correction after the global warp.

    Dense optical flow orig->aligned (Farneback on gradient images) is
    measured, but kept only where it can be trusted: inside the (eroded)
    truck-body mask, forward/backward consistent, and on textured pixels.
    The trusted vectors are averaged onto a coarse grid (~cell_px cells),
    unknown cells are filled by smooth extrapolation, the grid is blurred,
    damped and hard-clamped to max_disp_px. The result is a very smooth,
    small-amplitude field.

    Returns ('flow', mapx, mapy) absolute full-res sample maps into the
    *aligned* image, or None if there is too little trusted signal.
    """
    h, w = orig_gray.shape
    scale = min(1.0, max_dim / max(h, w))
    o = _grad_image(_resize(orig_gray, scale))
    a = _grad_image(_resize(aligned_gray, scale))
    sh, sw = o.shape

    fb_args = (0.5, 5, 31, 5, 7, 1.5, 0)   # pyr_scale, levels, win, iters, poly
    f_fw = cv2.calcOpticalFlowFarneback(o, a, None, *fb_args)
    f_bw = cv2.calcOpticalFlowFarneback(a, o, None, *fb_args)

    xs, ys = np.meshgrid(np.arange(sw, dtype=np.float32),
                         np.arange(sh, dtype=np.float32))
    bx = cv2.remap(f_bw[..., 0], xs + f_fw[..., 0], ys + f_fw[..., 1],
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    by = cv2.remap(f_bw[..., 1], xs + f_fw[..., 0], ys + f_fw[..., 1],
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fb_err = np.hypot(f_fw[..., 0] + bx, f_fw[..., 1] + by)

    if tight_mask is None:
        return None
    m = _resize(tight_mask, scale, nearest=True) > 0
    m = cv2.erode(m.astype(np.uint8) * 255,
                  np.ones((5, 5), np.uint8)) > 0   # stay off the repainted rim
    if m.sum() < 500:
        return None
    gm = o.astype(np.float32)   # o is already a gradient-magnitude image
    fmag = np.hypot(f_fw[..., 0], f_fw[..., 1])
    max_disp_s = max_disp_px * scale
    conf = (m & (fb_err < fb_thresh) & (gm > np.percentile(gm[m], 50))
            & (fmag < 1.5 * max_disp_s)).astype(np.float32)
    if conf.sum() < 500:
        return None

    gw = max(4, int(round(sw / cell_px)))
    gh = max(3, int(round(sh / cell_px)))
    den = cv2.resize(conf, (gw, gh), interpolation=cv2.INTER_AREA)
    cells = []
    for ch in range(2):
        num = cv2.resize(f_fw[..., ch] * conf, (gw, gh), interpolation=cv2.INTER_AREA)
        c = np.where(den > 0.05, num / np.maximum(den, 1e-6), np.nan)
        cells.append(c)
    if (~np.isnan(cells[0])).sum() < 6:
        return None
    cx = cv2.GaussianBlur(_fill_nan_grid(cells[0]), (0, 0), 1.0)
    cy = cv2.GaussianBlur(_fill_nan_grid(cells[1]), (0, 0), 1.0)

    cx *= damp
    cy *= damp
    mag = np.hypot(cx, cy)
    over = mag > max_disp_s
    if over.any():
        f = np.where(over, max_disp_s / (mag + 1e-9), 1.0)
        cx *= f
        cy *= f

    fx = cv2.resize(cx.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) / scale
    fy = cv2.resize(cy.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) / scale
    if not (np.isfinite(fx).all() and np.isfinite(fy).all()):
        return None
    xs_f, ys_f = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
    return ("flow", xs_f + fx, ys_f + fy)


def compose_map(M, flow, w, h):
    """Single remap combining the global warp M (AI->ORIG) and an optional
    residual flow measured on the M-aligned image:
        out(x) = ai( M^-1 (flow(x)) )
    so the AI image is resampled exactly once."""
    if flow is not None:
        px, py = flow[1], flow[2]
    else:
        px, py = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
    Minv = np.linalg.inv(to33(M).astype(np.float64))
    den = Minv[2, 0] * px + Minv[2, 1] * py + Minv[2, 2]
    mapx = (Minv[0, 0] * px + Minv[0, 1] * py + Minv[0, 2]) / den
    mapy = (Minv[1, 0] * px + Minv[1, 1] * py + Minv[1, 2]) / den
    return mapx.astype(np.float32), mapy.astype(np.float32)


def make_overlay(orig, aligned, mask=None, tight=None):
    """Edge overlay: original edges red, aligned-AI edges green (overlap=yellow).
    If a mask is given, the truck-body region is tinted so it is easy to inspect."""
    go = cv2.Canny(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), 60, 160)
    ga = cv2.Canny(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), 60, 160)
    out = np.zeros_like(orig)
    out[..., 2] = go
    out[..., 1] = ga
    if mask is not None:
        out[..., 0] = (mask > 0).astype(np.uint8) * 40
    if tight is not None:
        out[_outline(tight)] = (255, 255, 255)  # seg truck outline in white
    return out


def _global_similarity(orig_gray, ai_gray, mask):
    """Best similarity (translation+zoom+rotation), AI->ORIG, by grad-NCC."""
    candidates = [("identity", IDENTITY)]
    M_search, _ = search_transform(orig_gray, ai_gray, mask)
    if valid(M_search):
        candidates.append(("search", M_search))
        M_se, _ = ecc_refine(orig_gray, ai_gray, mask, M_search)
        if valid(M_se):
            candidates.append(("search+ecc", M_se))
    M_sift, _, _ = estimate_similarity(orig_gray, ai_gray, mask)
    if valid(M_sift):
        candidates.append(("sift", M_sift))
    scored = sorted(((align_score(orig_gray, ai_gray, mask, M), tag, M)
                     for tag, M in candidates), key=lambda t: t[0], reverse=True)
    return scored[0][2], scored[0][1]


def process(ai_path, save=True, verbose=True, diag=False, nonrigid=True):
    base = ai_path[:-len("_ai.png")]
    orig_path = base + ".png"
    seg_path = base + "_seg.png"

    # Date gate: only correct shots created strictly after the cutoff reference.
    if not past_cutoff(ai_path):
        if verbose:
            print(f"  SKIP (at/before cutoff): {os.path.basename(ai_path)}")
        return None

    if not os.path.exists(orig_path):
        if verbose:
            print(f"  SKIP (no original): {os.path.basename(ai_path)}")
        return None

    orig = cv2.imread(orig_path, cv2.IMREAD_COLOR)
    ai = cv2.imread(ai_path, cv2.IMREAD_COLOR)
    if orig is None or ai is None:
        print(f"  ERROR reading {os.path.basename(ai_path)}")
        return None

    oh, ow = orig.shape[:2]
    ai_r = ai if (ai.shape[0], ai.shape[1]) == (oh, ow) else \
        cv2.resize(ai, (ow, oh), interpolation=cv2.INTER_AREA)

    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    ai_gray = cv2.cvtColor(ai_r, cv2.COLOR_BGR2GRAY)
    mask = load_truck_mask(seg_path, (oh, ow))
    tight = truck_from_seg(seg_path, (oh, ow))

    # --- Stage 0: best global similarity. -------------------------------------
    M0, tag0 = _global_similarity(orig_gray, ai_gray, mask)

    # Scoring: we optimize the seg-silhouette chamfer when a mask is available
    # (what makes the seg mask actually fit), else fall back to grad-NCC.
    if tight is not None:
        score = lambda M: seg_score(orig_gray, ai_gray, tight, M)
        metric = "seg"
    else:
        score = lambda M: align_score(orig_gray, ai_gray, mask, M)
        metric = "ncc"

    id_score = score(IDENTITY)
    best_M, best_tag, best_score = IDENTITY, "identity", id_score
    if score(M0) > best_score:
        best_M, best_tag, best_score = M0, tag0, score(M0)

    # --- Stage 1: escalate to a global affine (anisotropic scale + shear) to
    # correct squish/stretch. Robust grad-NCC coordinate descent, then accept
    # only if it also improves the chosen (seg) metric. Single smooth transform.
    if best_tag != "identity" and mask is not None:
        Ma, _ = refine_affine(orig_gray, ai_gray, mask, best_M)
        if valid_generic(Ma, (oh, ow)) and score(Ma) > best_score + 1e-6:
            best_M, best_tag, best_score = Ma, best_tag + "+affine", score(Ma)

    apply_global = best_tag != "identity" and best_score > id_score + 1e-6
    if not apply_global:
        best_M, best_tag, best_score = IDENTITY, "identity", id_score

    # --- Stage 2: residual smooth-flow correction (spatially varying warp). ---
    # Gated on the whole-truck grad-NCC, a metric the flow cannot trivially
    # game, and on the seg chamfer not getting worse.
    flow = None
    if nonrigid and tight is not None:
        aligned = warp_gray(ai_gray, best_M, (ow, oh))
        f = flow_refine(orig_gray, aligned, tight)
        if f is not None:
            base_ncc = align_score(orig_gray, aligned, mask, IDENTITY)
            flowed = warp_gray(aligned, f, (ow, oh))
            new_ncc = align_score(orig_gray, flowed, mask, IDENTITY)
            base_ch = seg_score(orig_gray, aligned, tight, IDENTITY)
            new_ch = seg_score(orig_gray, flowed, tight, IDENTITY)
            # Combined-evidence gate: the two metrics vote (one clear win, or
            # two moderate ones), and neither may degrade beyond a trivial
            # amount. The flow never optimizes the chamfer directly, so a
            # chamfer improvement is honest evidence of better alignment.
            d_ncc = new_ncc - base_ncc
            d_ch = new_ch - base_ch          # chamfer px, higher = better
            if d_ncc > -0.012 and d_ch > -0.15 and d_ncc / 0.02 + d_ch / 0.5 > 1.0:
                flow, best_tag = f, best_tag + "+flow"
                best_score = new_ch if metric == "seg" else new_ncc

    # Magnitude cap: if the correction would move/zoom/rotate the frame by more
    # than MAX_CORRECTION_FRAC, don't correct this shot at all.
    if apply_global or flow is not None:
        if correction_too_large(best_M, flow, (oh, ow)):
            if verbose:
                print(f"  SKIP (>{int(MAX_CORRECTION_FRAC * 100)}% correction): "
                      f"{os.path.basename(ai_path)}")
            best_M, best_tag, best_score, flow = IDENTITY, "identity", id_score, None
            apply_global = False

    # Compose global warp + flow into ONE remap so the AI image is resampled
    # exactly once.
    if best_tag == "identity" and flow is None:
        fixed = ai_r.copy()
    else:
        mapx, mapy = compose_map(best_M, flow, ow, oh)
        fixed = cv2.remap(ai_r, mapx, mapy, cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
    apply_fix = best_tag != "identity"

    out_path = base + "_ai_fix.png"
    if save:
        cv2.imwrite(out_path, fixed)
    if diag:
        cv2.imwrite(base + "_diag_before.png", make_overlay(orig, ai_r, mask, tight))
        cv2.imwrite(base + "_diag_after.png", make_overlay(orig, fixed, mask, tight))
    if verbose:
        label = f"OK [{best_tag}]" if apply_fix else "COPY [aligned]"
        print(f"  {label} {metric}={best_score:.2f}(id {id_score:.2f}) "
              f"mask={'y' if mask is not None else 'n'}: {os.path.basename(out_path)}")
    return out_path


def main():
    args = sys.argv[1:]
    diag = "--diag" in args
    nonrigid = "--no-flow" not in args   # smooth-flow stage is on by default
    jobs = 1
    rest = []
    it = iter([a for a in args if a not in ("--diag", "--no-flow")])
    for a in it:
        if a == "--jobs":
            jobs = int(next(it))
        elif a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        else:
            rest.append(a)
    if not rest:
        print("usage: fix_ai_offset.py [--diag] [--no-flow] [--jobs N] "
              "<folder | ai_image.png> [...]")
        return
    targets = []
    for a in rest:
        if os.path.isdir(a):
            targets += sorted(glob.glob(os.path.join(a, "*_ai.png")))
        else:
            targets.append(a)
    print(f"Processing {len(targets)} AI images (jobs={jobs})...")

    ok = 0
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial
        fn = partial(process, save=True, verbose=False, diag=diag, nonrigid=nonrigid)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(fn, p): p for p in targets}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    if fut.result() is not None:
                        ok += 1
                except Exception as e:
                    print(f"  ERROR {os.path.basename(futs[fut])}: {e}")
                if i % 25 == 0 or i == len(targets):
                    print(f"  {i}/{len(targets)} done ({ok} fixed)")
    else:
        for p in targets:
            if process(p, diag=diag, nonrigid=nonrigid) is not None:
                ok += 1
    print(f"Done. {ok}/{len(targets)} written.")


if __name__ == "__main__":
    main()
