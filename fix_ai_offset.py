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

Two independent estimators are tried and the best-scoring result is kept:
  * SIFT features + RANSAC similarity (good for clear texture/scale change)
  * ECC intensity correlation on blurred images (good for small pure shifts)
A candidate is only accepted if it beats leaving the image untouched, and the
transform is clamped to a small, sane range so a bad fit can never wreck a frame.
"""

import sys
import glob
import os
import cv2
import numpy as np

MASK_DILATE_FRAC = 0.06   # grow truck mask by ~6% of image to cover the offset
IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], np.float32)


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


def _tps_fit(P, V, reg=1.0):
    """Fit a 2D thin-plate spline mapping control points P (n,2) -> values V (n,2).
    Returns (params (n+3, 2), P). `reg` smooths the fit."""
    n = len(P)
    diff = P[:, None, :] - P[None, :, :]
    d2 = (diff * diff).sum(-1)
    K = 0.5 * d2 * np.log(d2 + 1e-12)
    K[np.arange(n), np.arange(n)] = reg
    Pm = np.hstack([np.ones((n, 1)), P])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K
    L[:n, n:] = Pm
    L[n:, :n] = Pm.T
    rhs = np.zeros((n + 3, 2))
    rhs[:n] = V
    params = np.linalg.solve(L, rhs)
    return params, P


def _tps_apply(params, P, pts):
    """Evaluate the TPS at pts (m,2). Returns (m,2)."""
    n = len(P)
    diff = pts[:, None, :] - P[None, :, :]
    d2 = (diff * diff).sum(-1)
    U = 0.5 * d2 * np.log(d2 + 1e-12)
    Pm = np.hstack([np.ones((len(pts), 1)), pts])
    return U @ params[:n] + Pm @ params[n:]


def tps_refine(orig_gray, ai_gray, tight_mask, M_global, max_dim=1200,
               search_frac=0.03, step=22):
    """Boundary-guided thin-plate-spline. After the global warp M_global, the AI
    truck edge still deviates from the seg outline by small, spatially-varying
    amounts (lens/perspective 'curving' the AI reintroduced). We sample points
    along the seg truck outline, find the nearest strong AI edge along the local
    normal, and fit a TPS that snaps the AI edge onto the outline -- pinned by
    identity anchors on a border grid so the interior stays stable.

    Returns a ('flow', mapx, mapy) full-res warp to apply *after* M_global, or None.
    """
    h, w = orig_gray.shape
    scale = min(1.0, max_dim / max(h, w))
    sh, sw = int(round(h * scale)), int(round(w * scale))
    aligned = warp_gray(ai_gray, M_global, (w, h))
    a = cv2.GaussianBlur(cv2.resize(aligned, (sw, sh), interpolation=cv2.INTER_AREA),
                         (0, 0), 1.2).astype(np.float32)
    ms = cv2.resize(tight_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)

    gm = cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
    gthr = np.percentile(gm, 60)

    # Signed distance -> smooth inward normals along the outline.
    din = cv2.distanceTransform(ms, cv2.DIST_L2, 3)
    dout = cv2.distanceTransform(255 - ms, cv2.DIST_L2, 3)
    sd = din - dout
    nx = cv2.Sobel(sd, cv2.CV_32F, 1, 0, ksize=5)
    ny = cv2.Sobel(sd, cv2.CV_32F, 0, 1, ksize=5)

    cnts, _ = cv2.findContours(ms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    R = max(4, int(round(search_frac * sw)))

    src, dst = [], []            # aligned-AI point, outline point
    for p in cnt[::step]:
        px, py = int(p[0]), int(p[1])
        if not (0 <= px < sw and 0 <= py < sh):
            continue
        n = np.array([nx[py, px], ny[py, px]], np.float32)
        ln = np.hypot(*n)
        if ln < 1e-3:
            continue
        n /= ln
        best_t, best_g = 0.0, -1.0
        for t in range(-R, R + 1):
            qx, qy = px + n[0] * t, py + n[1] * t
            ix, iy = int(round(qx)), int(round(qy))
            if 0 <= ix < sw and 0 <= iy < sh and gm[iy, ix] > best_g:
                best_g, best_t = gm[iy, ix], t
        if best_g < gthr or abs(best_t) >= R:
            continue
        src.append([px + n[0] * best_t, py + n[1] * best_t])  # AI edge location
        dst.append([px, py])                                  # where it should be

    if len(src) < 8:
        return None

    # Identity anchors on a border grid to regularize the interior.
    gx = np.linspace(0, sw - 1, 7)
    gy = np.linspace(0, sh - 1, 7)
    for x in gx:
        for y in gy:
            if x in (gx[0], gx[-1]) or y in (gy[0], gy[-1]):
                src.append([x, y]); dst.append([x, y])

    P = np.array(dst, np.float64)   # outline-frame control points
    V = np.array(src, np.float64)   # where to sample in the aligned AI
    try:
        params, P = _tps_fit(P, V, reg=1.0)
    except np.linalg.LinAlgError:
        return None

    # Evaluate the (smooth) TPS on a coarse grid, then upsample the map.
    ecol = min(sw, 160)
    erow = max(2, int(round(ecol * sh / sw)))
    gx = np.linspace(0, sw - 1, ecol)
    gy = np.linspace(0, sh - 1, erow)
    mx, my = np.meshgrid(gx, gy)
    grid = np.stack([mx.ravel(), my.ravel()], 1)
    mapped = _tps_apply(params, P, grid).reshape(erow, ecol, 2)
    mapx = cv2.resize(mapped[..., 0].astype(np.float32), (w, h)) / scale
    mapy = cv2.resize(mapped[..., 1].astype(np.float32), (w, h)) / scale
    if not (np.isfinite(mapx).all() and np.isfinite(mapy).all()):
        return None
    # Reject if the field moved anything by an implausible amount.
    if max(np.abs(mapx - np.arange(w)[None, :]).max(),
           np.abs(mapy - np.arange(h)[:, None]).max()) > 0.10 * np.hypot(h, w):
        return None
    return ("flow", mapx.astype(np.float32), mapy.astype(np.float32))


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


def process(ai_path, save=True, verbose=True, diag=False, nonrigid=False):
    base = ai_path[:-len("_ai.png")]
    orig_path = base + ".png"
    seg_path = base + "_seg.png"
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

    # --- Stage 3: boundary-guided TPS (non-rigid curvature). ------------------
    flow = None
    if nonrigid and tight is not None and best_tag != "identity":
        f = tps_refine(orig_gray, ai_gray, tight, best_M)
        if f is not None:
            # Judge TPS by a metric it can't trivially game: whole-truck grad-NCC
            # (NOT the outline chamfer it directly optimizes, which would reward
            # interior 'water' distortion). Require a clear improvement.
            aligned = warp_gray(ai_gray, best_M, (ow, oh))
            base_ncc = align_score(orig_gray, aligned, mask, IDENTITY)
            warped = warp_gray(aligned, f, (ow, oh))
            new_ncc = align_score(orig_gray, warped, mask, IDENTITY)
            if new_ncc > base_ncc + 0.02:
                flow, best_tag = f, best_tag + "+tps"

    apply_fix = best_tag != "identity" and best_score > id_score + 1e-6
    if not apply_fix:
        best_M, flow = IDENTITY, None

    # Compose the final warp: global similarity/affine/homography, then flow.
    fixed = warp_gray(ai_r, best_M, (ow, oh), flags=cv2.INTER_CUBIC)
    if flow is not None:
        fixed = warp_gray(fixed, flow, (ow, oh), flags=cv2.INTER_CUBIC)

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
    nonrigid = "--tps" in args   # non-rigid TPS is opt-in (can distort; off by default)
    jobs = 1
    rest = []
    it = iter([a for a in args if a not in ("--diag", "--tps")])
    for a in it:
        if a == "--jobs":
            jobs = int(next(it))
        elif a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        else:
            rest.append(a)
    if not rest:
        print("usage: fix_ai_offset.py [--diag] [--tps] [--jobs N] "
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
