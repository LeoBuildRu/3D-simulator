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


def load_truck_mask(seg_path, shape):
    """Binary mask (uint8 0/255) of the truck body (blue in the seg image),
    dilated so it still covers the truck after a few-percent shift. Returns None
    if no seg file or no blue found."""
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
    k = int(round(MASK_DILATE_FRAC * max(shape)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(blue, ker)


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


def make_overlay(orig, aligned, mask=None):
    """Edge overlay: original edges red, aligned-AI edges green (overlap=yellow).
    If a mask is given, the truck-body region is tinted so it is easy to inspect."""
    go = cv2.Canny(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), 60, 160)
    ga = cv2.Canny(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), 60, 160)
    out = np.zeros_like(orig)
    out[..., 2] = go
    out[..., 1] = ga
    if mask is not None:
        out[..., 0] = (mask > 0).astype(np.uint8) * 40
    return out


def process(ai_path, save=True, verbose=True, diag=False):
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
    if (ai.shape[0], ai.shape[1]) != (oh, ow):
        ai_r = cv2.resize(ai, (ow, oh), interpolation=cv2.INTER_AREA)
    else:
        ai_r = ai

    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    ai_gray = cv2.cvtColor(ai_r, cv2.COLOR_BGR2GRAY)
    mask = load_truck_mask(seg_path, (oh, ow))

    # Primary estimator: global coarse-to-fine masked search (most robust).
    n_in = n_match = 0
    candidates = []
    M_search, _ = search_transform(orig_gray, ai_gray, mask)
    if valid(M_search):
        candidates.append(("search", M_search))
        # Polish the search result with masked ECC for sub-pixel accuracy.
        M_se, _ = ecc_refine(orig_gray, ai_gray, mask, M_search)
        if valid(M_se):
            candidates.append(("search+ecc", M_se))
    # SIFT as an independent cross-check (helps when scale change is larger).
    M_sift, n_in, n_match = estimate_similarity(orig_gray, ai_gray, mask)
    if valid(M_sift):
        candidates.append(("sift", M_sift))
    candidates.append(("identity", IDENTITY))

    scored = sorted(
        ((align_score(orig_gray, ai_gray, mask, M), tag, M) for tag, M in candidates),
        key=lambda t: t[0], reverse=True)
    best_score, best_tag, M = scored[0]
    id_score = align_score(orig_gray, ai_gray, mask, IDENTITY)

    # Only apply a correction if it beats leaving the image untouched by a margin.
    # Otherwise the truck body is already aligned and we copy the AI image as-is,
    # so every *_ai.png always has a matching *_ai_fix.png.
    apply_fix = not (best_tag == "identity" or best_score < id_score + 0.01)
    if not apply_fix:
        M = IDENTITY

    s, rot, tx, ty = decompose(M)
    fixed = cv2.warpAffine(ai_r, M, (ow, oh),
                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    out_path = base + "_ai_fix.png"
    if save:
        cv2.imwrite(out_path, fixed)
    if diag:
        cv2.imwrite(base + "_diag_before.png", make_overlay(orig, ai_r, mask))
        cv2.imwrite(base + "_diag_after.png", make_overlay(orig, fixed, mask))
    if verbose:
        tag = f"OK [{best_tag}]" if apply_fix else "COPY [aligned]"
        print(f"  {tag} s={s:.4f} rot={rot:+.2f} t=({tx:+.1f},{ty:+.1f}) "
              f"score={best_score:.3f}(id {id_score:.3f}) mask="
              f"{'y' if mask is not None else 'n'}: {os.path.basename(out_path)}")
    return out_path


def main():
    args = sys.argv[1:]
    diag = "--diag" in args
    args = [a for a in args if a != "--diag"]
    if not args:
        print("usage: fix_ai_offset.py [--diag] <folder | ai_image.png> [...]")
        return
    targets = []
    for a in args:
        if os.path.isdir(a):
            targets += sorted(glob.glob(os.path.join(a, "*_ai.png")))
        else:
            targets.append(a)
    print(f"Processing {len(targets)} AI images...")
    ok = 0
    for p in targets:
        if process(p, diag=diag) is not None:
            ok += 1
    print(f"Done. {ok}/{len(targets)} fixed.")


if __name__ == "__main__":
    main()
