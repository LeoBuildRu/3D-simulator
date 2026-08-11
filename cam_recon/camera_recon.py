"""
camera_recon — reconstruct a pinhole camera's 3D pose (and optionally its FOV)
from three drawn edges of a rectangle whose real width/length are known.

Intended use case
-----------------
A photo/render shows the open top rim of a truck body (a rectangle of known
W x L).  The operator traces THREE of its four top edges:

      far edge  (length W)          <-- fully visible, drawn
    +-------------------+
    |                   |
 side A (L)          side B (L)     <-- may be traced only partially
    |                   |
    +- - - - - - - - - -+
      near edge (length W)          <-- NOT drawn, camera sits closest to it

Only the *support lines* of the three strokes are used, so partial strokes are
fine — endpoints never need to land on real corners.

What is recoverable
-------------------
* side A x side B  ->  vanishing point V1 of the long axis.
* V1 + the image of the absolute conic (i.e. a known focal length) pins the
  line on which the second vanishing point V2 must lie; intersecting it with
  the far-edge line gives V2.  Two orthogonal vanishing directions => full
  camera ROTATION.
* Back-projecting the two side lines onto the (now oriented) rectangle plane
  and forcing their separation to equal W gives the plane distance => full
  camera TRANSLATION.
* L is *not* needed to solve; it only places the undrawn near edge, which is
  what makes the overlay a genuine, independent accuracy check.

FOV
---
With three lines the focal length is NOT recoverable: for every f there is a
pose that reproduces all three strokes exactly (6 line DOF vs. 7 unknowns).
`fov_determined` reports this honestly.  Supply either

  * a known FOV / focal length, or
  * a fourth stroke on any visible fragment of the near edge,

and the solution becomes unique — V2 = far x near, and
f^2 = -(V1-c)·(V2-c) closes the system.

Conventions
-----------
Image      : pixels, origin top-left, +x right, +y down.
Camera (cv): +x right, +y down, +z forward (looking direction).
Rect/world : +X = width axis, +Y = length axis pointing AWAY from the camera
             (near edge -> far edge), +Z = up (the camera's side).  Origin at
             the centre of the rectangle.  This is a Z-up right-handed frame,
             matching Panda3D, so the reported (h, p, r) are directly the
             sim's YAW / PITCH / ROLL — expressed relative to the truck body.
             `Solution.to_world()` re-expresses them in sim world coordinates
             once you say where the body sits.

Dependencies: numpy (required), scipy (optional, only for the refinement pass).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:  # refinement is optional — the closed form works without scipy
    from scipy.optimize import least_squares as _least_squares
except Exception:  # pragma: no cover
    _least_squares = None


__all__ = [
    "Segment",
    "Solution",
    "reconstruct",
    "focal_from_fov",
    "fov_from_focal",
    "hpr_from_matrix",
    "matrix_from_hpr",
    "project_points",
    "clip_segment_near",
    "rect_corners",
]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

_EPS = 1e-12


def _h(p) -> np.ndarray:
    """Pixel -> homogeneous image point."""
    p = np.asarray(p, dtype=float)
    return np.array([p[0], p[1], 1.0])


def _norm_line(l: np.ndarray) -> np.ndarray:
    """Scale a homogeneous line so (a, b) is a unit vector (=> l·p is a distance)."""
    n = math.hypot(float(l[0]), float(l[1]))
    return l / n if n > _EPS else l.copy()


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else v.copy()


def focal_from_fov(fov_deg: float, extent_px: float) -> float:
    """Focal length in pixels from a field of view spanning `extent_px` pixels."""
    return (extent_px * 0.5) / math.tan(math.radians(fov_deg) * 0.5)


def fov_from_focal(focal_px: float, extent_px: float) -> float:
    """Inverse of `focal_from_fov`, in degrees."""
    return math.degrees(2.0 * math.atan((extent_px * 0.5) / focal_px))


def _sigma_text(sx: Optional[float], sy: Optional[float]) -> str:
    """Render the FOV error bars for humans."""
    def one(s):
        if s is None:
            return "n/a"
        return "inf" if not math.isfinite(s) else "%.2f" % s
    if sx is None and sy is None:
        return "n/a"
    return "+/- %s deg h / %s deg v per 1 px of stroke error" % (one(sx), one(sy))


@dataclass(eq=False)          # numpy fields make a generated __eq__ a trap
class Segment:
    """A stroke the operator drew: two pixel endpoints of a (possibly partial) edge."""

    p0: np.ndarray
    p1: np.ndarray

    def __init__(self, p0, p1):
        self.p0 = np.asarray(p0, dtype=float)[:2]
        self.p1 = np.asarray(p1, dtype=float)[:2]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    def line(self) -> np.ndarray:
        """Homogeneous support line, normalised so l·(x, y, 1) is a pixel distance."""
        return _norm_line(np.cross(_h(self.p0), _h(self.p1)))

    def as_list(self) -> list:
        return [self.p0.tolist(), self.p1.tolist()]

    @staticmethod
    def from_list(data) -> "Segment":
        return Segment(data[0], data[1])


# --------------------------------------------------------------------------
# rotation conventions
# --------------------------------------------------------------------------

# Maps a vector's CV-camera components (x right, y down, z forward) to the same
# vector's Panda3D camera-local components (x right, y forward, z up).
_CV_TO_PANDA = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])


def matrix_from_hpr(h_deg: float, p_deg: float, r_deg: float) -> np.ndarray:
    """Panda3D HPR -> rotation matrix (node-local axes as columns, Z-up world)."""
    h, p, r = (math.radians(a) for a in (h_deg, p_deg, r_deg))
    ch, sh = math.cos(h), math.sin(h)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    rz = np.array([[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
    return rz @ rx @ ry


def hpr_from_matrix(m: np.ndarray) -> tuple:
    """Rotation matrix -> Panda3D (heading, pitch, roll) in degrees."""
    sp = float(np.clip(m[2, 1], -1.0, 1.0))
    p = math.asin(sp)
    if abs(sp) > 1.0 - 1e-9:                      # gimbal lock: fold roll into heading
        h = math.atan2(float(m[0, 2]), float(m[0, 0]))
        r = 0.0
    else:
        h = math.atan2(float(-m[0, 1]), float(m[1, 1]))
        r = math.atan2(float(-m[2, 0]), float(m[2, 2]))
    return math.degrees(h), math.degrees(p), math.degrees(r)


def _panda_hpr_from_cv(r_world_from_cv: np.ndarray) -> tuple:
    """CV camera axes expressed in the Z-up world -> Panda3D HPR degrees."""
    return hpr_from_matrix(r_world_from_cv @ _CV_TO_PANDA.T)


# --------------------------------------------------------------------------
# projection utilities
# --------------------------------------------------------------------------

def rect_corners(width: float, length: float) -> np.ndarray:
    """The rectangle's four corners in rect coordinates, ordered as a polygon.

    Order: near-left, near-right, far-right, far-left (+Y is the far side).
    """
    hw, hl = width * 0.5, length * 0.5
    return np.array([[-hw, -hl, 0.0],
                     [+hw, -hl, 0.0],
                     [+hw, +hl, 0.0],
                     [-hw, +hl, 0.0]])


def project_points(k: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """Camera-space points -> pixels.  Points at or behind z=0 come back as NaN."""
    pts_cam = np.atleast_2d(pts_cam)
    out = np.full((pts_cam.shape[0], 2), np.nan)
    z = pts_cam[:, 2]
    ok = z > 1e-9
    if np.any(ok):
        proj = (k @ pts_cam[ok].T).T
        out[ok] = proj[:, :2] / proj[:, 2:3]
    return out


def clip_segment_near(a: np.ndarray, b: np.ndarray, z_min: float = 1e-3):
    """Clip a camera-space segment against the near plane z = z_min.

    Returns the clipped (a, b) pair, or None when the whole segment is behind
    the camera.  Needed because with a wide FOV the reconstructed rectangle
    routinely straddles the image border and the near plane.
    """
    az, bz = float(a[2]), float(b[2])
    if az < z_min and bz < z_min:
        return None
    if az >= z_min and bz >= z_min:
        return a, b
    t = (z_min - az) / (bz - az)
    mid = a + t * (b - a)
    return (mid, b) if az < z_min else (a, mid)


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------

@dataclass
class Solution:
    """Everything the reconstruction produced, plus how much to trust it."""

    # intrinsics.  fx and fy are independent: fy != fx means non-square pixels,
    # i.e. an anamorphic / squeezed frame rather than a plain wide-angle one.
    focal_px: float                       # fx
    focal_y_px: float                     # fy
    principal_point: tuple
    image_size: tuple
    fov_x_deg: float
    fov_y_deg: float
    fov_determined: bool
    fov_uncertainty_deg: Optional[float]     # RMS h-FOV swing per 1 px of stroke error
    fov_y_uncertainty_deg: Optional[float]   # same for the vertical FOV
    anamorphic_solved: bool                  # fy was measured, not assumed equal to fx

    # extrinsics, expressed in the rectangle frame described in the module docstring
    position: np.ndarray                 # camera centre, rect coords
    hpr_deg: tuple                       # heading / pitch / roll, Panda3D convention
    r_cam_from_rect: np.ndarray          # 3x3, rect axes as columns in camera coords
    t_cam_from_rect: np.ndarray          # rect origin expressed in camera coords

    # geometry
    width: float
    length: float
    corners_rect: np.ndarray             # 4x3
    corners_cam: np.ndarray              # 4x3
    corners_img: np.ndarray              # 4x2 (NaN where behind the camera)
    height_above_plane: float
    distance_to_centre: float

    # quality
    residual_px: float
    residuals: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    # -- convenience ------------------------------------------------------
    @property
    def k(self) -> np.ndarray:
        cx, cy = self.principal_point
        return np.array([[self.focal_px, 0.0, cx],
                         [0.0, self.focal_y_px, cy],
                         [0.0, 0.0, 1.0]])

    @property
    def pixel_aspect(self) -> float:
        """fy / fx.  1.0 = square pixels; >1 = the frame is squeezed horizontally."""
        return self.focal_y_px / self.focal_px

    def rect_to_cam(self, pts_rect) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts_rect, dtype=float))
        return (self.r_cam_from_rect @ pts.T).T + self.t_cam_from_rect

    def project_rect(self, pts_rect) -> np.ndarray:
        return project_points(self.k, self.rect_to_cam(pts_rect))

    def _edge_pixels(self, a_cam, b_cam, samples: int = 64) -> np.ndarray:
        """One camera-space segment as a near-plane-clipped pixel polyline."""
        clipped = clip_segment_near(np.asarray(a_cam, dtype=float),
                                    np.asarray(b_cam, dtype=float))
        if clipped is None:
            return np.empty((0, 2))
        ca, cb = clipped
        ts = np.linspace(0.0, 1.0, max(2, samples))[:, None]
        return project_points(self.k, ca[None, :] * (1.0 - ts) + cb[None, :] * ts)

    def edge_polylines(self, samples: int = 64) -> list:
        """The four rectangle edges as pixel polylines, near-plane clipped.

        Returns a list of (name, Nx2 array) with names
        'near', 'right', 'far', 'left'.  Edges are sampled rather than drawn as
        straight 2-point lines so the result stays correct if a lens model is
        bolted on later; with a plain pinhole the samples are collinear anyway.
        """
        names = ["near", "right", "far", "left"]
        return [(name, self._edge_pixels(self.corners_cam[i],
                                         self.corners_cam[(i + 1) % 4], samples))
                for i, name in enumerate(names)]

    # -- the body hanging below the rim -----------------------------------
    def body_corners_cam(self, depth: float) -> np.ndarray:
        """The four bottom corners of a box `depth` deep, in camera coords."""
        bottom_rect = self.corners_rect - np.array([0.0, 0.0, float(depth)])
        return self.rect_to_cam(bottom_rect)

    def body_polylines(self, depth: float, samples: int = 64,
                       rings=()) -> list:
        """Wireframe of the body box hanging `depth` below the traced rim.

        Returns (kind, Nx2) pairs with kind in {'pillar', 'bottom', 'ring'}.
        The pillars are the four vertical corner drops: they are pure
        prediction from the volume, and in a correct solve they must run
        parallel to the real body's vertical edges in the photo, which makes
        them a much sharper check on pitch/roll/FOV than the flat rim alone.
        """
        bottom = self.body_corners_cam(depth)
        out = []
        for i in range(4):
            out.append(("pillar", self._edge_pixels(self.corners_cam[i],
                                                    bottom[i], samples)))
        for i in range(4):
            out.append(("bottom", self._edge_pixels(bottom[i],
                                                    bottom[(i + 1) % 4], samples)))
        for frac in rings:
            ring = self.rect_to_cam(
                self.corners_rect - np.array([0.0, 0.0, float(depth) * float(frac)]))
            for i in range(4):
                out.append(("ring", self._edge_pixels(ring[i], ring[(i + 1) % 4],
                                                      samples)))
        return out

    def body_faces(self, depth: float) -> list:
        """The box's four side quads plus its floor, as pixel polygons.

        Faces that touch or cross the near plane are dropped rather than
        clipped — they are only shading to help read the box as a solid, and a
        missing one is better than a wrong one.
        """
        top, bottom = self.corners_cam, self.body_corners_cam(depth)
        quads = [("side%d" % i, np.array([top[i], top[(i + 1) % 4],
                                          bottom[(i + 1) % 4], bottom[i]]))
                 for i in range(4)]
        quads.append(("floor", bottom))
        out = []
        for name, pts_cam in quads:
            if np.min(pts_cam[:, 2]) <= 1e-3:
                continue
            px = project_points(self.k, pts_cam)
            if np.all(np.isfinite(px)):
                out.append((name, px))
        return out

    def to_world(self, rect_centre=(0.0, 0.0, 0.0), rect_heading_deg: float = 0.0):
        """Re-express the camera in sim world coordinates.

        `rect_heading_deg` is the Panda heading of the rectangle's +Y axis
        (the near->far direction) in the world; `rect_centre` is where the
        rectangle's centre sits.  Returns (position_xyz, (h, p, r)).
        """
        yaw = math.radians(rect_heading_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        pos = rz @ self.position + np.asarray(rect_centre, dtype=float)
        r_world_from_cv = rz @ self.r_cam_from_rect.T
        return pos, _panda_hpr_from_cv(r_world_from_cv)

    def summary(self) -> str:
        x, y, z = self.position
        h, p, r = self.hpr_deg
        lines = [
            "camera position (rect frame)  X %+8.4f   Y %+8.4f   Z %+8.4f" % (x, y, z),
            "orientation                   YAW %+7.2f  PITCH %+7.2f  ROLL %+7.2f" % (h, p, r),
            "height above rim plane        %.4f" % self.height_above_plane,
            "distance to rim centre        %.4f" % self.distance_to_centre,
            "focal length                  fx %.2f px   fy %.2f px" % (
                self.focal_px, self.focal_y_px),
            "field of view                 %.2f deg horizontal / %.2f deg vertical%s"
            % (self.fov_x_deg, self.fov_y_deg,
               "" if self.fov_determined else "   (ASSUMED - not recoverable)"),
            "fov sensitivity               %s" % _sigma_text(
                self.fov_uncertainty_deg, self.fov_y_uncertainty_deg),
            "pixel aspect (fy/fx)          %.5f   %s" % (
                self.pixel_aspect,
                "measured - frame is %s"
                % ("squeezed horizontally" if self.pixel_aspect > 1.001 else
                   "stretched horizontally" if self.pixel_aspect < 0.999 else
                   "square within noise")
                if self.anamorphic_solved else "assumed (square pixels)"),
            "reprojection residual         %.3f px RMS" % self.residual_px,
        ]
        if self.residuals:
            lines.append("  per stroke: " + "  ".join(
                "%s %.2f" % (k, v) for k, v in self.residuals.items()))
        for w in self.warnings:
            lines.append("! " + w)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "focal_px": self.focal_px,
            "focal_y_px": self.focal_y_px,
            "pixel_aspect": self.pixel_aspect,
            "anamorphic_solved": self.anamorphic_solved,
            "principal_point": list(self.principal_point),
            "image_size": list(self.image_size),
            "fov_x_deg": self.fov_x_deg,
            "fov_y_deg": self.fov_y_deg,
            "fov_determined": self.fov_determined,
            "fov_uncertainty_deg": self.fov_uncertainty_deg,
            "fov_y_uncertainty_deg": self.fov_y_uncertainty_deg,
            "position": self.position.tolist(),
            "hpr_deg": list(self.hpr_deg),
            "width": self.width,
            "length": self.length,
            "corners_rect": self.corners_rect.tolist(),
            "corners_cam": self.corners_cam.tolist(),
            "corners_img": self.corners_img.tolist(),
            "height_above_plane": self.height_above_plane,
            "distance_to_centre": self.distance_to_centre,
            "residual_px": self.residual_px,
            "residuals": self.residuals,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------
# core solver
# --------------------------------------------------------------------------

def _second_vanishing_point(v1: np.ndarray, l_end: np.ndarray,
                            k_inv: np.ndarray) -> np.ndarray:
    """VP of the short edges: on the far-edge line and orthogonal to V1.

    Orthogonality of the two rectangle directions reads (K^-1 v1)·(K^-1 v2) = 0,
    i.e. v2 lies on the line m = K^-T K^-1 v1.  Intersect that with the drawn
    far-edge line.
    """
    m = k_inv.T @ (k_inv @ v1)
    return np.cross(m, l_end)


def _focal_from_two_vps(v1: np.ndarray, v2: np.ndarray,
                        cx: float, cy: float,
                        img_w: float = 0.0) -> Optional[float]:
    """Classic single-view calibration from two orthogonal vanishing points."""
    if abs(v1[2]) < _EPS or abs(v2[2]) < _EPS:
        return None                                   # a VP at infinity carries no scale
    a = v1[:2] / v1[2] - np.array([cx, cy])
    b = v2[:2] / v2[2] - np.array([cx, cy])
    f_sq = -float(a @ b)
    if not np.isfinite(f_sq) or f_sq <= 1.0:
        return None
    f = math.sqrt(f_sq)
    if img_w > 0:
        # Reject nonsense before it poisons the refinement: keep the horizontal
        # FOV inside 5..175 degrees.
        lo, hi = focal_from_fov(175.0, img_w), focal_from_fov(5.0, img_w)
        if not (lo <= f <= hi):
            return None
    return f


def _homography_dlt(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """Normalised 4-point DLT: H maps `src` (rim-plane metres) to `dst` (pixels)."""

    def normalise(p):
        c = p.mean(axis=0)
        s = np.sqrt(2.0) / max(np.sqrt(((p - c) ** 2).sum(axis=1)).mean(), _EPS)
        t = np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])
        q = (t @ np.c_[p, np.ones(len(p))].T).T
        return q, t

    a, ta = normalise(np.asarray(src, dtype=float))
    b, tb = normalise(np.asarray(dst, dtype=float))

    rows = []
    for (x, y, w), (u, v, z) in zip(a, b):
        rows.append([0, 0, 0, -z * x, -z * y, -z * w, v * x, v * y, v * w])
        rows.append([z * x, z * y, z * w, 0, 0, 0, -u * x, -u * y, -u * w])
    try:
        _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    except np.linalg.LinAlgError:
        return None
    h = vt[-1].reshape(3, 3)
    h = np.linalg.inv(tb) @ h @ ta
    if abs(h[2, 2]) < _EPS or not np.all(np.isfinite(h)):
        return None
    return h / h[2, 2]


def _corners_from_lines(l_side_a, l_side_b, l_far, l_near) -> Optional[np.ndarray]:
    """The four rim corners, as intersections of adjacent traced lines."""
    out = []
    for la, lb in ((l_near, l_side_a), (l_near, l_side_b),
                   (l_far, l_side_b), (l_far, l_side_a)):
        p = np.cross(la, lb)
        if abs(p[2]) < 1e-9:
            return None                       # adjacent edges traced parallel
        out.append(p[:2] / p[2])
    return np.asarray(out)


def _focals_from_homography(h: np.ndarray, cx: float, cy: float,
                            fx_fixed: Optional[float] = None,
                            fy_fixed: Optional[float] = None):
    """Separate fx and fy from one metric-plane homography.

    A homography of a plane whose coordinates are metric gives two constraints
    on the image of the absolute conic w = (K K^T)^-1:

        h1' w h2 = 0            the two rim axes are perpendicular
        h1' w h1 = h2' w h2     and carry the same metric scale

    The second one is what the known width:length ratio buys us, and it is what
    separates fx from fy.  With a = 1/fx^2 and b = 1/fy^2 both are linear, so
    this is a 2x2 solve (or a 1-unknown least squares when one focal is known).
    """
    p = h[:, 0].copy()
    q = h[:, 1].copy()
    p[0] -= cx * p[2]
    p[1] -= cy * p[2]
    q[0] -= cx * q[2]
    q[1] -= cy * q[2]

    m = np.array([[p[0] * q[0], p[1] * q[1]],
                  [p[0] ** 2 - q[0] ** 2, p[1] ** 2 - q[1] ** 2]])
    rhs = np.array([-p[2] * q[2], -(p[2] ** 2 - q[2] ** 2)])

    known = [fx_fixed, fy_fixed]
    free = [i for i in (0, 1) if known[i] is None]
    if not free:
        return fx_fixed, fy_fixed

    for i in (0, 1):
        if known[i] is not None:
            rhs = rhs - m[:, i] * (1.0 / known[i] ** 2)
    try:
        vals, *_ = np.linalg.lstsq(m[:, free], rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    ab = [None if known[i] is None else 1.0 / known[i] ** 2 for i in (0, 1)]
    for slot, val in zip(free, np.atleast_1d(vals)):
        ab[slot] = float(val)

    if any(v is None or not np.isfinite(v) or v <= 0 for v in ab):
        return None                           # no real camera reproduces this
    return 1.0 / math.sqrt(ab[0]), 1.0 / math.sqrt(ab[1])


def _plausible(f: float, extent: float) -> bool:
    """Keep a solved focal inside 5..175 degrees of field of view."""
    return (focal_from_fov(175.0, extent) <= f <= focal_from_fov(5.0, extent)
            if extent > 0 else True)


def _solve_focals(side_a, side_b, far_edge, near_edge, width, length,
                  cx, cy, img_w, img_h, fx_fixed=None, fy_fixed=None,
                  anamorphic=False):
    """Resolve (fx, fy) from whatever the strokes and the operator provide.

    Returns (fx, fy) or None if the strokes do not determine them.
    """
    if fx_fixed is not None and fy_fixed is not None:
        return fx_fixed, fy_fixed

    if anamorphic:
        if near_edge is None:
            return None                       # needs all four corners
        corners = _corners_from_lines(side_a.line(), side_b.line(),
                                      far_edge.line(), near_edge.line())
        if corners is None:
            return None
        # Rect coords: side_a on -X, side_b on +X, far on +Y, near on -Y.
        # (Swapping either pair mirrors the plane, which leaves both IAC
        # constraints unchanged, so the labelling here is free.)
        hw, hl = width * 0.5, length * 0.5
        src = np.array([[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl]])
        h = _homography_dlt(src, corners)
        if h is None:
            return None
        got = _focals_from_homography(h, cx, cy, fx_fixed, fy_fixed)
        if got is None:
            return None
        fx, fy = got
        if not (_plausible(fx, img_w) and _plausible(fy, img_h)):
            return None
        return fx, fy

    # Square pixels: one unknown, so one orthogonal vanishing-point pair is enough.
    if fx_fixed is not None:
        return fx_fixed, fx_fixed
    if fy_fixed is not None:
        return fy_fixed, fy_fixed
    if near_edge is None:
        return None
    f = _focal_from_two_vps(np.cross(side_a.line(), side_b.line()),
                            np.cross(far_edge.line(), near_edge.line()),
                            cx, cy, img_w)
    return None if f is None else (f, f)


def _intrinsics_uncertainty(side_a, side_b, far_edge, near_edge, width, length,
                            cx, cy, img_w, img_h, fx_fixed, fy_fixed,
                            anamorphic, delta: float = 1.0):
    """Error bars on the solved FOVs: RMS swing per `delta` px of stroke error.

    Perturbs every traced endpoint in turn and re-runs the closed form.  This is
    the honest way to expose conditioning — the anamorphic solve in particular
    degrades fast when the rim is close to fronto-parallel, and the operator
    needs to be told that rather than handed a confident wrong number.
    """
    if near_edge is None:
        return None, None
    segs = [side_a, side_b, far_edge, near_edge]
    base = _solve_focals(*segs, width, length, cx, cy, img_w, img_h,
                         fx_fixed, fy_fixed, anamorphic)
    if base is None:
        return None, None
    base_x = fov_from_focal(base[0], img_w)
    base_y = fov_from_focal(base[1], img_h)

    sx, sy = [], []
    for si in range(4):
        for pi in (0, 1):
            for axis in (0, 1):
                bumped = []
                for j, s in enumerate(segs):
                    p0, p1 = s.p0.copy(), s.p1.copy()
                    if j == si:
                        (p0 if pi == 0 else p1)[axis] += delta
                    bumped.append(Segment(p0, p1))
                got = _solve_focals(*bumped, width, length, cx, cy, img_w, img_h,
                                    fx_fixed, fy_fixed, anamorphic)
                if got is None:
                    return float("inf"), float("inf")
                sx.append(fov_from_focal(got[0], img_w) - base_x)
                sy.append(fov_from_focal(got[1], img_h) - base_y)
    return (float(np.sqrt(np.mean(np.square(sx)))),
            float(np.sqrt(np.mean(np.square(sy)))))


def _pose_from_focal(l_a, l_b, l_end, width, length, k, warnings):
    """Closed-form pose for a fixed intrinsic matrix.

    Returns (r_cam_from_rect, t_cam_from_rect, corners_cam) or None.
    """
    k_inv = np.linalg.inv(k)

    v1 = np.cross(l_a, l_b)                       # VP of the long sides
    if np.linalg.norm(v1) < _EPS:
        warnings.append("The two side strokes are the same line — cannot solve.")
        return None
    d1 = k_inv @ v1
    if np.linalg.norm(d1) < _EPS:
        return None

    v2 = _second_vanishing_point(v1, l_end, k_inv)
    d2 = k_inv @ v2
    if np.linalg.norm(d2) < _EPS:
        warnings.append("The far-edge stroke is degenerate w.r.t. the side strokes.")
        return None

    d_len = _unit(d1)                             # long axis (rect +/-Y)
    d_wid = d2 - float(d2 @ d_len) * d_len        # re-orthogonalise against noise
    if np.linalg.norm(d_wid) < 1e-9:
        warnings.append("Side and far strokes give parallel 3D directions — "
                        "the far stroke is probably not perpendicular to the sides.")
        return None
    d_wid = _unit(d_wid)

    n_a = k.T @ l_a                               # interpretation-plane normals
    n_b = k.T @ l_b
    n_e = k.T @ l_end
    corners_rect = rect_corners(width, length)

    # Three binary choices remain, and they are independent:
    #   eps_len  which way along the long axis is "away"  -> where the near edge lands
    #   sign(h)  the mirror image through the camera centre
    #   eps_wid  relabelling of the same geometry, i.e. which face is "up"
    # Enumerate all eight and score, so no single test has to be bullet-proof.
    best = None
    for eps_len in (1.0, -1.0):
        d_y = eps_len * d_len
        den_e = float(n_e @ d_y)
        if abs(den_e) < _EPS:
            continue                              # far stroke runs through V1
        for eps_wid in (1.0, -1.0):
            d_x = eps_wid * d_wid
            d_z = np.cross(d_x, d_y)              # X x Y = Z, right-handed Z-up rect frame
            den_a, den_b = float(n_a @ d_x), float(n_b @ d_x)
            if abs(den_a) < _EPS or abs(den_b) < _EPS:
                continue                          # a side stroke passes through V2
            ka, kb = float(n_a @ d_z) / den_a, float(n_b @ d_z) / den_b
            spread = ka - kb
            if abs(spread) < 1e-12:
                continue                          # the two side strokes coincide
            h_abs = width / abs(spread)           # plane distance is fixed by the known width

            for h in (-h_abs, h_abs):
                s_far = -h * float(n_e @ d_z) / den_e
                t_mid = -h * 0.5 * (ka + kb)
                s_mid = s_far - 0.5 * length

                r_cr = np.column_stack([d_x, d_y, d_z])
                t_cr = r_cr @ np.array([t_mid, s_mid, h])
                corners_cam = (r_cr @ corners_rect.T).T + t_cr

                # The far edge and the side strokes are what we actually saw, so
                # only those corners must be in front; the near edge is a
                # prediction and may legitimately fall behind a wide-FOV camera.
                far_ok = bool(np.all(corners_cam[2:4, 2] > 1e-6))
                near_ok = bool(np.all(corners_cam[0:2, 2] > 1e-6))
                # "Camera sits closest to the undrawn edge" => +Y points away.
                if abs(d_y[2]) > 1e-4:
                    order_ok = d_y[2] > 0
                else:
                    # Long axis square-on to the view: fall back to the fact that
                    # a rim seen from above puts its near edge lower in the image.
                    px = project_points(k, corners_cam)
                    order_ok = (np.all(np.isfinite(px))
                                and px[0:2, 1].mean() > px[2:4, 1].mean())
                above = (-h) > 0.0                # camera height above the plane is -h

                score = (8 if far_ok else 0) + (4 if order_ok else 0) \
                    + (2 if above else 0) + (1 if near_ok else 0)
                if best is None or score > best[0]:
                    best = (score, r_cr, t_cr, corners_cam, far_ok, order_ok, above)

    if best is None:
        warnings.append("No consistent plane distance — check that the two side "
                        "strokes really are the long edges of the same rectangle.")
        return None

    score, r_cr, t_cr, corners_cam, far_ok, order_ok, above = best
    if not far_ok:
        warnings.append("The traced far edge lands behind the camera — the strokes "
                        "are not consistent with a rectangle.")
    if not order_ok:
        warnings.append("Could not orient the body so the undrawn edge is the near "
                        "one; near/far may be swapped.")
    if not above:
        warnings.append("Could not place the camera above the rim plane.")
    return r_cr, t_cr, corners_cam


# --------------------------------------------------------------------------
# residuals / refinement
# --------------------------------------------------------------------------

# Which rectangle corners bound each named edge (indices into rect_corners()).
_EDGE_CORNERS = {"near": (0, 1), "right": (1, 2), "far": (2, 3), "left": (3, 0)}


def _stroke_residuals(k, r_cr, t_cr, corners_rect, strokes) -> tuple:
    """Signed pixel distances from each stroke endpoint to its rectangle edge."""
    res, per_stroke = [], {}
    for name, seg in strokes:
        i, j = _EDGE_CORNERS[name]
        pa = r_cr @ corners_rect[i] + t_cr
        pb = r_cr @ corners_rect[j] + t_cr
        # Image of a 3D line stays valid even when an endpoint is behind the camera.
        line = _norm_line(np.cross(k @ pa, k @ pb))
        d0 = float(line @ _h(seg.p0))
        d1 = float(line @ _h(seg.p1))
        res.extend((d0, d1))
        per_stroke[name] = math.sqrt(0.5 * (d0 * d0 + d1 * d1))
    return np.asarray(res), per_stroke


def _rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    axis = v / theta
    kx = np.array([[0.0, -axis[2], axis[1]],
                   [axis[2], 0.0, -axis[0]],
                   [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx)


def _matrix_to_rotvec(m: np.ndarray) -> np.ndarray:
    cos_t = float(np.clip((np.trace(m) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_t)
    if theta < 1e-9:
        return np.zeros(3)
    if abs(theta - math.pi) < 1e-6:                       # near-180 deg: use the symmetric part
        a = (m + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(a), 0.0, None))
        idx = int(np.argmax(axis))
        axis = axis * np.sign(a[idx] / (axis[idx] if axis[idx] > _EPS else 1.0))
        return _unit(axis) * theta
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    return axis / (2.0 * math.sin(theta)) * theta


def _refine(k, r_cr, t_cr, corners_rect, strokes, cx, cy, focal_mode,
            img_w=0.0, img_h=0.0):
    """Least-squares polish over pose (and focals, where they are observable)."""
    if _least_squares is None:
        return k, r_cr, t_cr

    r0 = _matrix_to_rotvec(r_cr)
    x0 = np.concatenate([r0, t_cr])
    bounds = (-np.inf, np.inf)
    # `focal_mode` says which focals the strokes actually observe:
    #   "none" fixed   "iso" one shared f   "x"/"y" one of them   "xy" both.
    # "iso" must stay a SINGLE parameter driving both entries: letting fx move
    # while fy sits still would smuggle in an anamorphic freedom the square-pixel
    # model does not have.
    mode = focal_mode
    seeds, extents = [], []
    if mode == "iso":
        seeds, extents = [math.log(k[0, 0])], [img_w]
    elif mode == "x":
        seeds, extents = [math.log(k[0, 0])], [img_w]
    elif mode == "y":
        seeds, extents = [math.log(k[1, 1])], [img_h]
    elif mode == "xy":
        seeds = [math.log(k[0, 0]), math.log(k[1, 1])]
        extents = [img_w, img_h]
    if seeds:
        x0 = np.concatenate([x0, seeds])
        if all(e > 0 for e in extents):
            # Keep free focals inside 5..175 deg of view; an ill-conditioned
            # solve would otherwise run off to a degenerate one.
            lo = [math.log(focal_from_fov(175.0, e)) for e in extents]
            hi = [math.log(focal_from_fov(5.0, e)) for e in extents]
            bounds = (np.array([-np.inf] * 6 + lo), np.array([np.inf] * 6 + hi))

    def unpack(x):
        rot = _rotvec_to_matrix(x[:3])
        tr = x[3:6]
        fx, fy = k[0, 0], k[1, 1]
        if mode == "iso":
            fx = fy = math.exp(x[6])
        elif mode == "x":
            fx = math.exp(x[6])
        elif mode == "y":
            fy = math.exp(x[6])
        elif mode == "xy":
            fx, fy = math.exp(x[6]), math.exp(x[7])
        kk = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return kk, rot, tr

    def fun(x):
        kk, rot, tr = unpack(x)
        return _stroke_residuals(kk, rot, tr, corners_rect, strokes)[0]

    kwargs = dict(xtol=1e-12, ftol=1e-12, max_nfev=500)
    if np.isscalar(bounds[0]):                        # scipy's 'lm' rejects bounds
        kwargs["method"] = "lm"
    else:
        kwargs["method"] = "trf"
        kwargs["bounds"] = bounds
    try:
        sol = _least_squares(fun, x0, **kwargs)
    except Exception:
        return k, r_cr, t_cr
    if not np.all(np.isfinite(sol.x)):
        return k, r_cr, t_cr
    return unpack(sol.x)


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def reconstruct(side_a: Segment,
                side_b: Segment,
                far_edge: Segment,
                width: float,
                length: float,
                image_size: Sequence[int],
                fov_x_deg: Optional[float] = None,
                focal_px: Optional[float] = None,
                near_edge: Optional[Segment] = None,
                principal_point: Optional[Sequence[float]] = None,
                refine: bool = True,
                assumed_fov_x_deg: float = 60.0,
                fov_y_deg: Optional[float] = None,
                focal_y_px: Optional[float] = None,
                anamorphic: bool = False) -> Solution:
    """Reconstruct camera pose (and FOV when observable) from traced rim edges.

    Parameters
    ----------
    side_a, side_b : strokes on the two long edges (length `length`).  Partial
        strokes are fine — only their support lines are used.
    far_edge : stroke on the short edge opposite the camera (length `width`).
    width, length : real dimensions of the rim rectangle, any consistent unit.
        `width` sets the reconstruction's scale.  With three strokes `length`
        only positions the undrawn near edge; with four it also carries the
        aspect-ratio information the anamorphic solve needs.
    image_size : (w, h) in pixels — needed for the principal point and FOV.
    fov_x_deg / focal_px : the camera's known horizontal intrinsics, if any.
        `focal_px` wins when both are given.
    fov_y_deg / focal_y_px : known vertical intrinsics.  Only needed for a
        frame with non-square pixels whose squeeze you already know.
    near_edge : optional stroke on any visible fragment of the near edge.  It
        completes the homography, which is what makes the focal lengths
        observable at all.
    anamorphic : solve fx and fy separately instead of assuming square pixels.
        Requires `near_edge`.  Use it when the frame may be squeezed or
        stretched — a render whose lens aspect did not match its buffer, a
        non-square-pixel capture, a still that was resized on one axis only.
        Note this models a linear anisotropic scale ONLY; it does not and
        cannot model radial (barrel / pincushion) lens distortion, which bends
        straight edges and breaks the whole line-based method.  Undistort first.
    principal_point : defaults to the image centre.
    refine : run a least-squares polish (needs scipy).  Never hurts; only
        changes anything when the problem is over-determined.
    assumed_fov_x_deg : used when the focal length is neither given nor
        observable.  The result is then one member of a one-parameter family —
        `fov_determined` is False and a warning says so.
    """
    warnings: list = []
    img_w, img_h = float(image_size[0]), float(image_size[1])
    if principal_point is None:
        cx, cy = img_w * 0.5, img_h * 0.5
    else:
        cx, cy = float(principal_point[0]), float(principal_point[1])

    l_a, l_b, l_e = side_a.line(), side_b.line(), far_edge.line()

    fx_fixed = (float(focal_px) if focal_px is not None else
                focal_from_fov(float(fov_x_deg), img_w) if fov_x_deg is not None
                else None)
    fy_fixed = (float(focal_y_px) if focal_y_px is not None else
                focal_from_fov(float(fov_y_deg), img_h) if fov_y_deg is not None
                else None)

    if anamorphic and near_edge is None:
        anamorphic = False
        warnings.append(
            "Separate horizontal / vertical FOV needs all four edges: three "
            "strokes give a homography with a free scale, so fx and fy cannot "
            "be told apart. Trace the near edge. Assuming square pixels.")

    fov_determined = True
    anamorphic_solved = False
    fov_sigma: Optional[float] = 0.0
    fov_y_sigma: Optional[float] = 0.0

    got = _solve_focals(side_a, side_b, far_edge, near_edge, width, length,
                        cx, cy, img_w, img_h, fx_fixed, fy_fixed, anamorphic)
    if got is None:
        fov_determined = False
        fov_sigma = fov_y_sigma = None
        f = fx_fixed if fx_fixed is not None else focal_from_fov(assumed_fov_x_deg, img_w)
        fx, fy = f, (fy_fixed if fy_fixed is not None else f)
        if near_edge is None:
            warnings.append(
                "FOV is not recoverable from three strokes alone: every focal "
                "length reproduces them exactly. Enter the FOV, or trace any "
                "visible piece of the near edge, to pin it down.")
        elif anamorphic:
            warnings.append(
                "The four strokes do not resolve fx and fy: no real camera "
                "reproduces them as a rectangle of this aspect ratio. Check the "
                "width / length you entered, then re-trace. Assuming square pixels.")
        else:
            warnings.append(
                "The near-edge stroke is (near-)parallel to the far edge, so the "
                "two vanishing points do not constrain the focal length. "
                "Falling back to the assumed FOV.")
    else:
        fx, fy = got
        anamorphic_solved = anamorphic
        if fx_fixed is None or fy_fixed is None:
            fov_sigma, fov_y_sigma = _intrinsics_uncertainty(
                side_a, side_b, far_edge, near_edge, width, length,
                cx, cy, img_w, img_h, fx_fixed, fy_fixed, anamorphic)
            worst = max((s for s in (fov_sigma, fov_y_sigma) if s is not None),
                        default=None)
            if worst is None or not math.isfinite(worst) or worst > 5.0:
                warnings.append(
                    "FOV is poorly conditioned here (+/- %s deg per pixel of stroke "
                    "error). %s Trace the edges longer, or enter the FOV instead."
                    % ("inf" if worst is None or not math.isfinite(worst)
                       else "%.1f" % worst,
                       "The rim is close to fronto-parallel, which is what "
                       "separating fx from fy needs most." if anamorphic else
                       "The near and far strokes are nearly parallel."))

    f = fx
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    solved = _pose_from_focal(l_a, l_b, l_e, width, length, k, warnings)
    if solved is None:
        raise ValueError("Reconstruction failed:\n  " + "\n  ".join(warnings or ["unknown"]))
    r_cr, t_cr, _ = solved

    corners_rect = rect_corners(width, length)

    # Attach each stroke to the rectangle edge it traced.  The two side strokes
    # are told apart by which side of the rect centre their 3D line landed on.
    side_a_x = _stroke_side(k, r_cr, t_cr, corners_rect, side_a)
    strokes = [("far", far_edge)]
    strokes.append(("right" if side_a_x >= 0 else "left", side_a))
    strokes.append(("left" if side_a_x >= 0 else "right", side_b))
    if near_edge is not None:
        strokes.append(("near", near_edge))

    # Let the polish move a focal only where the strokes actually observe it.
    observable = near_edge is not None and fov_determined
    if not observable:
        focal_mode = "none"
    elif anamorphic_solved:
        focal_mode = {(True, True): "xy", (True, False): "x",
                      (False, True): "y", (False, False): "none"}[
            (fx_fixed is None, fy_fixed is None)]
    else:
        focal_mode = "none" if (fx_fixed is not None or fy_fixed is not None) else "iso"
    if refine:
        k, r_cr, t_cr = _refine(k, r_cr, t_cr, corners_rect, strokes,
                                cx, cy, focal_mode, img_w, img_h)
        fx, fy = float(k[0, 0]), float(k[1, 1])

    res, per_stroke = _stroke_residuals(k, r_cr, t_cr, corners_rect, strokes)
    rms = float(np.sqrt(np.mean(res ** 2))) if res.size else 0.0

    corners_cam = (r_cr @ corners_rect.T).T + t_cr
    cam_pos = -r_cr.T @ t_cr

    if cam_pos[2] <= 0:
        warnings.append("Camera solved below the rim plane; the on-top branch was "
                        "not reachable with these strokes.")
    if not np.all(corners_cam[:, 2] > 0):
        warnings.append("Some rim corners lie behind the camera — the overlay is "
                        "clipped at the near plane.")
    if rms > 2.0:
        warnings.append("Residual is large (%.1f px): the strokes are not consistent "
                        "with a rectangle of this aspect ratio." % rms)

    return Solution(
        focal_px=fx,
        focal_y_px=fy,
        principal_point=(cx, cy),
        image_size=(img_w, img_h),
        fov_x_deg=fov_from_focal(fx, img_w),
        fov_y_deg=fov_from_focal(fy, img_h),
        fov_determined=fov_determined,
        fov_uncertainty_deg=fov_sigma,
        fov_y_uncertainty_deg=fov_y_sigma,
        anamorphic_solved=anamorphic_solved,
        position=cam_pos,
        hpr_deg=_panda_hpr_from_cv(r_cr.T),
        r_cam_from_rect=r_cr,
        t_cam_from_rect=t_cr,
        width=float(width),
        length=float(length),
        corners_rect=corners_rect,
        corners_cam=corners_cam,
        corners_img=project_points(k, corners_cam),
        height_above_plane=float(cam_pos[2]),
        distance_to_centre=float(np.linalg.norm(cam_pos)),
        residual_px=rms,
        residuals=per_stroke,
        warnings=warnings,
    )


def _stroke_side(k, r_cr, t_cr, corners_rect, seg: Segment) -> float:
    """+1 if the stroke traced the +X long edge, -1 if it traced the -X one."""
    best, best_d = 1.0, None
    for sign, (i, j) in ((1.0, _EDGE_CORNERS["right"]), (-1.0, _EDGE_CORNERS["left"])):
        pa = r_cr @ corners_rect[i] + t_cr
        pb = r_cr @ corners_rect[j] + t_cr
        line = _norm_line(np.cross(k @ pa, k @ pb))
        d = abs(float(line @ _h(seg.p0))) + abs(float(line @ _h(seg.p1)))
        if best_d is None or d < best_d:
            best, best_d = sign, d
    return best


# --------------------------------------------------------------------------
# persistence helper (shared with the test UI)
# --------------------------------------------------------------------------

def save_session(path, strokes: dict, width: float, length: float,
                 fov_x_deg: Optional[float], image_path: Optional[str] = None,
                 volume: Optional[float] = None,
                 depth: Optional[float] = None) -> None:
    data = {
        "image": image_path,
        "width": width,
        "length": length,
        "volume": volume,
        "depth": depth,
        "fov_x_deg": fov_x_deg,
        "strokes": {k: (v.as_list() if v is not None else None) for k, v in strokes.items()},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_session(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["strokes"] = {k: (Segment.from_list(v) if v else None)
                       for k, v in data.get("strokes", {}).items()}
    return data


# --------------------------------------------------------------------------
# self-test: synthesise a camera, trace three partial edges, recover it
# --------------------------------------------------------------------------

def _synthesise(rng, image_size=(1919, 1106), fov_x=90.0, pixel_aspect=1.0):
    """Random plausible over-the-rim camera + the pixel strokes it would produce.

    `pixel_aspect` = fy/fx, so values != 1 model a squeezed / stretched frame.
    """
    w, h = image_size
    f = focal_from_fov(fov_x, w)
    fy = f * pixel_aspect
    k = np.array([[f, 0.0, w * 0.5], [0.0, fy, h * 0.5], [0.0, 0.0, 1.0]])

    width = rng.uniform(2.0, 3.0)
    length = rng.uniform(5.0, 9.0)

    # Camera behind the near edge, above the rim, looking down the body.
    cam = np.array([rng.uniform(-1.5, 1.5),
                    -length * 0.5 - rng.uniform(0.3, 3.0),
                    rng.uniform(1.0, 4.0)])
    centre_dir = -cam
    heading = math.degrees(math.atan2(-centre_dir[0], centre_dir[1]))
    pitch = math.degrees(math.asin(np.clip(centre_dir[2] / np.linalg.norm(centre_dir), -1, 1)))
    roll = rng.uniform(-8.0, 8.0)
    heading += rng.uniform(-12.0, 12.0)
    pitch += rng.uniform(-8.0, 8.0)

    r_panda = matrix_from_hpr(heading, pitch, roll)
    r_world_from_cv = r_panda @ _CV_TO_PANDA
    r_cr = r_world_from_cv.T                       # rect axes in camera coords
    t_cr = -r_cr @ cam

    corners = rect_corners(width, length)
    cam_pts = (r_cr @ corners.T).T + t_cr
    if np.any(cam_pts[:, 2] <= 0.05):
        return None
    px = project_points(k, cam_pts)
    if not np.all(np.isfinite(px)):
        return None

    def partial(i, j, lo, hi):
        a, b = px[i], px[j]
        return Segment(a + (b - a) * lo, a + (b - a) * hi)

    return dict(k=k, f=f, fy=fy, width=width, length=length, image_size=image_size,
                cam=cam, hpr=(heading, pitch, roll), px=px,
                # sides traced over random sub-spans, far edge traced fully
                side_a=partial(1, 2, rng.uniform(0.0, 0.35), rng.uniform(0.6, 1.0)),
                side_b=partial(3, 0, rng.uniform(0.0, 0.35), rng.uniform(0.6, 1.0)),
                far=partial(2, 3, 0.0, 1.0),
                near=partial(0, 1, rng.uniform(0.0, 0.45), rng.uniform(0.6, 1.0)))


def _selftest(trials: int = 300, noise_px: float = 0.0, seed: int = 7) -> int:
    rng = np.random.default_rng(seed)
    stats = {"known-fov": [], "solved-fov": []}
    fov_err, fov_err_ok, flagged = [], [], 0
    done = 0
    while done < trials:
        scene = _synthesise(rng)
        if scene is None:
            continue
        done += 1

        def jitter(seg):
            if noise_px <= 0:
                return seg
            return Segment(seg.p0 + rng.normal(0, noise_px, 2),
                           seg.p1 + rng.normal(0, noise_px, 2))

        sa, sb = jitter(scene["side_a"]), jitter(scene["side_b"])
        fe, ne = jitter(scene["far"]), jitter(scene["near"])

        # (1) FOV known, three strokes
        sol = reconstruct(sa, sb, fe, scene["width"], scene["length"],
                          scene["image_size"], focal_px=scene["f"])
        stats["known-fov"].append(np.linalg.norm(sol.position - scene["cam"]))

        # (2) FOV unknown, four strokes
        sol2 = reconstruct(sa, sb, fe, scene["width"], scene["length"],
                           scene["image_size"], near_edge=ne)
        stats["solved-fov"].append(np.linalg.norm(sol2.position - scene["cam"]))
        err = abs(sol2.fov_x_deg - fov_from_focal(scene["f"], scene["image_size"][0]))
        fov_err.append(err)
        # A run is "flagged" when the module itself warned that the FOV here is
        # ill-conditioned; those are the ones the operator is told not to trust.
        sigma = sol2.fov_uncertainty_deg
        if sigma is None or not math.isfinite(sigma) or sigma > 5.0:
            flagged += 1
        else:
            fov_err_ok.append(err)

    print("self-test: %d scenes, stroke noise %.2f px" % (trials, noise_px))
    bad = 0
    # Body is roughly 2.5 x 7 units, so these limits are ~cm-scale at 0.5 px.
    limits = {"known-fov": 0.05 + 0.35 * noise_px,
              "solved-fov": 0.10 + 1.50 * noise_px}
    for name, errs in stats.items():
        errs = np.asarray(errs)
        p95 = np.percentile(errs, 95)
        ok = p95 <= limits[name]
        bad += int(not ok)
        print("  %-10s position error  mean %.3e  p95 %.3e  max %.3e   [%s <= %.3f]"
              % (name, errs.mean(), p95, errs.max(), "ok" if ok else "OVER", limits[name]))
    fov_err = np.asarray(fov_err)
    print("  fov error (4 strokes, all)       mean %.3e  p95 %.3e  max %.3e deg"
          % (fov_err.mean(), np.percentile(fov_err, 95), fov_err.max()))
    if fov_err_ok:
        fov_err_ok = np.asarray(fov_err_ok)
        print("  fov error (well-conditioned)     mean %.3e  p95 %.3e  max %.3e deg"
              % (fov_err_ok.mean(), np.percentile(fov_err_ok, 95), fov_err_ok.max()))
    print("  ill-conditioned & warned about: %d / %d" % (flagged, trials))
    return bad


def _selftest_anamorphic(trials: int = 300, noise_px: float = 0.0,
                         seed: int = 11) -> int:
    """Four strokes, squeezed frame, both focals unknown: can we recover both?"""
    rng = np.random.default_rng(seed)
    err_x, err_y, err_ar, err_pos, ok_ar, flagged = [], [], [], [], [], 0
    done = 0
    while done < trials:
        aspect = float(rng.uniform(0.7, 1.4))       # squeeze or stretch the frame
        scene = _synthesise(rng, fov_x=float(rng.uniform(50.0, 110.0)),
                            pixel_aspect=aspect)
        if scene is None:
            continue
        done += 1

        def jitter(seg):
            if noise_px <= 0:
                return seg
            return Segment(seg.p0 + rng.normal(0, noise_px, 2),
                           seg.p1 + rng.normal(0, noise_px, 2))

        w, h = scene["image_size"]
        sol = reconstruct(jitter(scene["side_a"]), jitter(scene["side_b"]),
                          jitter(scene["far"]), scene["width"], scene["length"],
                          scene["image_size"], near_edge=jitter(scene["near"]),
                          anamorphic=True)
        ex = abs(sol.fov_x_deg - fov_from_focal(scene["f"], w))
        ey = abs(sol.fov_y_deg - fov_from_focal(scene["fy"], h))
        ea = abs(sol.pixel_aspect - aspect)
        err_x.append(ex); err_y.append(ey); err_ar.append(ea)
        err_pos.append(np.linalg.norm(sol.position - scene["cam"]))
        sig = [s for s in (sol.fov_uncertainty_deg, sol.fov_y_uncertainty_deg)
               if s is not None]
        worst = max(sig) if sig else None
        if not sol.anamorphic_solved or worst is None or not math.isfinite(worst) \
                or worst > 5.0:
            flagged += 1
        else:
            ok_ar.append(ea)

    print("anamorphic self-test: %d squeezed scenes, stroke noise %.2f px"
          % (trials, noise_px))
    for name, e in (("h-fov err (deg)", err_x), ("v-fov err (deg)", err_y),
                    ("pixel aspect err", err_ar), ("position err", err_pos)):
        e = np.asarray(e)
        print("  %-18s mean %.3e  p95 %.3e  max %.3e"
              % (name, e.mean(), np.percentile(e, 95), e.max()))
    if ok_ar:
        ok_ar = np.asarray(ok_ar)
        print("  %-18s mean %.3e  p95 %.3e  max %.3e"
              % ("  (well-cond.)", ok_ar.mean(), np.percentile(ok_ar, 95), ok_ar.max()))
    print("  ill-conditioned & warned about: %d / %d" % (flagged, trials))
    # Judge the runs the module tells the operator to trust.  Separating fx from
    # fy leans entirely on the weaker of the two conic constraints, so it
    # amplifies stroke noise far more than the square-pixel solve does — which
    # is exactly why the ill-conditioned ones are flagged rather than returned
    # with a straight face.
    limit = 1e-6 + noise_px * 0.40
    p95 = np.percentile(np.asarray(ok_ar if len(ok_ar) else err_ar), 95)
    ok = p95 <= limit
    print("  well-conditioned aspect p95 %.3e <= %.3e ? %s"
          % (p95, limit, "ok" if ok else "OVER"))
    return int(not ok)


if __name__ == "__main__":
    import sys
    failures = 0
    for noise in (0.0, 0.5, 1.5):
        failures += _selftest(noise_px=noise)
        print()
    for noise in (0.0, 0.25, 0.5):
        failures += _selftest_anamorphic(noise_px=noise)
        print()
    print("FAIL" if failures else "OK")
    sys.exit(1 if failures else 0)
