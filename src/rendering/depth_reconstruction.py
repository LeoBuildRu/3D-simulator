# depth_reconstruction.py
# ---------------------------------------------------------------------------
# Local fill-landscape reconstruction from a stand snapshot's depth map.
#
# Flow (driven from the UI):
#   1. The user picks a stand snapshot and lines the live camera up with the
#      reference overlay so the 3D world matches the photo.
#   2. Anchor points are obtained — either by clicking on the truck in the
#      viewport, or by the automatic grid-of-rays search. These points are
#      used ONLY to calibrate depth; they do NOT bound the mesh.
#   3. Each anchor point is ray-cast against the loaded truck model to recover
#      its true 3D position (metric anchor points).
#   4. The depth map (linear, 8-bit grayscale) is calibrated to metric using
#      the anchors — a least-squares fit  Z = A*d + B  maps the normalised
#      depth value d∈[0,1] to a forward (perpendicular) distance from the
#      camera. The fit recovers min/max depth and is sign-agnostic, so it
#      doesn't matter whether "near" is bright or dark.
#   5. A grid is sampled over the REGION (the mask, or the whole frame when
#      there is no mask); every sample is unprojected through the current
#      camera lens at its calibrated depth, giving a 3D surface. That surface
#      is turned into a Panda mesh and added to the scene, anchored to the
#      current camera state.
#
# Assumptions (documented so they're easy to revisit):
#   A1  The mesh region is defined by the per-snapshot mask (alpha) if one
#       exists, otherwise the whole frame is reconstructed. The anchor points
#       only calibrate depth — they never bound the region.
#   A2  The depth PNG is linear and normalised by /255.
#   A3  "depth" = forward/perpendicular distance from the camera plane
#       (Panda camera looks down +Y), consistent with linear depth.
#   A4  The color overlay and depth map cover the same FOV (same scene).
#       The overlay is shown KeepAspectRatio (fills film width, letterboxed
#       vertically). fyh = window_aspect / color_aspect is the vertical
#       half-extent in film space. Normalized UV coords (u,v)∈[0,1]² are
#       shared between color and depth, so the depth resolution can differ.
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import math

import numpy as np

try:
    import trimesh
except ImportError:                       # pragma: no cover
    trimesh = None

from panda3d.core import (
    CollisionTraverser, CollisionHandlerQueue, CollisionNode, CollisionRay,
    CollisionPolygon, GeomNode, GeomVertexReader, Point2, Point3, LPoint3,
    Vec3, Material, BitMask32, TransparencyAttrib,
    GeomVertexFormat, GeomVertexData, Geom, GeomTriangles, GeomVertexWriter,
)


class DepthReconstructor:
    """Owns the N-point picking interaction and the depth->mesh pipeline."""

    # Any number of points is allowed; this many are required to calibrate
    # the depth law Z = A*d + B. The points ONLY calibrate depth — the mesh
    # region is defined by the mask (or the whole frame). Picking is finished
    # by the user (RMB / Esc).
    MIN_POINTS = 2
    GRID = 200          # mesh resolution across the region (GRID x GRID quads)
    # Texture tiling for the reconstructed surface: UV units per world metre.
    # UVs are a top-down planar projection of global XY, so the texture keeps
    # a real-world scale regardless of mesh size. Bump >1 to tile tighter.
    UV_PER_METER = 0.5
    # Local depth calibration: each control point corrects the metric depth
    # in its neighbourhood. We fit a global linear law Z = A*d + B over all
    # points, then interpolate the per-point residuals across the region with
    # inverse-distance weighting (Shepard). IDW_POWER controls locality —
    # higher = each point's influence stays tighter to its own area.
    IDW_POWER = 2.0
    IDW_EPS = 1e-6
    # Light surface smoothing: number of 3x3 mask-aware averaging passes over
    # the grid depths before unprojection. Evens out the terracing caused by
    # 8-bit depth quantization. 0 = off; 1-3 = gentle. Only averages valid
    # (kept) neighbours, so it doesn't bleed across the mask boundary.
    SMOOTH_ITERS = 2
    # ==================================================================
    # PIPELINE (mask-free). The reconstruction now runs as:
    #   Stage 1  build the relief over the WHOLE frame, cutting off
    #            over-long polygons (steep stretched triangles) and the
    #            cells around them (LONGPOLY_*).
    #   Stage 2  Boolean with the napolnitel (truck-body filler) mesh —
    #            keep only the relief that sits INSIDE the truck body,
    #            discarding all pile geometry spilling outside
    #            (CLIP_TO_NAPOLNITEL).
    #   Stage 3  (disabled by default) the legacy extrapolation + sealed-
    #            solid Boolean DIFFERENCE + volume measurement. Flip
    #            ENABLE_EXTRAPOLATION / ENABLE_VOLUME to restore it.
    # ------------------------------------------------------------------
    # Stage 1 — long-polygon cutoff. The relief is sampled on a grid that is
    # uniform in SCREEN space, so each cell spans a world distance that grows
    # with its distance from the camera. A grid edge is cut when it is too long
    # by EITHER of two tests (whichever triggers):
    #
    #   • RELATIVE — edge_len / Z (forward distance) > LONGPOLY_MAX_EDGE_RATIO.
    #     Scale-invariant: a smoothly sampled surface has edge_len/Z ≈ the
    #     angular pixel pitch (constant at every depth), while a depth
    #     discontinuity (silhouette / pile edge / wall) spikes well above it.
    #     This catches stretched bridging triangles at ANY distance — including
    #     near silhouettes whose edges are absolutely short. Lower = more
    #     aggressive. None disables this test.
    #
    #   • ABSOLUTE — edge_len > LONGPOLY_MAX_EDGE_M metres. The relative test
    #     alone permits edges up to ratio·Z, so on far geometry (large Z) it
    #     tolerates absolutely huge polygons (e.g. 1.2 m at Z=10 m). This hard
    #     cap bounds the world size of any kept polygon, so far objects don't
    #     keep over-long triangles. Make it generous enough not to chew up
    #     honest far surface but small enough to kill stretched ones. None
    #     disables this test.
    #
    # The offending cell is dropped, and so is every cell within
    # LONGPOLY_PROPAGATE_CELLS grid cells of it (Chebyshev radius) so the torn
    # region is cleanly removed instead of leaving a jagged fringe (0 = drop
    # only the offending cells). LONGPOLY_CUTOFF=False disables Stage 1.
    LONGPOLY_CUTOFF = True
    LONGPOLY_MAX_EDGE_RATIO = 0.15
    LONGPOLY_MAX_EDGE_M = 0.15
    LONGPOLY_PROPAGATE_CELLS = 1
    # Final mesh cleanup (runs on the finished triangle mesh — after the
    # truck-body Boolean and, if enabled, after extrapolation + volume, so it
    # catches near-vertical walls no matter which stage created them; the
    # volume number is computed before cleanup and is kept).
    #
    # STEEP cutoff: a triangle whose surface tilts more than STEEP_MAX_ANGLE_DEG
    # from horizontal (|normal·Zup| < cos(angle)) is near-vertical — a candidate
    # occlusion wall / silhouette artifact. Every such triangle, plus every
    # triangle whose centroid lies within STEEP_REMOVE_RADIUS_M metres of one,
    # is removed. The radius carves a clean margin around each strong drop.
    # STEEP_CUTOFF=False or STEEP_MAX_ANGLE_DEG=None disables it.
    #
    # NEAR-WALL ONLY (STEEP_EDGE_ONLY): the angle-of-repose cut is applied ONLY
    # within STEEP_NEAR_WALL_M metres of the nearest truck-body wall (the
    # napolnitel side walls, measured as XY distance to the truck footprint
    # perimeter). Steep faces there are occlusion artifacts where the load meets
    # the bed side; a steep face farther INTO the load is a real feature (a
    # ridge / crater wall) and is kept, so the cut never tears a chunk out of
    # the middle of the pile. To avoid punching holes, only the steep faces that
    # are connected (through other steep, near-wall faces) to an open mesh
    # boundary are removed — the removed region always reaches the rim. A fully
    # closed mesh (the volume solid, no open boundary) instead drops every
    # near-wall steep face directly. Set STEEP_EDGE_ONLY=False to remove every
    # steep face regardless of distance to the walls.
    STEEP_CUTOFF = True
    STEEP_MAX_ANGLE_DEG = 70.0
    STEEP_REMOVE_RADIUS_M = 0.01
    STEEP_EDGE_ONLY = True
    STEEP_NEAR_WALL_M = 0.5
    # Small-cluster removal: after the steep cut, drop every connected group of
    # triangles whose world bounding box is smaller than MIN_CLUSTER_SIZE_M in
    # its largest dimension — isolated specks left behind by the cutoffs. Set
    # to 0/None to keep all clusters.
    MIN_CLUSTER_SIZE_M = 1.00
    # Stage 0 — background removal by flood fill. A depth map often has a big
    # gray-ish area (sky / far wall above the truck) that isn't part of the
    # load. A paint-bucket flood fill is seeded from the image border and
    # grows across pixels whose 8-bit depth value stays within
    # BG_FLOOD_THRESHOLD of the SEED (border) value — a FIXED-RANGE fill, so
    # the flood stays in the gray/dark background and stops at the much
    # brighter (nearer) truck instead of bleeding through its soft silhouette.
    # Each disjoint border segment seeds its own fill, so backgrounds with
    # several gray shades are all caught. Only flooded blobs covering at least
    # BG_MIN_AREA_FRAC of the frame are treated as background — small fills
    # (e.g. a seed landing on real geometry) are ignored. BG_SEED_BORDERS
    # picks which edges to seed from ("top", "bottom", "left", "right").
    # Set BG_FLOODFILL=False to disable.
    BG_FLOODFILL = True
    BG_FLOOD_THRESHOLD = 10         # 0..255 depth units, tolerance from seed
    BG_MIN_AREA_FRAC = 0.02         # ignore flooded blobs smaller than this
    BG_SEED_BORDERS = ("top",)      # edges to seed the flood from
    # Stage 2 — clip the relief to the truck body. Every relief vertex is
    # tested for containment in the closed napolnitel mesh (TONAR_OBJ_*,
    # transform applied); a triangle survives only if all 3 of its corners
    # are inside. This is the mask-free replacement for the old 2D fill
    # mask: the truck-body solid bounds the mesh in 3D instead.
    CLIP_TO_NAPOLNITEL = True
    # Stage 3 — legacy extrapolation + sealed-solid Boolean DIFFERENCE +
    # volume. Disabled for now; set both True to bring the old behaviour
    # back (it rebuilds the geometry from the grid, see reconstruct()).
    ENABLE_EXTRAPOLATION = False
    ENABLE_VOLUME = ENABLE_EXTRAPOLATION
    # ==================================================================
    # Per-snapshot fill mask (<depth>-mask.png, RGBA). DISABLED by default
    # now (USE_MASKS=False) — the truck-body Boolean (Stage 2) bounds the
    # mesh instead. When turned back on and a mask exists, the mesh covers
    # exactly the masked pixels (alpha > MASK_ALPHA_MIN); otherwise the
    # WHOLE frame is reconstructed.
    USE_MASKS = False
    MASK_SUFFIX = "-mask.png"
    MASK_ALPHA_MIN = 0.5
    # The mesh-cell rule keeps a cell only when all 4 of its corners are in the
    # mask, which erodes the kept region inward by ~1 grid cell along the whole
    # boundary (plus the grid-node quantization of the edge). Dilate the
    # sampled mask by this many cells first so the surface reaches the true
    # mask edge instead of being trimmed short. 0 = off.
    MASK_DILATE_CELLS = 1
    # Extrapolation: after the masked relief is built, extend it to a fixed
    # axis-aligned rectangle in world XY (metres) by mirror-tiling the
    # pattern outside the mask's XY bbox. The trend (plane fit) is added
    # back so the seam stays C0-continuous. Set to None to disable and
    # keep the original mask-only mesh.
    TARGET_SIZE_M = (3.0, 6.0)  # XY-прямоугольник (м), покрывает MAZ ≈ 2.14×5.10
    # Density of the target grid (vertices per metre along X and Y).
    TARGET_RES_PER_M = 25
    # Light smoothing of the target grid AFTER extrapolation (mask data is
    # preserved — only extrapolated cells are smoothed). 0 = off.
    TARGET_SMOOTH_ITERS = 0  # сглаживание целевой сетки временно отключено
    # Outside the mask the relief descends. The descent rate at each external
    # cell is the steeper of two contenders:
    #   • the LOCAL OUTWARD GRADIENT at the nearest in-mask boundary cell —
    #     captures how fast the visible face was already dropping. A steep
    #     pile face that faces the front wall of the truck keeps falling at
    #     its observed angle until it hits the floor (no plateau, no fake
    #     "second pile").
    #   • tan(EXTRAP_ANGLE_DEG) — angle-of-repose fallback for cases where
    #     the boundary gradient is flat or rising (e.g. behind a hidden
    #     peak), so the relief still drops at a natural granular slope
    #     instead of plateauing.
    # 30-45° covers most granular materials; the gradient takes over when
    # the visible face is steeper than this.
    EXTRAP_ANGLE_DEG = 35.0
    # Hard cap on the descent rate so a noisy one-sided boundary gradient
    # can't blow up Z to nonsense values. Vertical = 90° → tan=∞; capping
    # at 75° keeps even single-pixel artifacts bounded.
    EXTRAP_MAX_ANGLE_DEG = 75.0
    # Floor for the extrapolated relief: external cells are clipped at
    # min(visible Z) − EXTRAP_FLOOR_FROM_MIN_M. None (default) → no clamp;
    # the MAZ Boolean bounds the volume from below. A value here can
    # introduce a "tent" plateau when the visible mask only shows the top
    # of the pile (so min(visible Z) is well above the actual truck floor)
    # — keep at None unless you have a reason to clip earlier than the
    # Boolean.
    EXTRAP_FLOOR_FROM_MIN_M = None
    # Optional absolute Z floor (metres). Takes effect when
    # EXTRAP_FLOOR_FROM_MIN_M is None — useful when you know the truck
    # floor in world Z.
    EXTRAP_FLOOR_M = None
    # Residual texture preservation: the visible relief's high-frequency
    # variation (original Z minus its smoothed trend) is mirror-tiled onto
    # the extrapolated area so the hidden back side keeps the look of the
    # visible front, instead of being a featureless smooth slope. Weight
    # 0 = pure descent (smooth), 1 = full mirror-residual amplitude.
    EXTRAP_TEXTURE_WEIGHT = 0.7
    # Exponential decay scale (metres) of the texture amplitude with
    # distance from the mask. Cells right next to the mask boundary get
    # almost the full residual; cells far away fade smoothly to the pure
    # descent baseline. exp(−d / decay).
    EXTRAP_TEXTURE_DECAY_M = 1.5
    # Smoothing iterations used to extract the low-frequency trend from
    # the in-mask field (residual = original − trend). Heavier smoothing
    # makes the residual capture only fine texture; lighter smoothing
    # keeps medium-scale features.
    EXTRAP_TEXTURE_SMOOTH_ITERS = 10
    # ------------------------------------------------------------------
    # Clipping the extrapolated relief into the truck filler volume.
    # After the rectangular relief is built, a headless Blender 2.70
    # instance is invoked: the relief is sealed into a closed solid
    # representing the empty space ABOVE it (floor = heightfield,
    # ceiling = flat plane above tonar, walls = perimeter), and a
    # Boolean DIFFERENCE is applied: tonar_napolnitel − sealed_solid.
    # The result is a closed manifold of the part of the container BELOW
    # the relief — the filler whose volume can be measured directly via
    # the divergence-theorem formula. This mirrors the server's
    # mesh_reconstruction.cpp + boolean_operations.cpp pipeline.
    # TONAR_OBJ_REL_PATH is resolved relative to the project root;
    # empty/None disables the step (the un-clipped open relief is kept).
    #
    # NOTE: both meshes are sent in their native coordinates. If the OBJ is
    # in a different frame than the Panda render world (e.g. Y-up vs Z-up),
    # set TONAR_OBJ_TRANSFORM to a 4×4 numpy matrix that maps OBJ-local
    # coords into world coords before sending. None = identity.
    TONAR_OBJ_REL_PATH = "assets/height_examples/stand/MAZ_napolnitel.obj"
    # tonar_napolnitel.obj is exported Y-up (Blender convention); Panda is
    # Z-up. Composed transform: Y↔Z swap (Y-up → Z-up), then R_z(180°) to
    # flip the truck's front/back. Net mapping: (x, y, z) → (−x, −z, y).
    # Set to None to skip any transform, or override at runtime for a
    # different alignment.
    TONAR_OBJ_TRANSFORM = [
        [-1.0,  0.0,  0.0, 0.0],
        [ 0.0,  0.0, -1.0, 0.0],
        [ 0.0,  1.0,  0.0, 0.0],
        [ 0.0,  0.0,  0.0, 1.0],
    ]
    # Boolean engine for the volume step (tonar − sealed_solid). "auto" uses
    # the in-process manifold3d library when it's importable (fast, no external
    # dependency) and only falls back to Blender if manifold3d is missing or
    # rejects the input. "manifold" / "blender" force one engine. If the chosen
    # engine(s) fail, Stage 3 still clips the extrapolated relief to the truck
    # body with a point-in-mesh test so the geometry never overhangs the bed
    # (only the volume number is then unavailable).
    BOOLEAN_ENGINE = "auto"         # "auto" | "manifold" | "blender"
    # Path to a local Blender 2.70 install used as the boolean fallback.
    # Empty/None or a missing file disables the Blender path. Timeout in s.
    BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender\blender.exe"
    BLENDER_TIMEOUT_S = 120
    # --- Automatic reference-point search ----------------------------
    # Instead of manual picking, cast an AUTO_GRID x AUTO_GRID grid of rays
    # across the screen; every ray that hits the truck is a candidate anchor.
    # Candidates are then rejected robustly (see _reject_outliers):
    #   • global  — a point whose depth->metric pair breaks the overall
    #     Z = A*d + B law by more than AUTO_GLOBAL_K·MAD (floored at
    #     AUTO_GLOBAL_ABS metres) is dropped (catches fill / 2D-3D mismatch);
    #   • local   — a point whose residual deviates from its neighbours'
    #     median by more than AUTO_LOCAL_THRESH metres is dropped (the
    #     "5 wall points agree, the 6th is off" case).
    # Thresholds are deliberately tight — culling many points is fine.
    AUTO_GRID = 70
    # Truck depth is found by projecting its vertices into a supersampled
    # z-buffer (nearest forward depth wins) instead of ray-casting each grid
    # point — O(vertices) once vs O(rays × polygons). BUF = AUTO_GRID * this.
    AUTO_BUF_SUPERSAMPLE = 8
    AUTO_REJECT_ITERS = 4
    AUTO_GLOBAL_K = 3.0
    AUTO_GLOBAL_ABS = 0.10      # metres — residual floor for global reject
    AUTO_LOCAL_WINDOW = 3       # grid-cell radius of the local neighbourhood
    AUTO_LOCAL_THRESH = 0.06    # metres — local residual deviation cutoff
    AUTO_LOCAL_MIN_NB = 5       # need this many neighbours to judge a point
    # The bed FLOOR must never be an anchor: fill can sit anywhere on it, so a
    # floor point's depth (fill surface) won't match the truck (floor). Only
    # near-vertical surfaces (the walls) are reliable. We compute the truck
    # surface normal at each candidate and drop those whose normal is more
    # vertical (|n·Zup|) than this — i.e. near-horizontal floor / shelves.
    AUTO_REJECT_FLOOR = True
    AUTO_FLOOR_MAX_NZ = 0.55
    # Build the debug point-grid (green = used as anchor, red = rejected).
    VISUALIZE_POINTS = False
    # Diagnostic visualization: colour EVERY truck-hit point by its fate so we
    # can see which stage drops it — green = anchor, blue = floor-filtered,
    # red = robust-rejected. Off = show only accepted (green).
    AUTO_VIZ_DIAGNOSTIC = True
    # A dedicated, definitely-non-zero collide mask. We stamp it onto the
    # truck's visible geometry and use the same bit for the picking ray, so
    # the hit test works regardless of whatever into-mask the .bam /
    # RenderPipeline left on the GeomNodes (the default GeomNode mask is
    # bit 20, but processed geometry often ends up with 0).
    PICK_MASK = BitMask32.bit(1)

    def __init__(self, panda_app):
        self.panda_app = panda_app

        # Own DirectObject so our event hooks (mouse1/mouse3/escape) live on
        # a different object than the ShowBase app — otherwise accepting
        # "mouse3" here would clobber the FlyCamera's RMB-look binding.
        try:
            from direct.showbase.DirectObject import DirectObject
            self._do = DirectObject()
        except Exception:
            self._do = None

        self._depth_path = ""
        self._color_path = ""

        self._picking = False
        # When set, finishing a pick hands the collected film coords to this
        # callback (used to bind anchor points to a camera preset) instead of
        # running a reconstruction.
        self._commit_cb = None
        self._films: list[tuple[float, float]] = []   # clicked film coords
        self._hits: list[Point3] = []                 # raycast world points
        # Film coords of the last successful pick — reused to auto-reconstruct
        # other snapshots from the SAME fixed stand camera (the bed corners
        # sit at the same screen positions, only the depth map differs).
        self._saved_films: list[tuple[float, float]] = []

        self._mesh_node = None                        # last reconstruction
        self._tonar_debug_node = None                 # translucent .obj overlay
        self._truck_np = None                         # cached pick target
        self._truck_collider = None                   # CollisionPolygon node
        self._collider_truck_id = None                # whose collider we built

        # Automatic point search / visualization state.
        self._truck_verts = None                      # cached world vertices
        self._truck_verts_id = None
        self._auto_mode = False                       # last build was auto?
        self._viz_on = bool(self.VISUALIZE_POINTS)
        self._viz_node = None
        self._auto_viz_data = None                    # [(Point3, accepted)]

        # UI callbacks (wired by MainWindow; safe to touch Qt — Panda is
        # stepped on the Qt thread).
        self.on_count = None         # callable(n_points: int)
        self.on_finished = None      # callable(success: bool, info: dict)
        self.on_picking_state = None # callable(active: bool)
        self.on_log = None           # callable(msg: str)

    # ==================================================================
    # Public API
    # ==================================================================
    def set_source(self, depth_path: str, color_path: str = "") -> None:
        """Point the reconstructor at the active stand snapshot."""
        self._depth_path = depth_path or ""
        self._color_path = color_path or ""

    def is_picking(self) -> bool:
        return self._picking

    def start_picking(self, commit_cb=None) -> None:
        """Enter point-picking mode.

        Normally finishing the pick (RMB / Esc) runs a reconstruction. If
        `commit_cb` is given the collected film coords are handed back to it
        instead — used to bind anchor points to a camera preset, where no
        depth map is required (we only raycast against the truck)."""
        if self._picking:
            return
        self._commit_cb = commit_cb
        # Preset binding only needs the truck collider, not a depth map.
        if commit_cb is None and (
                not self._depth_path or not os.path.exists(self._depth_path)):
            self._log("Нет карты глубины для реконструкции.")
            self._commit_cb = None
            return
        self._picking = True
        self.clear_points()
        app = self.panda_app
        # Resolve + prepare the truck up front so the first click isn't slow
        # and so the geometry is actually collidable.
        truck = self._prepare_truck()
        if truck is None:
            self._log("⚠️ Модель кузова не найдена — рейкаст не сможет работать.")
        # Freeze the fly camera so the view (and hence the picked points'
        # film coords) stays fixed, and so RMB is free to finish picking
        # instead of triggering look.
        self._set_fly_frozen(True)
        if self._do is not None:
            try:
                self._do.accept("mouse1", self._on_click)
                self._do.accept("mouse3", self._on_finish)   # RMB = finish
                self._do.accept("escape", self._on_finish)   # Esc = finish
            except Exception as exc:
                self._log(f"picking bind failed: {exc}")
        self._emit_picking_state(True)
        self._log("Режим выбора точек: кликайте точки на кузове; "
                  "ПКМ или Esc — завершить.")

    def stop_picking(self) -> None:
        if not self._picking:
            return
        self._picking = False
        self._commit_cb = None
        if self._do is not None:
            try:
                self._do.ignore_all()
            except Exception:
                pass
        self._set_fly_frozen(False)
        self._emit_picking_state(False)
        self._log("Режим выбора точек завершён.")

    def _set_fly_frozen(self, frozen: bool) -> None:
        fc = getattr(self.panda_app, "fly_cam", None)
        if fc is not None and hasattr(fc, "set_frozen"):
            try:
                fc.set_frozen(frozen)
            except Exception:
                pass

    def _on_finish(self) -> None:
        """RMB / Esc: finish picking and reconstruct if there are enough
        points; otherwise just cancel and restore the UI."""
        if not self._picking:
            return
        films = list(self._films)
        n = len(films)
        commit_cb = self._commit_cb   # captured before stop_picking clears it
        self.stop_picking()
        if commit_cb is not None:
            # Preset binding: hand the picked film coords back; do NOT rebuild.
            self.clear_points()
            try:
                commit_cb(films)
            except Exception as exc:
                self._log(f"commit опорных точек упал: {exc}")
            return
        if n >= self.MIN_POINTS:
            self._auto_mode = False     # manual pick overrides auto mode
            self.reconstruct()
        else:
            self._log(f"Недостаточно точек ({n}), нужно ≥ {self.MIN_POINTS}.")
            self.clear_points()
            self._finish(False, {})

    def toggle_picking(self) -> None:
        if self._picking:
            self.stop_picking()
        else:
            self.start_picking()

    def clear_points(self) -> None:
        self._films = []
        self._hits = []
        self._emit_count()

    def point_count(self) -> int:
        return len(self._films)

    def dispose_mesh(self) -> None:
        if self._mesh_node is not None:
            try:
                self._mesh_node.removeNode()
            except Exception:
                pass
            loaded = getattr(self.panda_app, "loaded_models", None)
            if loaded and self._mesh_node in loaded:
                try:
                    loaded.remove(self._mesh_node)
                except ValueError:
                    pass
            self._mesh_node = None
        if self._tonar_debug_node is not None:
            try:
                self._tonar_debug_node.removeNode()
            except Exception:
                pass
            self._tonar_debug_node = None

    # ==================================================================
    # Click handling
    # ==================================================================
    def _on_click(self) -> None:
        if not self._picking:
            return
        film = self._mouse_film()
        if film is None:
            return
        hit = self._raycast(film[0], film[1])
        if hit is None:
            self._log("Луч не попал в модель кузова — кликните по кузову.")
            return
        self._films.append(film)
        self._hits.append(hit)
        self._log(f"Точка {len(self._films)}: "
                  f"3D=({hit.x:.2f}, {hit.y:.2f}, {hit.z:.2f})")
        self._emit_count()

    def _mouse_film(self):
        """Current pointer position as film coords in [-1, 1] (top = +1).
        Uses the raw pointer (robust in the embedded window) rather than
        mouseWatcherNode."""
        app = self.panda_app
        win = getattr(app, "win", None)
        if win is None:
            return None
        try:
            ptr = win.getPointer(0)
            if not ptr.getInWindow():
                return None
            w = max(1, win.getXSize())
            h = max(1, win.getYSize())
            fx = (ptr.getX() / w) * 2.0 - 1.0
            fy = 1.0 - (ptr.getY() / h) * 2.0
            return (float(fx), float(fy))
        except Exception as exc:
            self._log(f"pointer read failed: {exc}")
            return None

    # ==================================================================
    # Raycast
    # ==================================================================
    def _find_truck_np(self):
        app = self.panda_app
        target_cuzov = getattr(app, "Target_Cuzov", None)
        loaded = getattr(app, "loaded_models", None) or []
        model_paths = getattr(app, "model_paths", {}) or {}
        if target_cuzov:
            for m in loaded:
                p = model_paths.get(id(m))
                if p and os.path.basename(p) == target_cuzov:
                    return m
        # Fall back to the most-recently loaded model that has a file path.
        for m in reversed(loaded):
            if model_paths.get(id(m)):
                return m
        return None

    def _prepare_truck(self):
        """Resolve the truck NodePath and build an explicit CollisionPolygon
        proxy from its triangles (visible-geometry collision is unreliable
        with RenderPipeline-processed geometry). Returns the NodePath/None."""
        truck = self._find_truck_np()
        self._truck_np = truck
        if truck is None or truck.is_empty():
            return None
        if self._collider_truck_id != id(truck) or self._truck_collider is None:
            try:
                ntri = self._build_truck_collider(truck)
                self._collider_truck_id = id(truck)
                ng = self._count_geoms(truck)
                self._log(f"Кузов для рейкаста: '{truck.get_name()}' "
                          f"(GeomNode-узлов: {ng}, полигонов коллайдера: {ntri}).")
            except Exception as exc:
                self._log(f"build truck collider failed: {exc}")
        return truck

    @staticmethod
    def _count_geoms(np_) -> int:
        try:
            return len(np_.find_all_matches("**/+GeomNode"))
        except Exception:
            return -1

    def _remove_truck_collider(self):
        if self._truck_collider is not None:
            try:
                self._truck_collider.remove_node()
            except Exception:
                pass
        self._truck_collider = None
        self._collider_truck_id = None

    def _build_truck_collider(self, truck) -> int:
        """Build a hidden CollisionNode of CollisionPolygons (in truck-local
        space) from every triangle of the truck's visible geometry."""
        self._remove_truck_collider()
        cnode = CollisionNode("depth_truck_collider")
        cnode.set_into_collide_mask(self.PICK_MASK)
        cnode.set_from_collide_mask(BitMask32.allOff())

        ntri = 0
        for gnp in truck.find_all_matches("**/+GeomNode"):
            gn = gnp.node()
            mat = gnp.get_mat(truck)        # geom space -> truck space
            for gi in range(gn.get_num_geoms()):
                geom = gn.get_geom(gi).decompose()
                vdata = geom.get_vertex_data()
                vreader = GeomVertexReader(vdata, "vertex")
                verts = []
                while not vreader.is_at_end():
                    p = vreader.get_data3()
                    verts.append(mat.xform_point(LPoint3(p[0], p[1], p[2])))
                for pi in range(geom.get_num_primitives()):
                    prim = geom.get_primitive(pi).decompose()
                    for t in range(prim.get_num_primitives()):
                        s = prim.get_primitive_start(t)
                        e = prim.get_primitive_end(t)
                        vi = [prim.get_vertex(i) for i in range(s, e)]
                        if len(vi) != 3:
                            continue
                        a, b, c = verts[vi[0]], verts[vi[1]], verts[vi[2]]
                        try:
                            poly = CollisionPolygon(a, b, c)
                        except Exception:
                            continue
                        cnode.add_solid(poly)
                        ntri += 1

        self._truck_collider = truck.attach_new_node(cnode)
        self._truck_collider.hide()
        return ntri

    def _raycast(self, fx: float, fy: float):
        """Ray-cast the film point against the truck collider; return the
        nearest surface point in world (render) coords, or None."""
        app = self.panda_app
        cam_np = app.cam
        truck = self._truck_np
        col = self._truck_collider
        if (truck is None or truck.is_empty()
                or col is None or col.is_empty()):
            truck = self._prepare_truck()
        if (truck is None or self._truck_collider is None
                or self._truck_collider.is_empty()):
            self._log("Нет коллайдера кузова для рейкаста.")
            return None

        trav = CollisionTraverser("depth_pick")
        queue = CollisionHandlerQueue()
        cnode = CollisionNode("depth_pick_ray")
        cnode.set_from_collide_mask(self.PICK_MASK)
        cnode.set_into_collide_mask(BitMask32.allOff())
        ray = CollisionRay()
        try:
            ray.set_from_lens(cam_np.node(), fx, fy)
        except Exception as exc:
            self._log(f"setFromLens failed: {exc}")
            return None
        cnode.add_solid(ray)
        cnp = cam_np.attach_new_node(cnode)

        hit = None
        try:
            trav.add_collider(cnp, queue)
            trav.traverse(self._truck_collider)
            n = queue.get_num_entries()
            if n > 0:
                queue.sort_entries()
                hit = queue.get_entry(0).get_surface_point(app.render)
            else:
                self._log_ray_miss(fx, fy)
        except Exception as exc:
            self._log(f"raycast failed: {exc}")
        finally:
            cnp.remove_node()
        return hit

    def _log_ray_miss(self, fx: float, fy: float) -> None:
        """Diagnostic: show where the ray runs in world vs the truck bounds,
        so we can tell a ray/lens problem from a geometry problem."""
        try:
            app = self.panda_app
            lens = app.cam.node().get_lens()
            near = Point3()
            far = Point3()
            lens.extrude(Point2(fx, fy), near, far)
            nw = app.render.get_relative_point(app.cam, near)
            fw = app.render.get_relative_point(app.cam, far)
            info = (f"raycast: 0 пересечений (film {fx:.2f},{fy:.2f}); "
                    f"ray near=({nw.x:.1f},{nw.y:.1f},{nw.z:.1f}) "
                    f"far=({fw.x:.1f},{fw.y:.1f},{fw.z:.1f})")
            if self._truck_np is not None and not self._truck_np.is_empty():
                mn, mx = Point3(), Point3()
                if self._truck_np.calc_tight_bounds(mn, mx):
                    info += (f"; кузов bbox=({mn.x:.1f},{mn.y:.1f},{mn.z:.1f})"
                             f"..({mx.x:.1f},{mx.y:.1f},{mx.z:.1f})")
            self._log(info)
        except Exception as exc:
            self._log(f"raycast: 0 пересечений (диагностика упала: {exc})")

    # ==================================================================
    # Reconstruction
    # ==================================================================
    def reconstruct(self) -> bool:
        if trimesh is None:
            self._log("trimesh не установлен — реконструкция невозможна.")
            self._finish(False, {})
            return False
        n_pts = min(len(self._films), len(self._hits))
        if n_pts < self.MIN_POINTS:
            self._log(f"Нужно ≥ {self.MIN_POINTS} точек (есть {n_pts}).")
            self._finish(False, {})
            return False

        depth = self._load_depth_norm()
        if depth is None:
            self._finish(False, {})
            return False
        H, W = depth.shape

        app = self.panda_app
        cam_np = app.cam
        render = app.render
        lens = cam_np.node().get_lens()

        cam_pos = cam_np.getPos(render)
        fwd = render.getRelativeVector(cam_np, Vec3(0, 1, 0))
        fwd.normalize()

        # Vertical film<->photo mapping factor (see _vertical_film_span /
        # the reference overlay: photo fills the film width, fy spans ±fyh).
        fyh = self._vertical_film_span(W, H)

        # --- Control points: (film xy) -> (depth value, metric depth) -----
        cfx = np.asarray([f[0] for f in self._films[:n_pts]], dtype=np.float64)
        cfy = np.asarray([f[1] for f in self._films[:n_pts]], dtype=np.float64)
        d_ctrl = np.empty(n_pts, dtype=np.float64)
        z_ctrl = np.empty(n_pts, dtype=np.float64)
        for i in range(n_pts):
            col, row = self._film_to_pixel(cfx[i], cfy[i], W, H, fyh)
            d_ctrl[i] = float(depth[row, col])
            P = self._hits[i]
            z_ctrl[i] = float((P - cam_pos).dot(fwd))

        # Global linear law Z = A*d + B (least squares over all points).
        A, B = self._fit_linear(d_ctrl, z_ctrl)
        if A is None:
            # Depth had no spread across the points — fall back to a constant
            # base; the per-point residual field then carries the shape.
            A, B = 0.0, float(z_ctrl.mean())
        # Per-point residuals the IDW field will interpolate.
        r_ctrl = z_ctrl - (A * d_ctrl + B)
        self._log(f"Калибровка: {n_pts} точек, Z = {A:.3f}*d + {B:.3f}; "
                  f"остатки {r_ctrl.min():+.2f}..{r_ctrl.max():+.2f} м "
                  f"(IDW-коррекция по районам).")

        # --- Region: defined by the mask, NOT by the picked points --------
        # The picked points are used ONLY for the depth calibration above.
        #   mask present -> the mesh covers exactly the masked pixels
        #                   (alpha > MASK_ALPHA_MIN), at full grid resolution
        #                   over the mask's bounding box;
        #   no mask      -> the whole frame is reconstructed and nothing is
        #                   culled.
        mask_alpha = self._load_mask_alpha(W, H) if self.USE_MASKS else None
        use_mask = mask_alpha is not None

        G = self.GRID
        stride = G + 1
        # Grid bounds in film space: the mask's film bounding box if we have a
        # mask (keeps full resolution where it matters), otherwise the whole
        # depth image ([-1, 1] x [-fyh, fyh]).
        if use_mask:
            fx0, fx1, fy0, fy1 = self._mask_film_bounds(mask_alpha, W, H, fyh)
        else:
            fx0, fx1, fy0, fy1 = -1.0, 1.0, -fyh, fyh
        xs = fx0 + (fx1 - fx0) * (np.arange(stride) / G)
        ys = fy0 + (fy1 - fy0) * (np.arange(stride) / G)
        FX, FY = np.meshgrid(xs, ys)          # (stride, stride)
        FX = FX.ravel()
        FY = FY.ravel()

        # Depth value + locally-corrected metric depth at every grid sample.
        cols, rows = self._film_to_pixel_vec(FX, FY, W, H, fyh)
        d_grid = depth[rows, cols]
        resid = self._idw_residual(FX, FY, cfx, cfy, r_ctrl)
        Z_grid = A * d_grid + B + resid

        # Region selection: keep masked nodes, or everything when no mask.
        if use_mask:
            inside = mask_alpha[rows, cols] > self.MASK_ALPHA_MIN
            n_mask = int(inside.sum())
            # Grow the kept region by a cell so the all-4-corners face rule
            # reaches the true mask edge instead of eroding it inward.
            if self.MASK_DILATE_CELLS > 0:
                inside = self._dilate_grid_mask(
                    inside.reshape(stride, stride),
                    self.MASK_DILATE_CELLS).ravel()
            self._log(f"Маска: регион = {n_mask} узлов "
                      f"(alpha > {self.MASK_ALPHA_MIN:.0%}, +"
                      f"{self.MASK_DILATE_CELLS} ячейка к краю); "
                      f"геометрические отсечения отключены.")
        else:
            inside = np.ones(FX.shape[0], dtype=bool)
            # Flood-fill background removal: a large gray-ish region flooded
            # from the image border (typically the sky/wall above the truck)
            # is dropped before reconstruction. Grid nodes that sample a
            # background pixel are marked out-of-region.
            bg = self._detect_background(depth)
            if bg is not None:
                bg_node = bg[rows, cols]
                n_bg = int(bg_node.sum())
                inside = inside & (~bg_node)
                self._log(
                    f"Фон (заливка): отброшено {int(bg.sum())} пикс. "
                    f"({bg.sum() / (H * W):.0%} кадра) → {n_bg} узлов сетки; "
                    f"реконструирую остаток кадра.")
            else:
                self._log("Маски нет: реконструирую весь кадр без отсечений.")

        # Gentle smoothing of the (kept) grid depths to kill the 8-bit
        # quantization terracing before the surface is built.
        if self.SMOOTH_ITERS > 0:
            Z_grid = self._smooth_grid_z(
                Z_grid.reshape(stride, stride),
                inside.reshape(stride, stride),
                self.SMOOTH_ITERS).ravel()

        # Unproject the in-region samples through the live lens.
        verts = np.zeros((FX.shape[0], 3), dtype=np.float64)
        for k in range(FX.shape[0]):
            if not inside[k]:
                continue
            wp = self._unproject(lens, cam_np, render,
                                 float(FX[k]), float(FY[k]), float(Z_grid[k]))
            verts[k, 0] = wp.x
            verts[k, 1] = wp.y
            verts[k, 2] = wp.z

        clip_volume = None

        # ---- Stage 1: relief over the whole frame, cutting off over-long
        #      polygons (relative to camera distance) and the cells around them.
        if self.LONGPOLY_CUTOFF:
            mr = (float("inf") if self.LONGPOLY_MAX_EDGE_RATIO is None
                  else float(self.LONGPOLY_MAX_EDGE_RATIO))
            me = (float("inf") if self.LONGPOLY_MAX_EDGE_M is None
                  else float(self.LONGPOLY_MAX_EDGE_M))
            long_prop = int(self.LONGPOLY_PROPAGATE_CELLS)
        else:
            mr, me, long_prop = float("inf"), float("inf"), 0
        faces, dropped = self._build_grid_faces(
            G, inside, verts, Z_grid, mr, long_prop, me)
        self._log(
            f"Срез длинных полигонов: edge/Z≤{mr:.3g} И edge≤{me:.3g} м "
            f"(рад. {long_prop}) — отброшено {dropped} треугольников.")

        if len(faces) == 0:
            self._log("Нет ячеек для меша после среза длинных полигонов.")
            self._finish(False, {})
            return False

        # ---- Stage 2: Boolean with the napolnitel mesh — keep only the relief
        #      INSIDE the truck body, cut off everything outside. Runs BEFORE
        #      extrapolation so the measured cloud (and the target rectangle
        #      centred on it) is bounded to the truck, not the whole frame.
        if self.CLIP_TO_NAPOLNITEL and len(faces) > 0:
            res = self._clip_relief_to_body(verts, faces)
            if res is not None:
                verts, faces = res

        if len(faces) == 0:
            self._log("Нет ячеек для меша (всё отсечено наполнителем).")
            self._finish(False, {})
            return False

        # ---- Stage 3 (optional): extrapolate the truck-bounded relief to
        #      TARGET_SIZE_M, then (optional) seal + Boolean DIFFERENCE with the
        #      napolnitel for a closed, measurable filler volume.
        if self.ENABLE_EXTRAPOLATION and self.TARGET_SIZE_M is not None:
            # Measured cloud = the vertices that survived Stages 1 & 2.
            used = np.unique(np.asarray(faces).reshape(-1))
            ext = self._build_target_extrapolated(verts[used])
            if ext is not None:
                verts, faces, ext_info = ext
                TX_m, TY_m = self.TARGET_SIZE_M
                sw_m, sh_m = ext_info["src_size_m"]
                tnx, tny = ext_info["target_grid"]
                target_dims = (tnx, tny)
                self._log(
                    f"Экстраполяция: измеренная зона {sw_m:.2f}×{sh_m:.2f} м → "
                    f"цель {TX_m:.1f}×{TY_m:.1f} м (mirror-tiling, сетка "
                    f"{tnx}×{tny}, {ext_info['src_valid_cells']} валидных "
                    f"ячеек источника).")
                volume_done = False
                if (self.ENABLE_VOLUME and self.TONAR_OBJ_REL_PATH):
                    # Seal the relief into the empty space above it, then run
                    # the Boolean DIFFERENCE tonar − sealed → the closed,
                    # measurable filler volume below the relief.
                    clip_res = self._clip_relief_to_target(
                        verts, faces, target_dims)
                    if clip_res is not None:
                        verts, faces, clip_volume = clip_res
                        volume_done = True
                if not volume_done:
                    # No volume mesh (volume off, or every boolean engine
                    # failed): the extrapolated TARGET_SIZE_M rectangle still
                    # overhangs the bed, so clip it to the truck body with the
                    # point-in-mesh test. Guarantees the geometry is bounded by
                    # the napolnitel even without a working boolean engine.
                    res = self._clip_relief_to_body(verts, faces)
                    if res is not None:
                        verts, faces = res
                        self._log("Объёмный boolean не выполнен — "
                                  "экстраполированный рельеф обрезан по кузову "
                                  "(point-in-mesh).")
            else:
                self._log("Экстраполяция: недостаточно данных — пропуск.")

        # ---- Final cleanup: strip near-vertical walls (with a metric removal
        #      radius) and tiny disconnected clusters from the displayed mesh.
        #      Runs last so it catches vertical faces from any stage; the volume
        #      number was already computed above and is kept.
        if (self.STEEP_CUTOFF or self.MIN_CLUSTER_SIZE_M) and len(faces) > 0:
            verts, faces = self._cleanup_relief(verts, faces)

        if len(faces) == 0:
            self._log("Нет ячеек для меша (регион вырожден / всё отсечено).")
            self._finish(False, {})
            return False

        # Z extent of the vertices actually used by the surviving faces.
        used = np.unique(np.asarray(faces).reshape(-1))
        z_min = float(verts[used, 2].min())
        z_max = float(verts[used, 2].max())

        # Vertex normals for proper lighting (computed from the grid mesh).
        normals = self._compute_normals(verts, faces)

        # Drop the previous reconstruction before adding the new one.
        self.dispose_mesh()
        try:
            node = self._build_panda_mesh(verts, faces, normals)
        except Exception as exc:
            self._log(f"Сборка меша упала: {exc}")
            self._finish(False, {})
            return False
        if node is None or node.is_empty():
            self._log("Пустой меш реконструкции.")
            self._finish(False, {})
            return False

        # Texture it with the texture set currently selected in the UI.
        self._apply_selected_texture(node)
        self._mesh_node = node
        self.panda_app.depth_recon_mesh = node

        # Remember the film points so other stand snapshots can be rebuilt
        # automatically (same camera, different depth map).
        self._saved_films = [(float(f[0]), float(f[1]))
                             for f in self._films[:n_pts]]
        # Visualization: in AUTO mode the green/blue/red diagnostic grid was
        # already built (before this call) — don't clobber it. In MANUAL mode
        # show the used anchors (green) at their 3D positions (4-tuple format
        # the viz builder expects; the old 2-tuple form crashed it).
        if self._viz_on and not self._auto_mode:
            manual_viz = [(float(p.x), float(p.y), float(p.z), 0)
                          for p in self._hits[:n_pts]]
            self._auto_viz_data = manual_viz
            self._build_point_viz(manual_viz)

        info = {"A": A, "B": B, "z_min": z_min, "z_max": z_max,
                "points": int(n_pts),
                "verts": int(verts.shape[0]), "faces": int(len(faces))}
        if clip_volume is not None:
            info["volume_m3"] = float(clip_volume)
        self._log(f"✅ Реконструкция готова по {n_pts} точкам: "
                  f"{info['faces']} треугольников.")
        self._finish(True, info)
        return True

    # ------------------------------------------------------------------
    def has_saved_points(self) -> bool:
        return len(self._saved_films) >= self.MIN_POINTS

    def has_manual_saved_points(self) -> bool:
        """True only for points saved from a MANUAL pick. These are reused
        automatically on snapshot switch; auto-found points are not."""
        return self.has_saved_points() and not self._auto_mode

    def set_saved_films(self, films) -> None:
        """Load externally-stored anchor film coords (e.g. bound to a camera
        preset) as the saved points. Marked as a manual pick so they rebuild
        automatically on snapshot selection (reconstruct_saved)."""
        cleaned: list[tuple[float, float]] = []
        for f in films or []:
            try:
                cleaned.append((float(f[0]), float(f[1])))
            except (TypeError, ValueError, IndexError):
                continue
        self._saved_films = cleaned
        self._auto_mode = False

    def clear_saved_points(self) -> None:
        self._saved_films = []
        self._auto_mode = False
        self._auto_viz_data = None
        self._dispose_viz()

    def reconstruct_saved(self, depth_path: str = "") -> bool:
        """Rebuild from the saved film points against a (new) depth map by
        re-raycasting them at the current camera/truck. Used to auto-build
        another stand snapshot without re-picking."""
        if not self.has_saved_points():
            return False
        if self._picking:
            return False
        if depth_path:
            self._depth_path = depth_path
        if not self._depth_path or not os.path.exists(self._depth_path):
            self._log("Авто-реконструкция: нет карты глубины.")
            return False
        truck = self._prepare_truck()
        if truck is None:
            self._log("Авто-реконструкция: нет модели кузова.")
            return False

        films, hits = [], []
        for (fx, fy) in self._saved_films:
            hit = self._raycast(fx, fy)
            if hit is not None:
                films.append((fx, fy))
                hits.append(hit)
        if len(films) < self.MIN_POINTS:
            self._log(f"Авто-реконструкция: рейкаст дал {len(films)} точек "
                      f"(<{self.MIN_POINTS}) — пропуск.")
            return False
        self._films = films
        self._hits = hits
        self._emit_count()
        self._log(f"Авто-реконструкция по {len(films)} сохранённым точкам…")
        return self.reconstruct()

    # ==================================================================
    # Automatic reference-point search
    # ==================================================================
    def set_visualize(self, on: bool) -> None:
        """Toggle the point-grid debug visualisation."""
        self._viz_on = bool(on)
        if self._viz_on:
            if self._auto_viz_data:
                self._build_point_viz(self._auto_viz_data)
        else:
            self._dispose_viz()

    def reconstruct_auto(self, depth_path: str = "") -> bool:
        """Find reference points automatically (grid of rays + robust
        rejection) and reconstruct. The camera must already be aligned."""
        if self._picking:
            return False
        if depth_path:
            self._depth_path = depth_path
        if not self._depth_path or not os.path.exists(self._depth_path):
            self._log("Авто-точки: нет карты глубины.")
            self._finish(False, {})
            return False
        depth = self._load_depth_norm()
        if depth is None:
            self._finish(False, {})
            return False
        H, W = depth.shape

        fyh = self._vertical_film_span(W, H)

        truck = self._find_truck_np()
        self._truck_np = truck
        if truck is None or truck.is_empty():
            self._log("Авто-точки: модель кузова не найдена.")
            self._finish(False, {})
            return False

        self._log(f"Авто-поиск опорных точек: сетка "
                  f"{self.AUTO_GRID}×{self.AUTO_GRID} (z-буфер кузова)…")
        films, hits, viz, n_hit = self._auto_find_points(depth, fyh, W, H, truck)
        self._log(f"Авто-точки: в кузов попало {n_hit}, принято "
                  f"{len(films)} (отброшено {n_hit - len(films)}).")

        self._auto_viz_data = viz
        if self._viz_on:
            self._build_point_viz(viz)

        if len(films) < self.MIN_POINTS:
            self._log("Авто-точки: принято слишком мало точек.")
            self._finish(False, {})
            return False

        self._films = films
        self._hits = hits
        self._auto_mode = True
        self._emit_count()
        return self.reconstruct()

    def _get_truck_vertices_world(self, truck):
        """All truck vertices in world (render) coords, cached per truck."""
        if self._truck_verts is not None and self._truck_verts_id == id(truck):
            return self._truck_verts
        render = self.panda_app.render
        verts = []
        for gnp in truck.find_all_matches("**/+GeomNode"):
            gn = gnp.node()
            mat = gnp.get_mat(render)        # geom space -> world
            for gi in range(gn.get_num_geoms()):
                vdata = gn.get_geom(gi).get_vertex_data()
                vr = GeomVertexReader(vdata, "vertex")
                while not vr.is_at_end():
                    p = vr.get_data3()
                    w = mat.xform_point(LPoint3(p[0], p[1], p[2]))
                    verts.append((w[0], w[1], w[2]))
        arr = (np.asarray(verts, dtype=np.float64)
               if verts else np.zeros((0, 3), dtype=np.float64))
        self._truck_verts = arr
        self._truck_verts_id = id(truck)
        return arr

    @staticmethod
    def _mat_to_np(m):
        return np.array([[m.get_cell(i, j) for j in range(4)]
                         for i in range(4)], dtype=np.float64)

    def _auto_depth_grid(self, truck):
        """Forward-depth of the truck per grid cell, via a supersampled
        projected z-buffer (nearest vertex wins). Returns an (N,N) array with
        +inf where no truck surface projects. O(vertices), no ray-casting."""
        V = self._get_truck_vertices_world(truck)
        if V.shape[0] == 0:
            return None
        app = self.panda_app
        cam_np = app.cam
        render = app.render
        lens = cam_np.node().get_lens()

        M = self._mat_to_np(render.get_mat(cam_np))      # world -> cam (rows)
        Pj = self._mat_to_np(lens.get_projection_mat())  # cam  -> clip
        Vh = np.hstack([V, np.ones((V.shape[0], 1))])
        cam = Vh @ M                       # (V,4)
        Zf = cam[:, 1]                     # forward distance (cam +Y)
        clip = cam @ Pj                    # (V,4)
        w = clip[:, 3]
        good = (Zf > 1e-4) & (np.abs(w) > 1e-12)
        fx = np.where(good, clip[:, 0] / np.where(good, w, 1.0), 9.0)
        fy = np.where(good, clip[:, 1] / np.where(good, w, 1.0), 9.0)
        inb = good & (np.abs(fx) <= 1.0) & (np.abs(fy) <= 1.0)

        N = int(self.AUTO_GRID)
        K = max(1, int(self.AUTO_BUF_SUPERSAMPLE))
        BUF = N * K
        px = np.clip(((fx + 1.0) * 0.5 * BUF).astype(np.int64), 0, BUF - 1)
        py = np.clip(((1.0 - fy) * 0.5 * BUF).astype(np.int64), 0, BUF - 1)
        buf = np.full(BUF * BUF, np.inf, dtype=np.float64)
        sel = inb
        np.minimum.at(buf, py[sel] * BUF + px[sel], Zf[sel])
        buf = buf.reshape(BUF, BUF)
        # Min-pool each KxK block down to the grid (nearest surface per cell).
        Zg = buf.reshape(N, K, N, K).min(axis=(1, 3))
        return Zg

    def _auto_find_points(self, depth, fyh, W, H, truck):
        """Find anchor candidates via the truck z-buffer, reject outliers, and
        return (films_accepted, hits_accepted, viz_films_accepted, n_hits)."""
        Zg = self._auto_depth_grid(truck)
        if Zg is None:
            return [], [], [], 0
        N = int(self.AUTO_GRID)

        # Cell-centre film coords for the grid.
        jj, ii = np.meshgrid(np.arange(N), np.arange(N))
        fxg = ((jj + 0.5) / N) * 2.0 - 1.0
        fyg = 1.0 - ((ii + 0.5) / N) * 2.0
        hit_mask = np.isfinite(Zg)

        # Depth-map value per cell.
        cols, rows = self._film_to_pixel_vec(fxg.ravel(), fyg.ravel(),
                                             W, H, fyh)
        dg = depth[rows, cols].reshape(N, N)
        Zsafe = np.where(hit_mask, Zg, 0.0)

        # Unproject every truck-hit cell to a 3D world grid (reused below for
        # the surface-normal floor test and for the accepted anchors).
        app = self.panda_app
        cam_np = app.cam
        render = app.render
        lens = cam_np.node().get_lens()
        Pw = np.full((N, N, 3), np.nan, dtype=np.float64)
        for i in range(N):
            for j in range(N):
                if not hit_mask[i, j]:
                    continue
                wp = self._unproject(lens, cam_np, render,
                                     float(fxg[i, j]), float(fyg[i, j]),
                                     float(Zg[i, j]))
                Pw[i, j, 0] = wp.x; Pw[i, j, 1] = wp.y; Pw[i, j, 2] = wp.z

        # Candidate mask: build an AABB from all truck-hit points and drop
        # the lower half (world +Z), keeping only the upper part of the bed.
        cand = hit_mask
        if hit_mask.any():
            zw = Pw[:, :, 2]
            zs = zw[hit_mask]
            z_min = float(zs.min())
            z_max = float(zs.max())
            z_mid = 0.5 * (z_min + z_max)
            cand = hit_mask & (zw >= z_mid)
            n_drop = int(hit_mask.sum() - cand.sum())
            self._log(f"Авто-точки: AABB по Z=[{z_min:.2f}, {z_max:.2f}], "
                      f"отброшена нижняя половина — {n_drop} точек.")

        accepted = self._reject_outliers(fxg, fyg, dg, Zsafe, cand)

        diag = bool(self.AUTO_VIZ_DIAGNOSTIC)
        films_acc, hits_acc, viz = [], [], []
        n_floor_f = n_robust = 0
        for i in range(N):
            for j in range(N):
                if not hit_mask[i, j]:
                    continue
                fx = float(fxg[i, j]); fy = float(fyg[i, j])
                px = float(Pw[i, j, 0])
                py = float(Pw[i, j, 1])
                pz = float(Pw[i, j, 2])
                if accepted[i, j]:
                    films_acc.append((fx, fy))
                    hits_acc.append(Point3(px, py, pz))
                    viz.append((px, py, pz, 0))        # green = anchor
                elif diag and not cand[i, j]:
                    viz.append((px, py, pz, 1))        # blue = floor-filtered
                    n_floor_f += 1
                elif diag:
                    viz.append((px, py, pz, 2))        # red = robust-rejected
                    n_robust += 1
        if diag:
            self._log(f"Авто-точки (диагностика): принято {len(films_acc)}, "
                      f"фильтр дна {n_floor_f}, робастно отброшено {n_robust}.")
        return films_acc, hits_acc, viz, int(hit_mask.sum())

    @staticmethod
    def _surface_verticality(Pw):
        """|normal·Zup| per grid cell from the unprojected truck surface.
        ~1 = horizontal (floor), ~0 = vertical (wall). NaN-safe; cells without
        full neighbours get 0 (kept)."""
        N = Pw.shape[0]
        du = np.full((N, N, 3), np.nan)
        dv = np.full((N, N, 3), np.nan)
        du[:, 1:-1, :] = Pw[:, 2:, :] - Pw[:, :-2, :]    # screen +x direction
        dv[1:-1, :, :] = Pw[:-2, :, :] - Pw[2:, :, :]    # screen +y (up) dir
        nrm = np.cross(du, dv)
        ln = np.linalg.norm(nrm, axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            nz = np.abs(nrm[:, :, 2]) / ln
        return np.where(np.isfinite(nz), nz, 0.0)

    def _reject_outliers(self, fxg, fyg, dg, Zg, hit_mask):
        """Robust rejection of bad anchor candidates. Returns a bool grid of
        accepted points. See the AUTO_* constants for the thresholds."""
        N = fxg.shape[0]
        inl = hit_mask.copy()
        A = B = None

        # Stage A: global robust linear d -> Z fit (drops fill / mismatch).
        for _ in range(int(self.AUTO_REJECT_ITERS)):
            if int(inl.sum()) < self.MIN_POINTS:
                break
            Af, Bf = self._fit_linear(dg[inl], Zg[inl])
            if Af is None:
                Af, Bf = 0.0, float(Zg[inl].mean())
            A, B = Af, Bf
            r = Zg - (A * dg + B)
            rin = r[inl]
            center = float(np.median(rin))
            mad = 1.4826 * float(np.median(np.abs(rin - center)))
            thr = max(self.AUTO_GLOBAL_K * mad, self.AUTO_GLOBAL_ABS)
            new = inl & (np.abs(r - center) <= thr)
            if int(new.sum()) == int(inl.sum()):
                inl = new
                break
            inl = new

        if A is None:
            return inl

        # Stage B: local residual smoothness (the "5 agree, 6th is off" case).
        r = Zg - (A * dg + B)
        # Scale the neighbourhood with the grid so it covers a constant screen
        # area regardless of AUTO_GRID — keeps the statistics robust as the
        # point count grows.
        Wr = max(int(self.AUTO_LOCAL_WINDOW),
                 int(round(self.AUTO_LOCAL_WINDOW * N / 50.0)))
        rej = np.zeros((N, N), dtype=bool)
        for (gi, gj) in np.argwhere(inl):
            i0, i1 = max(0, gi - Wr), min(N, gi + Wr + 1)
            j0, j1 = max(0, gj - Wr), min(N, gj + Wr + 1)
            win_in = inl[i0:i1, j0:j1].copy()
            win_in[gi - i0, gj - j0] = False     # exclude self
            if int(win_in.sum()) < self.AUTO_LOCAL_MIN_NB:
                continue
            pred = float(np.median(r[i0:i1, j0:j1][win_in]))
            if abs(float(r[gi, gj]) - pred) > self.AUTO_LOCAL_THRESH:
                rej[gi, gj] = True
        return inl & (~rej)

    # fate code -> RGBA: 0 anchor (green), 1 floor-filtered (blue),
    # 2 robust-rejected (red).
    _VIZ_COLORS = {
        0: (0.1, 1.0, 0.1, 1.0),
        1: (0.15, 0.6, 1.0, 1.0),
        2: (1.0, 0.1, 0.1, 1.0),
    }

    def _build_point_viz(self, viz) -> None:
        """Draw the auto-search points in the 3D WORLD at their hit positions,
        one node per fate with a FLAT colour (`set_color`). Drawing in `render`
        is guaranteed to display under RenderPipeline (unlike render2d). `viz`
        items are (px, py, pz, fate)."""
        self._dispose_viz()
        if not viz:
            return
        try:
            from panda3d.core import GeomPoints
            groups = {}
            for item in viz:
                if len(item) < 3:
                    continue           # expects (px, py, pz[, fate])
                fate = item[3] if len(item) > 3 else 0
                groups.setdefault(fate, []).append(
                    (item[0], item[1], item[2]))
            if not groups:
                return

            root = self.panda_app.render.attach_new_node("auto_point_viz")
            for fate, pts in groups.items():
                fmt = GeomVertexFormat.get_v3()
                vdata = GeomVertexData("auto_pts", fmt, Geom.UHStatic)
                vdata.set_num_rows(len(pts))
                vw = GeomVertexWriter(vdata, "vertex")
                prim = GeomPoints(Geom.UHStatic)
                for idx, (px, py, pz) in enumerate(pts):
                    vw.add_data3f(float(px), float(py), float(pz))
                    prim.add_vertex(idx)
                prim.close_primitive()
                geom = Geom(vdata)
                geom.add_primitive(prim)
                gnode = GeomNode(f"auto_pts_{fate}")
                gnode.add_geom(geom)
                node = root.attach_new_node(gnode)
                r, g, b, a = self._VIZ_COLORS.get(fate, self._VIZ_COLORS[0])
                node.set_color(r, g, b, a, 1)
                node.set_render_mode_thickness(12)

            root.set_light_off(1)
            root.set_depth_test(False)     # draw on top of the truck
            root.set_depth_write(False)
            root.set_bin("fixed", 100)
            self._viz_node = root
        except Exception as exc:
            self._log(f"point viz failed: {exc}")

    def _dispose_viz(self) -> None:
        n = getattr(self, "_viz_node", None)
        if n is not None:
            try:
                n.remove_node()
            except Exception:
                pass
        self._viz_node = None

    # ------------------------------------------------------------------
    def _compute_normals(self, verts, faces):
        """Per-vertex normals from the grid mesh, oriented so they face
        'up' (+Z) — the fill surface is seen from above, so up-facing
        normals light correctly."""
        try:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
            if normals.shape[0] != verts.shape[0]:
                raise ValueError("normal count mismatch")
        except Exception as exc:
            self._log(f"normals fallback (+Z): {exc}")
            normals = np.tile([0.0, 0.0, 1.0], (verts.shape[0], 1))
        # Flip the whole field if it mostly points down.
        if float(np.nanmean(normals[:, 2])) < 0.0:
            normals = -normals
        return normals

    def _build_panda_mesh(self, verts, faces, normals):
        """Build a Panda NodePath with V3N3T2 geometry. UVs are a top-down
        planar projection of global XY scaled by UV_PER_METER."""
        app = self.panda_app
        scale = float(self.UV_PER_METER)
        n = int(verts.shape[0])

        fmt = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("depth_recon", fmt, Geom.UHStatic)
        vdata.set_num_rows(n)
        vw = GeomVertexWriter(vdata, "vertex")
        nw = GeomVertexWriter(vdata, "normal")
        tw = GeomVertexWriter(vdata, "texcoord")
        for i in range(n):
            x, y, z = float(verts[i, 0]), float(verts[i, 1]), float(verts[i, 2])
            vw.add_data3f(x, y, z)
            nx, ny, nz = (float(normals[i, 0]), float(normals[i, 1]),
                          float(normals[i, 2]))
            nw.add_data3f(nx, ny, nz)
            # Top-down planar UV from global XY (one tile per metre at scale=1).
            tw.add_data2f(x * scale, y * scale)

        tris = GeomTriangles(Geom.UHStatic)
        for f in faces:
            tris.add_vertices(int(f[0]), int(f[1]), int(f[2]))
        tris.close_primitive()

        geom = Geom(vdata)
        geom.add_primitive(tris)
        gnode = GeomNode("depth_recon_mesh")
        gnode.add_geom(geom)
        node = app.render.attach_new_node(gnode)
        # Keep it out of the model-distribution / cleanup lists' way but
        # track it for disposal via depth_recon_mesh.
        return node

    def _build_tonar_debug_mesh(self, verts, faces):
        """Attach a semi-transparent visualization of the tonar_napolnitel
        mesh (with TONAR_OBJ_TRANSFORM already applied) to the scene, so
        the user can see exactly which volume the Blender boolean is
        operating on. Returns the attached NodePath."""
        app = self.panda_app
        n = int(verts.shape[0])
        fmt = GeomVertexFormat.getV3()
        vdata = GeomVertexData("depth_recon_tonar_debug", fmt, Geom.UHStatic)
        vdata.set_num_rows(n)
        vw = GeomVertexWriter(vdata, "vertex")
        for i in range(n):
            vw.add_data3f(float(verts[i, 0]),
                          float(verts[i, 1]),
                          float(verts[i, 2]))
        tris = GeomTriangles(Geom.UHStatic)
        for f in faces:
            tris.add_vertices(int(f[0]), int(f[1]), int(f[2]))
        tris.close_primitive()
        geom = Geom(vdata)
        geom.add_primitive(tris)
        gnode = GeomNode("depth_recon_tonar_debug")
        gnode.add_geom(geom)
        node = app.render.attach_new_node(gnode)
        # Translucent cyan overlay, both faces visible. Disable depth
        # write so the overlay doesn't block geometry behind it from
        # passing the depth test (otherwise the relief / truck would
        # disappear behind the see-through tonar).
        node.set_transparency(TransparencyAttrib.M_alpha)
        node.set_color(0.35, 0.75, 1.0, 0.25)
        node.set_two_sided(True)
        try:
            node.set_depth_write(False)
        except Exception:
            pass
        # Don't get picked up by the truck raycast.
        try:
            node.set_collide_mask(BitMask32.allOff())
        except Exception:
            pass
        return node

    def _apply_selected_texture(self, node) -> None:
        """Apply the UI-selected texture set (panda_app.current_texture_set)
        to the reconstructed mesh, mirroring the truck/perlin material path.
        Falls back to a plain material if texturing isn't available."""
        app = self.panda_app
        applier = getattr(app, "_apply_textures_and_material", None)
        if callable(applier):
            try:
                applier(node)
                return
            except Exception as exc:
                self._log(f"apply selected texture failed: {exc}")
        # Fallback: minimal RP-friendly material.
        self._apply_material(node)

    # ------------------------------------------------------------------
    def _load_depth_norm(self):
        """Load the depth PNG as an HxW float array normalised to [0, 1]."""
        try:
            from PIL import Image
            im = Image.open(self._depth_path).convert("L")
            arr = np.asarray(im, dtype=np.float32) / 255.0
            return arr
        except Exception as exc:
            self._log(f"Не удалось прочитать карту глубины: {exc}")
            return None

    def _detect_background(self, depth):
        """Detect a large gray-ish background region in the depth map via a
        paint-bucket flood fill seeded from the image border(s). Returns an
        HxW bool array (True = background, to exclude from reconstruction), or
        None when the flood finds nothing big enough / the feature is off.

        The flood grows across pixels whose 8-bit depth value stays within
        BG_FLOOD_THRESHOLD of the SEED (border) value (cv2 FIXED range), so it
        spreads over the gray/dark background but stops at the much brighter
        (nearer) truck instead of bleeding through its soft silhouette. Only
        flooded blobs covering >= BG_MIN_AREA_FRAC of the frame are kept, so a
        seed that lands on real geometry (small fill) is ignored."""
        if not self.BG_FLOODFILL:
            return None
        try:
            import cv2
        except Exception as exc:
            self._log(f"Фон: OpenCV недоступен ({exc}) — фон не удаляю.")
            return None

        H, W = depth.shape
        img = np.clip(depth * 255.0, 0.0, 255.0).astype(np.uint8)
        thr = int(self.BG_FLOOD_THRESHOLD)
        min_area = int(self.BG_MIN_AREA_FRAC * H * W)
        # 8-connected, fixed range (compare to seed, not neighbour),
        # mask-only (image left untouched), fill mask with 1.
        flags = (8 | cv2.FLOODFILL_FIXED_RANGE
                 | cv2.FLOODFILL_MASK_ONLY | (1 << 8))

        # Seed pixels along each requested border edge.
        seeds = []
        borders = self.BG_SEED_BORDERS or ()
        if "top" in borders:
            seeds += [(c, 0) for c in range(W)]
        if "bottom" in borders:
            seeds += [(c, H - 1) for c in range(W)]
        if "left" in borders:
            seeds += [(0, r) for r in range(H)]
        if "right" in borders:
            seeds += [(W - 1, r) for r in range(H)]

        bg = np.zeros((H, W), dtype=bool)
        # Track every pixel any flood has already covered, so each disjoint
        # border segment is flooded once (and we don't reflood the same blob).
        seen = np.zeros((H, W), dtype=bool)
        for (sx, sy) in seeds:
            if seen[sy, sx]:
                continue
            m = np.zeros((H + 2, W + 2), dtype=np.uint8)
            try:
                cv2.floodFill(img, m, (int(sx), int(sy)), 0,
                              (thr,) * 3, (thr,) * 3, flags)
            except Exception as exc:
                self._log(f"Фон: floodFill упал: {exc}")
                return None
            region = m[1:-1, 1:-1] > 0
            seen |= region
            if int(region.sum()) >= min_area:
                bg |= region

        if not bg.any():
            return None
        return bg

    def _mask_path(self) -> str:
        if not self._depth_path:
            return ""
        return os.path.splitext(self._depth_path)[0] + self.MASK_SUFFIX

    def _load_mask_alpha(self, W, H):
        """Load the fill mask's alpha channel as an HxW float array in [0, 1]
        (resized to the depth size if needed). Returns None when there is no
        mask file for this snapshot."""
        mp = self._mask_path()
        if not mp or not os.path.exists(mp):
            self._log("Маска не найдена — применяю геометрические отсечения.")
            return None
        try:
            from PIL import Image
            im = Image.open(mp).convert("RGBA")
            if im.size != (W, H):
                im = im.resize((W, H), Image.NEAREST)
            return np.asarray(im, dtype=np.float32)[..., 3] / 255.0
        except Exception as exc:
            self._log(f"Не удалось прочитать маску: {exc}")
            return None

    def _mask_film_bounds(self, mask_alpha, W, H, fyh):
        """Film-space bounding box (fx0, fx1, fy0, fy1) of the masked pixels
        (alpha > MASK_ALPHA_MIN), with a 1-pixel margin so the edge is fully
        covered. Falls back to the whole frame if the mask is empty."""
        sel = mask_alpha > self.MASK_ALPHA_MIN
        ys_idx, xs_idx = np.where(sel)
        if xs_idx.size == 0:
            return -1.0, 1.0, -fyh, fyh
        c0 = max(0, int(xs_idx.min()) - 1)
        c1 = min(W - 1, int(xs_idx.max()) + 1)
        r0 = max(0, int(ys_idx.min()) - 1)
        r1 = min(H - 1, int(ys_idx.max()) + 1)
        # Pixel -> film, inverse of _film_to_pixel:
        #   u = col/(W-1), fx = 2u - 1;  v = row/(H-1), fy = fyh*(1 - 2v).
        wd = float(max(1, W - 1))
        hd = float(max(1, H - 1))
        fx0 = 2.0 * (c0 / wd) - 1.0
        fx1 = 2.0 * (c1 / wd) - 1.0
        # Rows grow downward, so the top row (r0) is the HIGH fy.
        fy_hi = fyh * (1.0 - 2.0 * (r0 / hd))
        fy_lo = fyh * (1.0 - 2.0 * (r1 / hd))
        return fx0, fx1, fy_lo, fy_hi

    def _vertical_film_span(self, W_depth, H_depth) -> float:
        """Half-extent (in film fy units) that the photo occupies vertically:
        fyh = window_aspect / photo_aspect. The photo fills the film width,
        so vertically it spans fy in [-fyh, +fyh].

        The aspect ratio is taken from the COLOR overlay (what the user sees),
        falling back to the depth image if no color image is available. This
        matters when color and depth have different aspect ratios."""
        try:
            win = self.panda_app.win
            aw = float(win.get_x_size()) / float(max(1, win.get_y_size()))
        except Exception:
            aw = float(W_depth) / float(max(1, H_depth))
        # Use the color image's aspect to match the overlay the user sees.
        W_overlay, H_overlay = self._color_image_size(W_depth, H_depth)
        ap = float(W_overlay) / float(max(1, H_overlay))
        if ap <= 1e-9:
            return 1.0
        return aw / ap

    def _color_image_size(self, W_fallback, H_fallback):
        """Return (W, H) of the color overlay image, or the fallback size."""
        if self._color_path and os.path.exists(self._color_path):
            try:
                from PIL import Image
                with Image.open(self._color_path) as im:
                    return im.width, im.height
            except Exception:
                pass
        return W_fallback, H_fallback

    @staticmethod
    def _film_to_pixel(fx, fy, W, H, fyh=1.0):
        # Horizontal: photo fills the film width -> u directly from fx.
        u = (fx + 1.0) * 0.5
        # Vertical: photo spans fy in [-fyh, fyh] (window/photo aspect),
        # so v = (1 - fy/fyh) / 2.
        if abs(fyh) < 1e-9:
            fyh = 1.0
        v = (1.0 - fy / fyh) * 0.5
        col = int(round(min(max(u, 0.0), 1.0) * (W - 1)))
        row = int(round(min(max(v, 0.0), 1.0) * (H - 1)))
        return col, row

    @staticmethod
    def _fit_linear(d, z):
        """Least-squares fit z = A*d + B. Returns (A, B) or (None, None)."""
        if d.size < 2 or float(np.ptp(d)) < 1e-6:
            return None, None
        Amat = np.vstack([d, np.ones_like(d)]).T
        try:
            sol, *_ = np.linalg.lstsq(Amat, z, rcond=None)
            return float(sol[0]), float(sol[1])
        except Exception:
            return None, None

    @staticmethod
    def _film_to_pixel_vec(FX, FY, W, H, fyh):
        if abs(fyh) < 1e-9:
            fyh = 1.0
        u = (FX + 1.0) * 0.5
        v = (1.0 - FY / fyh) * 0.5
        cols = np.clip(np.round(u * (W - 1)), 0, W - 1).astype(np.int64)
        rows = np.clip(np.round(v * (H - 1)), 0, H - 1).astype(np.int64)
        return cols, rows

    def _idw_residual(self, FX, FY, cfx, cfy, r):
        """Inverse-distance-weighted (Shepard) interpolation of the per-point
        residuals: each control point corrects metric depth in its own area,
        blending by distance toward the nearest points."""
        dx = FX[:, None] - cfx[None, :]
        dy = FY[:, None] - cfy[None, :]
        d2 = dx * dx + dy * dy
        w = 1.0 / (np.power(d2, self.IDW_POWER * 0.5) + self.IDW_EPS)
        wsum = w.sum(axis=1)
        wsum[wsum < 1e-12] = 1e-12
        return (w * r[None, :]).sum(axis=1) / wsum

    @staticmethod
    def _dilate_grid_mask(mask2d, iters):
        """8-connected (3x3) binary dilation of a (stride,stride) bool grid,
        repeated `iters` times. Grows the kept region outward by one grid cell
        per pass — the inverse of the all-4-corners erosion in
        _build_grid_faces. Dilated nodes still have valid depth (Z_grid is
        computed for every node), so they're safe to include."""
        m = mask2d.copy()
        for _ in range(int(iters)):
            out = m.copy()
            out[:-1, :] |= m[1:, :]
            out[1:, :] |= m[:-1, :]
            out[:, :-1] |= m[:, 1:]
            out[:, 1:] |= m[:, :-1]
            out[:-1, :-1] |= m[1:, 1:]
            out[1:, 1:] |= m[:-1, :-1]
            out[:-1, 1:] |= m[1:, :-1]
            out[1:, :-1] |= m[:-1, 1:]
            m = out
        return m

    @staticmethod
    def _smooth_grid_z(Z, valid, iters):
        """Mask-aware 3x3 box smoothing of a (stride,stride) depth grid,
        repeated `iters` times. Each valid cell is replaced by the average of
        its valid 3x3 neighbours; invalid cells are left untouched and never
        contribute (no bleeding across removed regions)."""
        Zc = Z.astype(np.float64).copy()
        w0 = valid.astype(np.float64)
        Hh, Ww = Zc.shape
        for _ in range(int(iters)):
            acc = np.zeros_like(Zc)
            wsum = np.zeros_like(Zc)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    # dest[y,x] gathers neighbour src[y+dy, x+dx].
                    ys_d, ye_d = max(0, -dy), Hh - max(0, dy)
                    xs_d, xe_d = max(0, -dx), Ww - max(0, dx)
                    ys_s, ye_s = max(0, dy), Hh - max(0, -dy)
                    xs_s, xe_s = max(0, dx), Ww - max(0, -dx)
                    vw = w0[ys_s:ye_s, xs_s:xe_s]
                    acc[ys_d:ye_d, xs_d:xe_d] += (
                        Zc[ys_s:ye_s, xs_s:xe_s] * vw)
                    wsum[ys_d:ye_d, xs_d:xe_d] += vw
            new = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), Zc)
            Zc = np.where(valid, new, Zc)
        return Zc

    @staticmethod
    def _build_grid_faces(G, inside, verts, Zf, max_edge_ratio,
                          propagate_cells=0, max_edge_m=float("inf")):
        """Vectorised grid triangulation. A cell becomes 2 triangles only when
        all 4 corners are in-region AND none of its edges is "too long". An edge
        is too long when it fails EITHER test:

          • RELATIVE — edge_len / Z (forward distance at the edge midpoint)
            exceeds max_edge_ratio. Scale-invariant: a smoothly sampled surface
            has edge_len/Z ≈ const (the angular pixel pitch) at every depth, so
            this catches depth discontinuities (silhouettes / pile edges) at any
            distance. Pass max_edge_ratio=inf to disable.
          • ABSOLUTE — edge_len exceeds max_edge_m metres. Bounds the world size
            of any kept polygon so far geometry (large Z, where the relative
            test alone tolerates edges up to ratio·Z) doesn't keep over-long
            triangles. Pass max_edge_m=inf to disable.

        (Near-vertical / steep faces and tiny disconnected clusters are removed
        afterwards by _cleanup_relief, which works on the final triangle mesh
        with a metric radius — see that method.)

        Zf is the per-node forward distance from the camera (the calibrated
        depth Z that the verts were unprojected from), same length as verts.

        propagate_cells > 0 grows the set of cut cells outward by that many
        grid cells (Chebyshev radius) before triangulating, so the torn region
        around a long polygon is removed cleanly instead of leaving a jagged
        fringe.

        Returns (faces (T,3) int64, dropped_count)."""
        stride = G + 1
        V = verts.reshape(stride, stride, 3)
        ins = inside.reshape(stride, stride)
        Z = np.asarray(Zf, dtype=np.float64).reshape(stride, stride)
        # Forward distance at each edge midpoint (clamped away from 0 so the
        # ratio stays finite for samples that landed near the camera plane).
        eps = 1e-3
        ZmH = np.maximum(0.5 * (Z[:, 1:] + Z[:, :-1]), eps)    # (stride, G)
        ZmV = np.maximum(0.5 * (Z[1:, :] + Z[:-1, :]), eps)    # (G, stride)
        ZmD = np.maximum(0.5 * (Z[1:, 1:] + Z[:-1, :-1]), eps)  # (G, G)

        # Edge vectors between neighbouring grid vertices, and their 3D lengths.
        eH = V[:, 1:, :] - V[:, :-1, :]    # (stride, G, 3) horizontal
        eV = V[1:, :, :] - V[:-1, :, :]    # (G, stride, 3) vertical
        eD = V[1:, 1:, :] - V[:-1, :-1, :]  # (G, G, 3) diagonal
        lH = np.linalg.norm(eH, axis=2)
        lV = np.linalg.norm(eV, axis=2)
        lD = np.linalg.norm(eD, axis=2)
        mr = float(max_edge_ratio)
        me = float(max_edge_m)

        # Per-edge: short enough RELATIVE to distance AND under the ABSOLUTE cap.
        okH = ((lH / ZmH) <= mr) & (lH <= me)
        okV = ((lV / ZmV) <= mr) & (lV <= me)
        okD = ((lD / ZmD) <= mr) & (lD <= me)

        ti = np.arange(G)[:, None]
        si = np.arange(G)[None, :]
        a = ti * stride + si          # top-left corner index of each cell
        b = a + 1                     # top-right
        c = a + stride                # bottom-left
        d = c + 1                     # bottom-right

        cell_in = (ins[:-1, :-1] & ins[:-1, 1:] &
                   ins[1:, :-1] & ins[1:, 1:])    # (G, G)

        # A cell is "good" only when all five of its edges (4 sides + the
        # shared diagonal) pass the length test.
        cell_ok = (okH[:-1, :] & okH[1:, :] &
                   okV[:, :-1] & okV[:, 1:] & okD)
        bad = cell_in & (~cell_ok)
        if propagate_cells and int(propagate_cells) > 0 and bad.any():
            bad = DepthReconstructor._dilate_grid_mask(bad, int(propagate_cells))
        keep = cell_in & (~bad)       # (G, G)

        if keep.any():
            f1 = np.stack([a[keep], b[keep], d[keep]], axis=1)
            f2 = np.stack([a[keep], d[keep], c[keep]], axis=1)
            faces = np.concatenate([f1, f2], axis=0).astype(np.int64)
        else:
            faces = np.zeros((0, 3), np.int64)

        # In-region triangles dropped (by cutoff + propagation).
        dropped = int(2 * int(cell_in.sum()) - faces.shape[0])
        return faces, dropped

    # ------------------------------------------------------------------
    # Final mesh cleanup: steep-face removal + small-cluster removal
    # ------------------------------------------------------------------
    def _cleanup_relief(self, verts, faces):
        """Remove near-vertical faces and tiny disconnected clusters from a
        finished triangle mesh.

          1. STEEP cut: a triangle whose surface tilts more than
             STEEP_MAX_ANGLE_DEG from horizontal (|n·Zup| < cos(angle)) is
             removed, together with every triangle whose centroid is within
             STEEP_REMOVE_RADIUS_M metres of a steep one (a metric removal
             radius around each strong drop).
          2. CLUSTER cut: of what remains, every connected group of triangles
             whose world bounding box is smaller than MIN_CLUSTER_SIZE_M in its
             largest dimension is dropped.

        Operates on the final mesh (after the truck-body Boolean and, when
        enabled, after extrapolation/volume), so it catches vertical walls
        regardless of which stage produced them. Returns (verts, faces)
        re-indexed onto the surviving triangles."""
        verts = np.asarray(verts, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if faces.shape[0] == 0:
            return verts, faces
        n0 = faces.shape[0]

        # --- 1. steep faces (per-triangle normal) ---
        if self.STEEP_CUTOFF and self.STEEP_MAX_ANGLE_DEG is not None:
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]
            nrm = np.cross(v1 - v0, v2 - v0)
            ln = np.linalg.norm(nrm, axis=1)
            nz = np.where(ln > 1e-12, np.abs(nrm[:, 2]) / np.maximum(ln, 1e-12),
                          1.0)
            cos_t = math.cos(math.radians(float(self.STEEP_MAX_ANGLE_DEG)))
            steep = nz < cos_t
            cent = (v0 + v1 + v2) / 3.0

            # Optional metric margin: grow the steep set to faces within
            # STEEP_REMOVE_RADIUS_M of a steep one, so the cut keeps a clean
            # border around each strong drop.
            def _grow(mask):
                radius = float(self.STEEP_REMOVE_RADIUS_M or 0.0)
                if radius > 0.0 and mask.any() and (~mask).any():
                    try:
                        from scipy.spatial import cKDTree
                        near = cKDTree(cent[mask]).query_ball_point(
                            cent, r=radius)
                        return np.fromiter((len(x) > 0 for x in near),
                                           dtype=bool, count=len(near))
                    except Exception as exc:
                        self._log(f"Очистка: радиус среза недоступен ({exc}).")
                return mask

            edge_only = bool(getattr(self, "STEEP_EDGE_ONLY", True))
            if not edge_only:
                # Remove every steep face (+ margin), regardless of position.
                steep = _grow(steep)
                faces = faces[~steep]
                self._log(
                    f"Очистка: удалено {int(steep.sum())} крутых граней "
                    f"(>{float(self.STEEP_MAX_ANGLE_DEG):g}°, без ограничения "
                    f"по борту).")
            elif steep.any():
                # Near-wall only: "removable" = steep (grown by the margin) AND
                # within STEEP_NEAR_WALL_M of the nearest truck-body wall (XY
                # distance to the napolnitel footprint perimeter). A steep face
                # farther into the load is a real feature and is kept.
                removable = _grow(steep)
                near = float(getattr(self, "STEEP_NEAR_WALL_M", 0.5) or 0.0)
                poly = self._truck_wall_footprint()
                if poly is None:
                    self._log("Очистка: борт кузова (napolnitel) не найден — "
                              "ограничение по борту пропущено.")
                elif near > 0.0 and removable.any():
                    try:
                        dw = self._dist_point_to_polygon_edges(
                            cent[:, :2], poly)
                        removable = removable & (dw <= near)
                    except Exception as exc:
                        self._log(f"Очистка: расстояние до борта "
                                  f"недоступно ({exc}).")
                # Avoid punching internal holes: on an OPEN mesh keep only the
                # removable faces connected (through other removable faces) to
                # an open boundary, so the removed region always reaches the
                # rim. A CLOSED mesh (volume solid) has no open boundary — there
                # the near-wall steep faces are the perimeter walls, drop them
                # directly.
                bface = self._boundary_faces(faces)
                if bface.any():
                    remove = self._flood_faces_to_boundary(
                        faces, removable, bface)
                else:
                    remove = removable
                faces = faces[~remove]
                self._log(
                    f"Очистка: срез откоса в {near:g} м от борта — удалено "
                    f"{int(remove.sum())} граней (без внутренних дыр).")
            if faces.shape[0] == 0:
                self._log("Очистка: после среза вертикалей не осталось граней.")
                return verts, faces

        # --- 2. thin / small disconnected clusters ---
        # A cluster is judged by its WIDTH — the narrowest extent of its XY
        # footprint (minor principal axis). Using the width, not the longest
        # span, removes both compact specks AND long-thin slivers (which are
        # long but narrow, so the old max-span test let them through), while
        # keeping the genuinely large main area (wide in both axes).
        min_c = float(self.MIN_CLUSTER_SIZE_M or 0.0)
        if min_c > 0.0 and faces.shape[0] > 0 and trimesh is not None:
            try:
                import trimesh as tm
                mesh = tm.Trimesh(vertices=verts, faces=faces, process=False)
                adj = mesh.face_adjacency
                labels = tm.graph.connected_component_labels(
                    adj, node_count=faces.shape[0])
                keep = np.ones(faces.shape[0], dtype=bool)
                for lab in np.unique(labels):
                    sel = labels == lab
                    pts = verts[np.unique(faces[sel].reshape(-1))]
                    if self._cluster_width_xy(pts) < min_c:
                        keep[sel] = False
                faces = faces[keep]
            except Exception as exc:
                self._log(f"Очистка: удаление кластеров пропущено ({exc}).")

        if faces.shape[0] == 0:
            return verts, faces
        # Re-index onto the surviving vertices.
        used = np.unique(faces.reshape(-1))
        remap = np.full(verts.shape[0], -1, dtype=np.int64)
        remap[used] = np.arange(used.shape[0])
        new_verts = verts[used]
        new_faces = remap[faces]
        if new_faces.shape[0] != n0:
            ang = self.STEEP_MAX_ANGLE_DEG
            ang_s = "off" if (not self.STEEP_CUTOFF or ang is None) \
                else f">{float(ang):g}°"
            self._log(
                f"Очистка рельефа: грани {n0} → {new_faces.shape[0]} "
                f"(вертикали {ang_s}, узкие/мелкие кластеры <{min_c:g} м).")
        return new_verts, new_faces

    @staticmethod
    def _boundary_faces(faces):
        """Bool mask of faces that own at least one OPEN (naked) edge — an edge
        used by a single triangle. These sit on the outer perimeter or a hole
        rim. A fully closed mesh returns an all-False mask."""
        f = np.asarray(faces, dtype=np.int64)
        nf = f.shape[0]
        if nf == 0:
            return np.zeros(0, dtype=bool)
        e = np.sort(np.concatenate(
            [f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0), axis=1)
        # Edges are stacked [all e01; all e12; all e20], so edge-row k belongs
        # to face k % nf  →  tile, not repeat.
        face_of = np.tile(np.arange(nf), 3)
        _, inv, cnt = np.unique(e, axis=0, return_inverse=True,
                                return_counts=True)
        naked = cnt[inv] == 1
        bf = np.zeros(nf, dtype=bool)
        np.logical_or.at(bf, face_of, naked)
        return bf

    @staticmethod
    def _flood_faces_to_boundary(faces, removable, boundary_face):
        """Of the `removable` faces, return the mask of those connected —
        through other removable faces, across shared edges — to an open-boundary
        face. Removing exactly these never opens an internal hole, because the
        removed region always reaches the mesh edge; an isolated removable speck
        sitting inside the surface is left in place. Pure numpy + scipy."""
        f = np.asarray(faces, dtype=np.int64)
        nf = f.shape[0]
        rem = np.asarray(removable, dtype=bool)
        seed = rem & np.asarray(boundary_face, dtype=bool)
        if nf == 0 or not seed.any():
            return np.zeros(nf, dtype=bool)
        # Face adjacency: two faces are adjacent when they share an edge, found
        # as identical consecutive rows in the lexicographically sorted edge
        # list.
        e = np.sort(np.concatenate(
            [f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0), axis=1)
        # Edge-row k belongs to face k % nf (edges stacked by type, see above).
        face_of = np.tile(np.arange(nf), 3)
        order = np.lexsort((e[:, 1], e[:, 0]))
        e_s = e[order]
        fo_s = face_of[order]
        same = np.all(e_s[1:] == e_s[:-1], axis=1)
        a = fo_s[:-1][same]
        b = fo_s[1:][same]
        # Keep only removable–removable adjacency, so the flood can't cross a
        # kept (non-removable) face.
        m = rem[a] & rem[b]
        a, b = a[m], b[m]
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components
            ij = (np.concatenate([a, b]), np.concatenate([b, a]))
            M = csr_matrix((np.ones(ij[0].size), ij), shape=(nf, nf))
            _, labels = connected_components(M, directed=False)
            seed_labels = np.unique(labels[seed])
            return rem & np.isin(labels, seed_labels)
        except Exception:
            # Fallback: remove just the boundary-touching removable faces.
            return seed

    def _truck_wall_footprint(self):
        """XY footprint polygon of the truck body (napolnitel) — the convex
        hull of its vertices projected to the ground plane, i.e. the outline of
        the bed walls. Cached on the instance (the truck is fixed per session).
        Returns an (M,2) array of ordered polygon vertices, or None."""
        cached = getattr(self, "_wall_footprint_xy", None)
        if cached is not None:
            return cached
        loaded = self._load_tonar_obj()
        if loaded is None:
            return None
        tv, _ = loaded
        xy = np.asarray(tv, dtype=np.float64)[:, :2]
        if xy.shape[0] < 3:
            return None
        try:
            from scipy.spatial import ConvexHull
            poly = xy[ConvexHull(xy).vertices]
        except Exception:
            mn = xy.min(axis=0)
            mx = xy.max(axis=0)
            poly = np.array([[mn[0], mn[1]], [mx[0], mn[1]],
                             [mx[0], mx[1]], [mn[0], mx[1]]])
        self._wall_footprint_xy = poly
        return poly

    @staticmethod
    def _dist_point_to_polygon_edges(pts_xy, poly):
        """Unsigned distance from each XY point to the nearest edge segment of
        a closed polygon (its perimeter = the truck walls). Vectorised over
        points × polygon edges. Returns (N,) distances in metres."""
        P = np.asarray(pts_xy, dtype=np.float64)            # (N,2)
        A = np.asarray(poly, dtype=np.float64)              # (M,2)
        B = np.roll(A, -1, axis=0)                          # (M,2) next vertex
        AB = B - A                                          # (M,2)
        ab2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-12)  # (M,)
        AP = P[:, None, :] - A[None, :, :]                  # (N,M,2)
        t = np.clip(np.einsum("nmj,mj->nm", AP, AB) / ab2[None, :], 0.0, 1.0)
        proj = A[None, :, :] + t[:, :, None] * AB[None, :, :]  # (N,M,2)
        d = np.linalg.norm(P[:, None, :] - proj, axis=2)    # (N,M)
        return d.min(axis=1)

    @staticmethod
    def _cluster_width_xy(pts):
        """Width of a point cluster = the narrowest extent of its XY footprint,
        measured along the cluster's own principal axes (PCA), so orientation
        doesn't matter. A long-thin sliver has a small width even though it is
        long; a compact speck is small in both axes; the main load area is wide
        in both. Returns the minor-axis extent in metres (0 for <2 points)."""
        xy = np.asarray(pts, dtype=np.float64)[:, :2]
        if xy.shape[0] < 2:
            return 0.0
        c = xy - xy.mean(axis=0)
        try:
            # Right singular vectors = principal axes of the 2D footprint.
            _, _, vt = np.linalg.svd(c, full_matrices=False)
            proj = c @ vt.T                     # coords along principal axes
            ext = proj.max(axis=0) - proj.min(axis=0)
        except Exception:
            # Fallback: axis-aligned minor extent.
            ext = c.max(axis=0) - c.min(axis=0)
        return float(ext.min())

    # ------------------------------------------------------------------
    # Gradient-aware extrapolation with mirror-tiled texture
    # ------------------------------------------------------------------
    def _build_target_extrapolated(self, valid_xyz):
        """Extend the measured relief to a TARGET_SIZE_M axis-aligned rectangle.
        `valid_xyz` is the (M,3) cloud of measured world points (after Stage 1
        cutoff + Stage 2 truck-body clip), so the source bbox is bounded to the
        truck and the target rectangle is centred on the real load — NOT on the
        whole frame.
        Inside the measured region: original measurements preserved verbatim.
        Outside:
        each cell's Z is set by

            Z(out) = Z(nearest in-mask seed)
                   + min(outward_grad_at_seed, −tan(EXTRAP_ANGLE_DEG))
                     × distance_to_seed
                   + texture_weight × mirror_residual × exp(−dist / decay)

        where outward_grad is the directional derivative of Z at the boundary
        cell in the direction pointing to this external cell. Taking the
        steeper (more negative) of that and the angle-of-repose lets a steep
        visible pile face keep falling at its observed rate (so the relief
        reaches the truck floor instead of forming a fake 'second pile'),
        while a flat or hidden peak still drops at a natural granular slope.

        The mirror-tiled residual carries the high-frequency texture of the
        visible side over to the hidden side, so the back of the pile keeps
        the look of a granular surface instead of being a featureless slope.
        Texture amplitude fades exponentially with distance from the mask.

        A floor clamp at min(visible Z) − EXTRAP_FLOOR_FROM_MIN_M (or an
        absolute EXTRAP_FLOOR_M) catches whatever the descent + texture
        produce so the extrapolated relief doesn't rise back up far from
        the mask via averaging artifacts.

        Returns (verts_target, faces_target, info) or None on failure."""
        TX_m, TY_m = (float(v) for v in self.TARGET_SIZE_M)
        if TX_m <= 0.0 or TY_m <= 0.0:
            return None
        valid_xyz = np.asarray(valid_xyz, dtype=np.float64)
        if valid_xyz.shape[0] < 10:
            return None
        sxmin = float(valid_xyz[:, 0].min())
        sxmax = float(valid_xyz[:, 0].max())
        symin = float(valid_xyz[:, 1].min())
        symax = float(valid_xyz[:, 1].max())
        sw = sxmax - sxmin
        sh = symax - symin
        if sw < 1e-3 or sh < 1e-3:
            return None

        # Target rectangle, centred on mask centroid.
        cx = 0.5 * (sxmin + sxmax)
        cy = 0.5 * (symin + symax)
        txmin = cx - 0.5 * TX_m
        txmax = cx + 0.5 * TX_m
        tymin = cy - 0.5 * TY_m
        tymax = cy + 0.5 * TY_m

        # Working grid covers union of mask bbox and target rect, with a
        # half-metre pad so the bilinear sampler doesn't clamp at the edge.
        pad = 0.5
        gxmin = min(sxmin, txmin) - pad
        gxmax = max(sxmax, txmax) + pad
        gymin = min(symin, tymin) - pad
        gymax = max(symax, tymax) + pad
        gw = gxmax - gxmin
        gh = gymax - gymin

        # Rasterize masked verts onto the working grid.
        src_field, src_valid = self._rasterize_xyz(
            valid_xyz, gxmin, gymin, gw, gh)
        if not src_valid.any():
            return None

        ny_s, nx_s = src_field.shape
        cell_dx = gw / max(nx_s - 1, 1)
        cell_dy = gh / max(ny_s - 1, 1)

        # Pass 1: BFS-Dijkstra from in-mask seeds. For every cell we get the
        # (y, x) of its nearest in-mask cell and the path distance (in cell
        # steps, 1 ortho / √2 diag, accumulated). The directional info we
        # use afterwards comes from the true Euclidean vector cell → seed.
        seed_y, seed_x, _dist_cells = self._bfs_seeds(src_valid)

        # Pass 2: ∂Z/∂x, ∂Z/∂y inside the mask (one-sided at boundary,
        # central in the interior). Smooth the gradient field a couple of
        # iterations to suppress single-cell noise from the rasterizer.
        grad_y, grad_x = self._masked_gradient_2d(
            src_field, src_valid, cell_dx, cell_dy)
        grad_x = self._smooth_grid_z(grad_x, src_valid, 2)
        grad_y = self._smooth_grid_z(grad_y, src_valid, 2)

        # Per-cell world XY and the true Euclidean vector to its seed.
        yy_idx, xx_idx = np.meshgrid(
            np.arange(ny_s), np.arange(nx_s), indexing="ij")
        cell_X = gxmin + xx_idx * cell_dx
        cell_Y = gymin + yy_idx * cell_dy
        seed_X = gxmin + seed_x * cell_dx
        seed_Y = gymin + seed_y * cell_dy
        dvec_x = cell_X - seed_X
        dvec_y = cell_Y - seed_Y
        dist_m = np.sqrt(dvec_x * dvec_x + dvec_y * dvec_y)
        safe_dist = np.where(dist_m > 1e-9, dist_m, 1.0)
        out_x = dvec_x / safe_dist
        out_y = dvec_y / safe_dist

        # Outward directional derivative of Z at the seed.
        seed_gx = grad_x[seed_y, seed_x]
        seed_gy = grad_y[seed_y, seed_x]
        out_grad = seed_gx * out_x + seed_gy * out_y

        # Effective descent rate (per metre): pick the more negative of
        # outward gradient and −tan(angle of repose); cap at −tan(max angle)
        # so a noisy one-sided gradient can't drive Z to infinity.
        tan_aor = math.tan(math.radians(float(self.EXTRAP_ANGLE_DEG)))
        tan_max = math.tan(math.radians(float(self.EXTRAP_MAX_ANGLE_DEG)))
        eff_desc = np.minimum(out_grad, -tan_aor)
        eff_desc = np.maximum(eff_desc, -tan_max)

        # Baseline Z field: in-mask keeps original, others get seed_Z + descent.
        seed_z = src_field[seed_y, seed_x]
        src_extrap = np.where(
            src_valid, src_field, seed_z + eff_desc * dist_m)

        # Texture: mirror-tile high-frequency residual from the visible
        # side onto the extrapolated cells, attenuated by distance from
        # the mask. CRITICAL: only apply texture where descent is governed
        # by the angle-of-repose fallback (hidden peak side). Where the
        # local outward gradient is steeper than AoR (the camera-facing
        # wall side, with the visible face falling toward the floor), the
        # texture is suppressed — otherwise mirror-tiling produces an
        # "echo" of the visible peak at a symmetric distance, which the
        # observer reads as a fake second pile near the near wall.
        tex_w = float(self.EXTRAP_TEXTURE_WEIGHT)
        if tex_w > 0.0:
            smooth_iters = max(1, int(self.EXTRAP_TEXTURE_SMOOTH_ITERS))
            z_for_trend = np.where(src_valid, src_field, 0.0)
            z_trend = self._smooth_grid_z(
                z_for_trend, src_valid, smooth_iters)
            residual_field = np.where(src_valid, src_field - z_trend, 0.0)
            sample_X = self._mirror_into(cell_X, sxmin, sxmax)
            sample_Y = self._mirror_into(cell_Y, symin, symax)
            sampled_residual = self._bilinear_sample(
                residual_field, sample_X, sample_Y,
                gxmin, gxmax, gymin, gymax)
            decay = max(float(self.EXTRAP_TEXTURE_DECAY_M), 1e-6)
            attenuation = np.exp(-dist_m / decay)
            # Smooth ramp 0..1 as out_grad goes from −tan(AoR) (steep) up
            # to 0 (flat) — full texture above 0, zero texture in steep
            # descent zones.
            tan_aor_safe = max(tan_aor, 1e-6)
            gradient_weight = np.clip(
                (out_grad + tan_aor_safe) / tan_aor_safe, 0.0, 1.0)
            texture_add = np.where(
                src_valid, 0.0,
                tex_w * gradient_weight * sampled_residual * attenuation)
            src_extrap = src_extrap + texture_add

        # Floor clamp — kills any artifact that would lift the extrapolated
        # relief back above the bottom of the visible pile.
        floor_z = None
        if self.EXTRAP_FLOOR_FROM_MIN_M is not None:
            z_min_visible = float(src_field[src_valid].min())
            floor_z = z_min_visible - float(self.EXTRAP_FLOOR_FROM_MIN_M)
        elif self.EXTRAP_FLOOR_M is not None:
            floor_z = float(self.EXTRAP_FLOOR_M)
        if floor_z is not None:
            src_extrap = np.maximum(src_extrap, floor_z)

        # Sample on the target grid.
        tnx = max(2, int(round(TX_m * self.TARGET_RES_PER_M)) + 1)
        tny = max(2, int(round(TY_m * self.TARGET_RES_PER_M)) + 1)
        TXg, TYg = np.meshgrid(np.linspace(txmin, txmax, tnx),
                               np.linspace(tymin, tymax, tny))
        Z_target = self._bilinear_sample(
            src_extrap, TXg, TYg, gxmin, gxmax, gymin, gymax)

        # Smoothing of extrapolated cells only (preserve direct in-mask
        # measurements).
        iters = int(self.TARGET_SMOOTH_ITERS)
        if iters > 0:
            valid_direct = self._bilinear_sample(
                src_valid.astype(np.float64), TXg, TYg,
                gxmin, gxmax, gymin, gymax) > 0.5
            ones = np.ones_like(Z_target, dtype=bool)
            Z_smooth = self._smooth_grid_z(Z_target, ones, iters)
            Z_target = np.where(valid_direct, Z_target, Z_smooth)

        n_total = tnx * tny
        verts_t = np.empty((n_total, 3), dtype=np.float64)
        verts_t[:, 0] = TXg.ravel()
        verts_t[:, 1] = TYg.ravel()
        verts_t[:, 2] = Z_target.ravel()
        faces_t = self._build_regular_grid_faces(tnx, tny)

        info = {
            "src_bbox": (sxmin, sxmax, symin, symax),
            "target_bbox": (txmin, txmax, tymin, tymax),
            "src_size_m": (sw, sh),
            "target_size_m": (TX_m, TY_m),
            "target_grid": (tnx, tny),
            "src_valid_cells": int(src_valid.sum()),
            "extrap_angle_deg": float(self.EXTRAP_ANGLE_DEG),
            "extrap_texture_weight": tex_w,
            "extrap_floor_z": floor_z,
        }
        return verts_t, faces_t, info

    def _rasterize_xyz(self, pts, xmin, ymin, w, h):
        """Bin (M, 3) world points into a regular XY grid covering
        [xmin, xmin+w] × [ymin, ymin+h] at TARGET_RES_PER_M density.
        Returns (field (ny, nx), valid_mask (ny, nx))."""
        nx = max(2, int(round(w * self.TARGET_RES_PER_M)) + 1)
        ny = max(2, int(round(h * self.TARGET_RES_PER_M)) + 1)
        ix = np.clip(
            np.round((pts[:, 0] - xmin) / max(w, 1e-9) * (nx - 1)),
            0, nx - 1).astype(np.int64)
        iy = np.clip(
            np.round((pts[:, 1] - ymin) / max(h, 1e-9) * (ny - 1)),
            0, ny - 1).astype(np.int64)
        accum = np.zeros((ny, nx), dtype=np.float64)
        counts = np.zeros((ny, nx), dtype=np.int64)
        np.add.at(accum, (iy, ix), pts[:, 2].astype(np.float64))
        np.add.at(counts, (iy, ix), 1)
        valid = counts > 0
        field = np.where(valid, accum / np.maximum(counts, 1), 0.0)
        return field, valid

    @staticmethod
    def _bfs_seeds(valid):
        """Multi-source Dijkstra-style BFS over the boolean array `valid`.
        For each grid cell, returns the (y, x) coords of its nearest True
        cell and the accumulated path distance in cell-step units (1 for
        orthogonal moves, √2 for diagonals). True cells get distance 0 and
        seed = themselves.

        Returns (seed_y int32, seed_x int32, dist float64), all with the
        shape of `valid`. Cells unreachable from any seed (only possible
        when `valid` is empty) keep distance=inf and seed=−1."""
        Hh, Ww = valid.shape
        seed_y = np.full((Hh, Ww), -1, dtype=np.int32)
        seed_x = np.full((Hh, Ww), -1, dtype=np.int32)
        dist = np.full((Hh, Ww), np.inf, dtype=np.float64)
        yi, xi = np.where(valid)
        if yi.size == 0:
            return seed_y, seed_x, dist
        seed_y[yi, xi] = yi.astype(np.int32)
        seed_x[yi, xi] = xi.astype(np.int32)
        dist[yi, xi] = 0.0

        sqrt2 = math.sqrt(2.0)
        # Worst-case BFS depth is the grid's longest axis; ×2 gives slack
        # for the Gauss-Seidel relaxation order.
        max_iters = max(Hh, Ww) * 2
        for _ in range(max_iters):
            improved = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    step = 1.0 if (dx == 0 or dy == 0) else sqrt2
                    ys_d, ye_d = max(0, -dy), Hh - max(0, dy)
                    xs_d, xe_d = max(0, -dx), Ww - max(0, dx)
                    ys_s, ye_s = max(0, dy), Hh - max(0, -dy)
                    xs_s, xe_s = max(0, dx), Ww - max(0, -dx)
                    src_dist = dist[ys_s:ye_s, xs_s:xe_s]
                    tgt_dist = dist[ys_d:ye_d, xs_d:xe_d]
                    cand = src_dist + step
                    upd = cand < tgt_dist
                    if upd.any():
                        tgt_dist[upd] = cand[upd]
                        seed_y[ys_d:ye_d, xs_d:xe_d][upd] = \
                            seed_y[ys_s:ye_s, xs_s:xe_s][upd]
                        seed_x[ys_d:ye_d, xs_d:xe_d][upd] = \
                            seed_x[ys_s:ye_s, xs_s:xe_s][upd]
                        improved = True
            if not improved:
                break
        return seed_y, seed_x, dist

    @staticmethod
    def _masked_gradient_2d(field, valid, dx, dy):
        """Per-cell ∂Z/∂x and ∂Z/∂y, computed using only in-mask neighbours.
        Central difference when both neighbours are in-mask; forward or
        backward one-sided diff when only one side is. Out-of-mask cells
        return 0 gradient. dx, dy are the world-space cell sizes in metres.
        Returns (grad_y, grad_x)."""
        Hh, Ww = field.shape
        v = valid.astype(bool)
        f = field.astype(np.float64)

        # X direction (right neighbour at column j+1, left at j-1).
        v_R = np.zeros_like(v); v_R[:, :-1] = v[:, 1:]
        v_L = np.zeros_like(v); v_L[:, 1:] = v[:, :-1]
        z_R = np.zeros_like(f); z_R[:, :-1] = f[:, 1:]
        z_L = np.zeros_like(f); z_L[:, 1:] = f[:, :-1]
        central_x = (z_R - z_L) / (2.0 * dx)
        fwd_x = (z_R - f) / dx
        bwd_x = (f - z_L) / dx
        grad_x = np.where(
            v & v_R & v_L, central_x,
            np.where(v & v_R, fwd_x,
                     np.where(v & v_L, bwd_x, 0.0)))

        # Y direction (down neighbour at row i+1, up at i-1).
        v_D = np.zeros_like(v); v_D[:-1, :] = v[1:, :]
        v_U = np.zeros_like(v); v_U[1:, :] = v[:-1, :]
        z_D = np.zeros_like(f); z_D[:-1, :] = f[1:, :]
        z_U = np.zeros_like(f); z_U[1:, :] = f[:-1, :]
        central_y = (z_D - z_U) / (2.0 * dy)
        fwd_y = (z_D - f) / dy
        bwd_y = (f - z_U) / dy
        grad_y = np.where(
            v & v_D & v_U, central_y,
            np.where(v & v_D, fwd_y,
                     np.where(v & v_U, bwd_y, 0.0)))

        return grad_y, grad_x

    @staticmethod
    def _mirror_into(t, lo, hi):
        """Triangle-wave reflection of `t` into [lo, hi]. Used to sample the
        in-mask residual texture at mirrored positions for the back side
        of the pile. Vectorised."""
        span = float(hi) - float(lo)
        if span <= 1e-9:
            return np.full_like(t, lo)
        period = 2.0 * span
        d = np.mod(t - lo, period)
        return np.where(d <= span, lo + d, lo + (period - d))

    @staticmethod
    def _bilinear_sample(field, xs, ys, xmin, xmax, ymin, ymax):
        """Bilinear sample `field` (ny, nx) defined over
        [xmin, xmax] × [ymin, ymax] at world coords (xs, ys). Clamps to the
        nearest edge outside the source domain."""
        ny, nx = field.shape
        if (nx < 2 or ny < 2 or
                xmax - xmin <= 1e-9 or ymax - ymin <= 1e-9):
            fill = float(field.mean()) if field.size else 0.0
            return np.full_like(xs, fill, dtype=np.float64)
        u = (xs - xmin) / (xmax - xmin) * (nx - 1)
        v = (ys - ymin) / (ymax - ymin) * (ny - 1)
        u = np.clip(u, 0.0, nx - 1)
        v = np.clip(v, 0.0, ny - 1)
        i0 = np.floor(u).astype(np.int64)
        j0 = np.floor(v).astype(np.int64)
        i1 = np.minimum(i0 + 1, nx - 1)
        j1 = np.minimum(j0 + 1, ny - 1)
        fu = u - i0
        fv = v - j0
        z00 = field[j0, i0]
        z01 = field[j0, i1]
        z10 = field[j1, i0]
        z11 = field[j1, i1]
        return ((1.0 - fu) * (1.0 - fv) * z00 + fu * (1.0 - fv) * z01 +
                (1.0 - fu) * fv * z10 + fu * fv * z11)

    @staticmethod
    def _build_regular_grid_faces(nx, ny):
        """Triangulate a regular (nx by ny)-vertex grid: 2 tris per cell."""
        cx = nx - 1
        cy = ny - 1
        if cx <= 0 or cy <= 0:
            return np.zeros((0, 3), np.int64)
        ti = np.arange(cy)[:, None]
        si = np.arange(cx)[None, :]
        a = (ti * nx + si).ravel()
        b = a + 1
        c = a + nx
        d = c + 1
        f1 = np.stack([a, b, d], axis=1)
        f2 = np.stack([a, d, c], axis=1)
        return np.concatenate([f1, f2], axis=0).astype(np.int64)

    # ------------------------------------------------------------------
    # Local clipping (relief ∩ tonar_napolnitel) via point-in-mesh test
    # ------------------------------------------------------------------
    def _tonar_obj_path(self) -> str:
        """Absolute path to tonar_napolnitel.obj, resolved relative to the
        project root (two levels above this module file)."""
        rel = self.TONAR_OBJ_REL_PATH
        if not rel:
            return ""
        here = os.path.dirname(os.path.abspath(__file__))
        proj_root = os.path.dirname(os.path.dirname(here))
        return os.path.normpath(os.path.join(proj_root, rel))

    def _load_tonar_obj(self):
        """Load tonar_napolnitel.obj as (verts float32 (N,3),
        faces uint32 (M,3)). Applies TONAR_OBJ_TRANSFORM if set.
        Returns None on failure."""
        if trimesh is None:
            self._log("trimesh не установлен — boolean невозможен.")
            return None
        p = self._tonar_obj_path()
        if not p or not os.path.exists(p):
            self._log(f"tonar OBJ не найден: {p}")
            return None
        try:
            m = trimesh.load(p, force="mesh", process=False)
        except Exception as exc:
            self._log(f"Не удалось загрузить {os.path.basename(p)}: {exc}")
            return None
        verts = np.asarray(m.vertices, dtype=np.float64)
        faces = np.asarray(m.faces, dtype=np.uint32)

        # Blender-exported OBJs carry per-corner UV/normals, and
        # trimesh.load(process=False) treats every distinct (v, vt, vn)
        # tuple as a separate vertex — so a 36-vertex tonar becomes 108
        # vertices, all in coincident triples that share no topology
        # with their neighbours. Carve in Blender 2.70 sees this as a
        # disconnected triangle soup and bails with "non intersecting
        # group is not IN or OUT". Collapse exact-position duplicates
        # back into a manifold mesh.
        n_before = verts.shape[0]
        unique_v, inverse = np.unique(
            np.round(verts, decimals=6), axis=0, return_inverse=True)
        if unique_v.shape[0] < n_before:
            # Recover the original (un-rounded) coords for the surviving
            # vertices by taking the first occurrence of each unique key.
            first_idx = np.zeros(unique_v.shape[0], dtype=np.int64)
            seen = np.full(unique_v.shape[0], -1, dtype=np.int64)
            for i, j in enumerate(inverse):
                if seen[j] < 0:
                    seen[j] = i
            first_idx = seen
            verts = verts[first_idx]
            faces = inverse[faces].astype(np.uint32)
            self._log(f"Tonar: дедупликация вершин {n_before} → "
                      f"{verts.shape[0]} (Carve требует разделяемой "
                      f"топологии, не triangle soup).")

        tr = self.TONAR_OBJ_TRANSFORM
        if tr is not None:
            try:
                T = np.asarray(tr, dtype=np.float64).reshape(4, 4)
                v4 = np.column_stack([verts, np.ones(verts.shape[0])])
                verts = (v4 @ T.T)[:, :3]
                # A transform with negative determinant (reflection,
                # not a proper rotation) reverses every triangle's
                # winding — the result has inward-facing normals, and
                # Blender 2.70's Carve solver then silently bails out
                # of any Boolean op (mod_apply returns FINISHED but the
                # mesh is unmodified). Restore outward winding by
                # swapping the first and last index of every face.
                if float(np.linalg.det(T[:3, :3])) < 0.0:
                    faces = faces[:, [2, 1, 0]].astype(np.uint32)
                    self._log("TONAR_OBJ_TRANSFORM имеет det<0 — winding "
                              "перевёрнут, чтобы нормали смотрели наружу.")
            except Exception as exc:
                self._log(f"TONAR_OBJ_TRANSFORM некорректен: {exc} — игнорирую.")
        return verts.astype(np.float32), faces

    def _clip_relief_to_body(self, relief_verts, relief_faces):
        """Stage 2 — clip the relief to the truck body via a Boolean with the
        napolnitel (filler) mesh. The napolnitel OBJ is a closed solid that
        fills the inside of the truck body; we keep only the relief that lies
        INSIDE it and discard every triangle that spills outside (the pile
        geometry beyond the bed).

        Each relief vertex is point-in-mesh tested against the napolnitel
        solid; a triangle survives only when all 3 of its corners are inside.
        Surviving faces are re-indexed onto the vertices they actually use.
        Returns (verts (N,3) float64, faces (M,3) int64) or None on any
        failure (no OBJ / non-watertight mesh / containment unavailable), in
        which case the caller keeps the un-clipped relief.
        """
        loaded = self._load_tonar_obj()
        if loaded is None:
            return None
        t_verts, t_faces = loaded
        t_verts = np.asarray(t_verts, dtype=np.float64)
        t_faces = np.asarray(t_faces, dtype=np.int64)

        # Translucent overlay so the user sees the body the relief is cut to.
        try:
            if self._tonar_debug_node is not None:
                try:
                    self._tonar_debug_node.remove_node()
                except Exception:
                    pass
                self._tonar_debug_node = None
            self._tonar_debug_node = self._build_tonar_debug_mesh(
                t_verts, t_faces)
        except Exception as exc:
            self._log(f"tonar overlay build failed: {exc}")

        try:
            body = trimesh.Trimesh(vertices=t_verts, faces=t_faces,
                                   process=False)
        except Exception as exc:
            self._log(f"Клиппинг по кузову: не удалось собрать napolnitel: "
                      f"{exc}")
            return None
        if not body.is_watertight:
            self._log("Клиппинг по кузову: napolnitel не водонепроницаем — "
                      "point-in-mesh может быть неточным.")

        rv = np.asarray(relief_verts, dtype=np.float64)
        rf = np.asarray(relief_faces, dtype=np.int64)
        try:
            inside_v = body.contains(rv)               # (N,) bool
        except Exception as exc:
            self._log(f"Клиппинг по кузову: contains() упал: {exc} — "
                      f"оставляю необрезанный рельеф.")
            return None

        face_in = inside_v[rf].all(axis=1)             # all 3 corners inside
        kept = rf[face_in]
        if kept.shape[0] == 0:
            self._log("Клиппинг по кузову: внутри napolnitel не осталось "
                      "треугольников — оставляю необрезанный рельеф.")
            return None

        # Re-index onto the surviving vertices to drop the orphans.
        used = np.unique(kept.reshape(-1))
        remap = np.full(rv.shape[0], -1, dtype=np.int64)
        remap[used] = np.arange(used.shape[0])
        new_verts = rv[used]
        new_faces = remap[kept]
        self._log(
            f"Клиппинг по кузову (napolnitel): оставлено "
            f"{new_faces.shape[0]}/{rf.shape[0]} тр., "
            f"{new_verts.shape[0]} верш. — отсечена геометрия вне кузова.")
        return new_verts, new_faces

    def _clip_relief_to_target(self, relief_verts, relief_faces, grid_dims):
        """Turn the open rectangular relief heightfield into the closed
        FILLER VOLUME inside tonar_napolnitel via a headless Blender 2.70
        Boolean DIFFERENCE.

        Algorithm (mirrors mesh_reconstruction.cpp::create_mesh_advanced
        and boolean_operations.cpp::perform_boolean_difference):
          1. Seal the relief into a watertight solid representing the
             empty space ABOVE the relief — floor = the heightfield,
             ceiling = flat plane at z_high = max(tonar.z) + 1 m, walls
             = perimeter strips connecting them.
          2. Blender DIFFERENCE: tonar − sealed_solid → the part of the
             container BELOW the relief = the filler.

        The result is a closed manifold with a measurable volume (the
        Stokes/divergence-theorem formula  V = |Σ v0·(v1×v2)| / 6).
        Returns (verts, faces, volume_m3) or None on any failure (the
        caller then keeps the un-clipped, open relief).
        """
        if grid_dims is None:
            self._log("Клиппинг: нет размеров сетки рельефа — пропуск.")
            return None
        tnx, tny = int(grid_dims[0]), int(grid_dims[1])
        loaded = self._load_tonar_obj()
        if loaded is None:
            return None
        t_verts, t_faces = loaded
        t_verts = t_verts.astype(np.float64)
        t_faces = np.asarray(t_faces, dtype=np.int64)

        # Refresh the translucent tonar overlay before the boolean runs,
        # so the user sees the .obj that's fed into the operation even if
        # Blender later fails.
        try:
            if self._tonar_debug_node is not None:
                try:
                    self._tonar_debug_node.remove_node()
                except Exception:
                    pass
                self._tonar_debug_node = None
            self._tonar_debug_node = self._build_tonar_debug_mesh(
                t_verts, t_faces)
            self._log(f"Tonar overlay: {t_verts.shape[0]} верш., "
                      f"{t_faces.shape[0]} тр., alpha 25%.")
        except Exception as exc:
            self._log(f"tonar overlay build failed: {exc}")

        r_verts = np.asarray(relief_verts, dtype=np.float64)
        if r_verts.shape[0] != tnx * tny:
            self._log(
                f"Клиппинг: вершин рельефа ({r_verts.shape[0]}) ≠ "
                f"tnx*tny ({tnx*tny}) — пропуск.")
            return None

        # z_high — потолок sealed_solid, гарантированно выше любого z в tonar.
        z_high = float(t_verts[:, 2].max()) + 1.0
        sealed_v, sealed_f = self._build_sealed_solid(
            r_verts, tnx, tny, z_high)
        self._log(
            f"Sealed solid: {sealed_v.shape[0]} верш., {sealed_f.shape[0]} тр. "
            f"(сетка {tnx}×{tny}, z_high={z_high:.2f}).")

        res = self._mesh_boolean(
            t_verts, t_faces, sealed_v, sealed_f, operation="DIFFERENCE")
        if res is None:
            return None
        new_verts, new_faces, engine = res
        vol = self._mesh_volume(new_verts, new_faces)
        self._log(
            f"Boolean DIFFERENCE ({engine}): tonar − sealed = "
            f"{new_verts.shape[0]} верш., {new_faces.shape[0]} тр.; "
            f"объём ≈ {vol:.3f} м³.")
        return new_verts, new_faces, float(vol)

    # ------------------------------------------------------------------
    # Boolean engine dispatch (manifold3d in-process, Blender fallback)
    # ------------------------------------------------------------------
    def _mesh_boolean(self, verts_a, faces_a, verts_b, faces_b,
                      operation="DIFFERENCE"):
        """Run a CSG boolean A∘B and return (verts, faces, engine_name) or
        None. Honours BOOLEAN_ENGINE: "auto" tries the in-process manifold3d
        engine first and only falls back to Blender if it is missing/fails;
        "manifold" / "blender" force one engine. No external process is needed
        for the manifold path, so the volume step works without a Blender
        install."""
        engine = str(getattr(self, "BOOLEAN_ENGINE", "auto")).lower()
        order = {"auto": ("manifold", "blender"),
                 "manifold": ("manifold",),
                 "blender": ("blender",)}.get(engine, ("manifold", "blender"))
        for eng in order:
            if eng == "manifold":
                r = self._manifold_boolean(
                    verts_a, faces_a, verts_b, faces_b, operation)
                if r is not None:
                    return r[0], r[1], "manifold3d"
            elif eng == "blender":
                r = self._blender_boolean(
                    verts_a, faces_a, verts_b, faces_b, operation)
                if r is not None:
                    return r[0], r[1], "blender"
        return None

    def _manifold_boolean(self, verts_a, faces_a, verts_b, faces_b,
                          operation="DIFFERENCE"):
        """In-process CSG boolean via manifold3d. `operation` is
        DIFFERENCE / INTERSECT / UNION. Returns (verts (N,3) float64,
        faces (M,3) int64) in world coords, or None if manifold3d is missing,
        an operand isn't a valid 2-manifold, or the result is empty."""
        try:
            import manifold3d as m3d
        except Exception as exc:
            self._log(f"manifold3d недоступен ({exc}) — пробую Blender.")
            return None

        def _to_manifold(V, F):
            mesh = m3d.Mesh(
                vert_properties=np.asarray(V, dtype=np.float32),
                tri_verts=np.asarray(F, dtype=np.uint32))
            man = m3d.Manifold(mesh)
            return man

        try:
            Ma = _to_manifold(verts_a, faces_a)
            Mb = _to_manifold(verts_b, faces_b)
            # An invalid (non-manifold) operand yields an empty Manifold.
            if Ma.is_empty() or Mb.is_empty():
                self._log("manifold3d: операнд не 2-многообразие "
                          "(empty Manifold) — fallback на Blender.")
                return None
            op = str(operation).upper()
            if op == "DIFFERENCE":
                R = Ma - Mb
            elif op == "INTERSECT":
                R = Ma ^ Mb
            elif op == "UNION":
                R = Ma + Mb
            else:
                self._log(f"manifold3d: неизвестная операция {operation}.")
                return None
            if R.is_empty():
                self._log("manifold3d: результат пуст — fallback на Blender.")
                return None
            msh = R.to_mesh()
            rv = np.asarray(msh.vert_properties, dtype=np.float64)[:, :3]
            rf = np.asarray(msh.tri_verts, dtype=np.int64)
            if rv.shape[0] == 0 or rf.shape[0] == 0:
                return None
            return rv, rf
        except Exception as exc:
            self._log(f"manifold3d boolean упал: {exc} — fallback на Blender.")
            return None

    # ------------------------------------------------------------------
    # Sealed-solid construction
    # ------------------------------------------------------------------
    @staticmethod
    def _build_sealed_solid(relief_verts, tnx, tny, z_high):
        """Build a closed watertight manifold representing the empty
        space ABOVE the rectangular relief grid: floor = the heightfield,
        ceiling = flat plane at z_high, walls = vertical strips around
        the four perimeter edges. Mirrors mesh_reconstruction.cpp:1067-
        1146 (floor + ceiling + horizontal + vertical wall edges, then a
        global winding flip so all normals face OUTWARD).

        Returns (verts (2*tnx*tny, 3) float64, faces (M, 3) int64)."""
        W = int(tnx)
        H = int(tny)
        r = np.asarray(relief_verts, dtype=np.float64).reshape(-1, 3)
        n = W * H

        # Floor (relief) + ceiling (flat at z_high). Floor verts go first
        # so flat_index(iy, ix) == iy * W + ix in both layers (+ n for the
        # ceiling).
        floor_v = r.copy()
        ceil_v = r.copy()
        ceil_v[:, 2] = float(z_high)
        verts = np.concatenate([floor_v, ceil_v], axis=0)

        # Per-cell flat indices for the 4 corners (matches the C++ a/b/c/d
        # naming: a = top-left, b = bottom-left, c = top-right, d = bottom-
        # right of cell (iy, ix)).
        iy = np.arange(H - 1)[:, None]
        ix = np.arange(W - 1)[None, :]
        a = (iy * W + ix).ravel()
        b = ((iy + 1) * W + ix).ravel()
        c = (iy * W + (ix + 1)).ravel()
        d = ((iy + 1) * W + (ix + 1)).ravel()

        # Floor — wound (c, b, a) and (d, b, c). Normal points +Z (INTO
        # the body, which sits above the relief).
        floor_tris = np.concatenate([
            np.stack([c, b, a], axis=1),
            np.stack([d, b, c], axis=1),
        ], axis=0)
        # Ceiling — wound (a, b, c) and (c, b, d) at the upper layer.
        # Normal points −Z (INTO the body, which sits below the ceiling).
        a2, b2, c2, d2 = a + n, b + n, c + n, d + n
        ceil_tris = np.concatenate([
            np.stack([a2, b2, c2], axis=1),
            np.stack([c2, b2, d2], axis=1),
        ], axis=0)

        # Walls — only the 4 outer edges of the rectangle. C++ uses two
        # winding branches depending on which side of the edge holds the
        # body; for a regular rectangle the assignment per edge is fixed.
        wall_chunks = []
        # Top edge (iy = 0): body is on the iy=0 side (south). Branch 2.
        iy0 = 0
        f0 = iy0 * W + np.arange(W - 1)
        f1 = iy0 * W + np.arange(1, W)
        c0 = f0 + n
        c1 = f1 + n
        wall_chunks.append(np.stack([f1, f0, c0], axis=1))
        wall_chunks.append(np.stack([f1, c0, c1], axis=1))
        # Bottom edge (iy = H - 1): body is on the iy=H-2 side. Branch 1.
        iy1 = H - 1
        f0 = iy1 * W + np.arange(W - 1)
        f1 = iy1 * W + np.arange(1, W)
        c0 = f0 + n
        c1 = f1 + n
        wall_chunks.append(np.stack([f0, f1, c1], axis=1))
        wall_chunks.append(np.stack([f0, c1, c0], axis=1))
        # Left edge (ix = 0): body is on the ix=0 side (right). Branch 1
        # in the vertical-edge loop ("right && !left").
        ix0 = 0
        f0 = np.arange(H - 1) * W + ix0
        f1 = np.arange(1, H) * W + ix0
        c0 = f0 + n
        c1 = f1 + n
        wall_chunks.append(np.stack([f0, f1, c1], axis=1))
        wall_chunks.append(np.stack([f0, c1, c0], axis=1))
        # Right edge (ix = W - 1): body is on the ix=W-2 side. Branch 2.
        ix1 = W - 1
        f0 = np.arange(H - 1) * W + ix1
        f1 = np.arange(1, H) * W + ix1
        c0 = f0 + n
        c1 = f1 + n
        wall_chunks.append(np.stack([f1, f0, c0], axis=1))
        wall_chunks.append(np.stack([f1, c0, c1], axis=1))

        walls = np.concatenate(wall_chunks, axis=0)
        faces = np.concatenate(
            [floor_tris, ceil_tris, walls], axis=0).astype(np.int64)
        # Global winding flip: triangles above were built with normals
        # pointing INTO the body (matches the C++ convention); swap v0
        # and v2 → normals now face OUTWARD, which is what Carve expects
        # to treat the mesh as a closed volume.
        faces = faces[:, [2, 1, 0]]
        return verts, faces

    @staticmethod
    def _mesh_volume(verts, faces):
        """Signed volume of a closed triangle mesh by the divergence
        theorem: V = |Σ v0 · (v1 × v2)| / 6. Open / non-manifold meshes
        give a meaningless number — caller is responsible for ensuring
        the mesh is closed."""
        v = np.asarray(verts, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int64)
        if f.shape[0] == 0:
            return 0.0
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        cross_ = np.cross(v1, v2)
        triple = np.einsum("ij,ij->i", v0, cross_)
        return float(abs(triple.sum()) / 6.0)

    # ------------------------------------------------------------------
    # Blender 2.70 headless boolean
    # ------------------------------------------------------------------
    def _blender_boolean(self, verts_a, faces_a, verts_b, faces_b,
                         operation="DIFFERENCE"):
        """Run Blender 2.70's Boolean modifier on mesh A with mesh B as
        the operand. `operation` is forwarded to mod.operation
        ('DIFFERENCE' / 'INTERSECT' / 'UNION'). Returns (verts (N,3)
        float64, faces (M,3) int64) of the result in world coords, or
        None on any failure (Blender missing, non-zero exit, empty
        result)."""
        import subprocess
        import tempfile
        import shutil

        exe = self.BLENDER_EXE
        if not exe or not os.path.exists(exe):
            self._log(f"Blender 2.70 не найден ({exe}) — клиппинг пропущен.")
            return None
        if verts_a.shape[0] == 0 or faces_a.shape[0] == 0:
            return None
        if verts_b.shape[0] == 0 or faces_b.shape[0] == 0:
            return None

        op = str(operation).upper()
        if op not in ("DIFFERENCE", "INTERSECT", "UNION"):
            self._log(f"Неизвестная булева операция: {operation}")
            return None

        # Persistent debug dump under the project root so the user (or
        # we, later) can open the inputs / output in Blender for
        # inspection. Overwritten on every call.
        debug_dir = self._boolean_debug_dir()

        tmp = tempfile.mkdtemp(prefix="depth_recon_bool_")
        keep_tmp = False
        try:
            mesh_a = os.path.join(tmp, "mesh_a.obj")
            mesh_b = os.path.join(tmp, "mesh_b.obj")
            result = os.path.join(tmp, "result.obj")
            script = os.path.join(tmp, "boolean.py")
            self._write_obj(mesh_a, verts_a, faces_a)
            self._write_obj(mesh_b, verts_b, faces_b)
            self._write_blender_boolean_script(script)
            # Mirror the inputs into the debug folder *before* we run
            # Blender, so we still have them if Blender hangs / crashes.
            self._copy_to_debug(mesh_a, debug_dir, "mesh_a.obj")
            self._copy_to_debug(mesh_b, debug_dir, "mesh_b.obj")
            self._copy_to_debug(script, debug_dir, "boolean.py")
            cmd = [exe, "-b", "-P", script, "--",
                   mesh_a, mesh_b, result, op]
            self._log(f"Blender boolean {op}: запуск "
                      f"'{os.path.basename(exe)}' "
                      f"(A: {verts_a.shape[0]}v/{faces_a.shape[0]}f, "
                      f"B: {verts_b.shape[0]}v/{faces_b.shape[0]}f)…")
            self._log(f"Debug-дамп входов: {debug_dir}")
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=float(self.BLENDER_TIMEOUT_S))
            except subprocess.TimeoutExpired:
                self._log(f"Blender boolean: таймаут "
                          f"{self.BLENDER_TIMEOUT_S}s — клиппинг пропущен.")
                keep_tmp = True
                return None
            except Exception as exc:
                self._log(f"Blender запуск упал: {exc}")
                return None
            # Surface the [bool] diagnostics from the embedded script.
            bool_lines = [ln for ln in (proc.stdout or "").splitlines()
                          if ln.startswith("[bool]")]
            for ln in bool_lines:
                self._log(ln)
            # Save the full Blender stdout/stderr alongside the OBJ
            # dumps regardless of outcome.
            try:
                with open(os.path.join(debug_dir, "blender_stdout.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(proc.stdout or "")
                with open(os.path.join(debug_dir, "blender_stderr.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(proc.stderr or "")
            except Exception:
                pass
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip()[-600:]
                self._log(f"Blender вернул код {proc.returncode}; "
                          f"temp сохранён: {tmp}\n…{tail}")
                keep_tmp = True
                return None
            if not os.path.exists(result):
                self._log(f"Blender завершился, но result.obj не создан. "
                          f"temp: {tmp}")
                keep_tmp = True
                return None
            # Copy the result into the debug folder too.
            self._copy_to_debug(result, debug_dir, "result.obj")
            rv, rf = self._read_obj(result)
            if rv is None or rv.shape[0] == 0 or rf.shape[0] == 0:
                self._log(f"Результат Blender boolean пуст. temp: {tmp}")
                keep_tmp = True
                return None
            return rv, rf
        finally:
            if not keep_tmp:
                try:
                    shutil.rmtree(tmp, ignore_errors=True)
                except Exception:
                    pass

    def _boolean_debug_dir(self):
        """Return (and create) a stable project-relative directory where
        the latest Blender-boolean inputs and outputs are dumped. Path:
        <project_root>/debug_boolean/. Falls back to a temp folder if
        the project root is unwritable."""
        here = os.path.dirname(os.path.abspath(__file__))
        proj_root = os.path.dirname(os.path.dirname(here))
        d = os.path.join(proj_root, "debug_boolean")
        try:
            os.makedirs(d, exist_ok=True)
            return d
        except Exception:
            import tempfile
            return tempfile.mkdtemp(prefix="depth_recon_dbg_")

    @staticmethod
    def _copy_to_debug(src, dst_dir, dst_name):
        try:
            import shutil
            shutil.copyfile(src, os.path.join(dst_dir, dst_name))
        except Exception:
            pass

    @staticmethod
    def _write_obj(path, verts, faces):
        """Minimal Wavefront OBJ writer (v + f only, 1-indexed)."""
        with open(path, "w", encoding="utf-8") as f:
            for v in verts:
                f.write(f"v {float(v[0])} {float(v[1])} {float(v[2])}\n")
            for tri in faces:
                f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} "
                        f"{int(tri[2]) + 1}\n")

    @staticmethod
    def _read_obj(path):
        """Read a Wavefront OBJ (v + f) and triangulate any n-gons by
        fanning from vertex 0. Returns (verts (N,3) float64,
        faces (M,3) int64) or (None, None) if the file has no vertices."""
        verts = []
        faces = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        verts.append((float(parts[1]), float(parts[2]),
                                      float(parts[3])))
                elif line.startswith("f "):
                    idxs = []
                    ok = True
                    for tok in line.split()[1:]:
                        slash = tok.find("/")
                        s = tok if slash < 0 else tok[:slash]
                        if not s:
                            continue
                        try:
                            idxs.append(int(s) - 1)
                        except ValueError:
                            ok = False
                            break
                    if ok and len(idxs) >= 3:
                        for i in range(1, len(idxs) - 1):
                            faces.append((idxs[0], idxs[i], idxs[i + 1]))
        if not verts:
            return None, None
        return (np.asarray(verts, dtype=np.float64),
                np.asarray(faces, dtype=np.int64))

    @staticmethod
    def _write_blender_boolean_script(path):
        """Headless Blender 2.70 script that mirrors the server's
        boolean_operations.cpp pattern, with diagnostics around Carve:
        select mesh_a explicitly before applying the modifier, check the
        return code of modifier_apply, log vert/face counts before and
        after the apply, and exit non-zero if Carve silently produced
        the same mesh as the input (which happens when the solver bails
        out on a non-manifold or self-intersecting operand)."""
        src = (
            "import bpy\n"
            "import sys\n"
            "\n"
            'argv = sys.argv[sys.argv.index("--") + 1:]\n'
            "mesh_a, mesh_b, out_file = argv[0], argv[1], argv[2]\n"
            "op = argv[3] if len(argv) > 3 else 'DIFFERENCE'\n"
            "print('[bool] op=' + op)\n"
            "\n"
            "bpy.ops.object.select_all(action='SELECT')\n"
            "bpy.ops.object.delete()\n"
            "\n"
            "bpy.ops.import_scene.obj(filepath=mesh_a)\n"
            "ob_a = bpy.context.selected_objects[0]\n"
            "ob_a.name = 'MeshA'\n"
            "bpy.ops.object.select_all(action='DESELECT')\n"
            "\n"
            "bpy.ops.import_scene.obj(filepath=mesh_b)\n"
            "ob_b = bpy.context.selected_objects[0]\n"
            "ob_b.name = 'MeshB'\n"
            "bpy.ops.object.select_all(action='DESELECT')\n"
            "\n"
            "va0 = len(ob_a.data.vertices); fa0 = len(ob_a.data.polygons)\n"
            "vb0 = len(ob_b.data.vertices); fb0 = len(ob_b.data.polygons)\n"
            "print('[bool] A in: {}v/{}f  B in: {}v/{}f'"
            ".format(va0, fa0, vb0, fb0))\n"
            "def _bbox(o):\n"
            "    bb = [o.matrix_world * v.co for v in o.data.vertices]\n"
            "    if not bb: return None\n"
            "    xs=[p.x for p in bb]; ys=[p.y for p in bb]; "
            "zs=[p.z for p in bb]\n"
            "    return (min(xs), min(ys), min(zs),\n"
            "            max(xs), max(ys), max(zs))\n"
            "print('[bool] A bbox:', _bbox(ob_a))\n"
            "print('[bool] B bbox:', _bbox(ob_b))\n"
            "\n"
            "mod = ob_a.modifiers.new(name='Boolean', type='BOOLEAN')\n"
            "mod.operation = op\n"
            "mod.object = ob_b\n"
            "bpy.context.scene.objects.active = ob_a\n"
            "ob_a.select = True\n"
            "res = bpy.ops.object.modifier_apply(modifier=mod.name)\n"
            "print('[bool] modifier_apply -> ' + repr(res))\n"
            "\n"
            "va1 = len(ob_a.data.vertices); fa1 = len(ob_a.data.polygons)\n"
            "print('[bool] A out: {}v/{}f'.format(va1, fa1))\n"
            "\n"
            "if va1 == va0 and fa1 == fa0:\n"
            "    print('[bool] ERROR: Carve made no change. modifier_apply\\n'\n"
            "          '  returned without modifying ob_a.data — Carve\\n'\n"
            "          '  likely bailed on a non-manifold / self-intersect\\n'\n"
            "          '  operand. modifier_apply result was: ' + repr(res))\n"
            "    sys.exit(2)\n"
            "\n"
            "bpy.ops.object.select_all(action='DESELECT')\n"
            "ob_a.select = True\n"
            "bpy.context.scene.objects.active = ob_a\n"
            "bpy.ops.export_scene.obj(filepath=out_file, use_selection=True)\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)

    @staticmethod
    def _unproject(lens, cam_np, render, fx, fy, Z):
        """Unproject a film point at forward distance Z into world coords."""
        near = Point3()
        far = Point3()
        lens.extrude(Point2(fx, fy), near, far)
        d = far - near
        ln = d.length()
        if ln > 1e-9:
            d /= ln
        if abs(d.y) < 1e-6:
            d.y = 1e-6 if d.y >= 0 else -1e-6
        cam_pt = d * (Z / d.y)
        return render.getRelativePoint(cam_np, cam_pt)

    def _apply_material(self, node) -> None:
        try:
            # PERFORMANCE preset (simplepbr): recompute upward normals + apply
            # a diffuse tan material so the depth-fill mesh is lit by the sun
            # (RP's green-emission convention renders it flat/black here).
            if not self.panda_app.use_render_pipeline:
                self.panda_app.relight_generated_mesh(
                    node, base_color=(0.62, 0.55, 0.47, 1.0))
                return
            mat = Material()
            mat.set_base_color((0.62, 0.55, 0.47, 1.0))
            mat.set_emission((0, 1, 0, 0))   # RP: G = normal strength
            node.set_material(mat, 1)
            node.set_two_sided(True)
        except Exception as exc:
            self._log(f"material apply failed: {exc}")

    # ==================================================================
    # Callbacks / logging
    # ==================================================================
    def _emit_count(self) -> None:
        if callable(self.on_count):
            try:
                self.on_count(len(self._films))
            except Exception:
                pass

    def _finish(self, success: bool, info: dict) -> None:
        if callable(self.on_finished):
            try:
                self.on_finished(success, info)
            except Exception:
                pass

    def _emit_picking_state(self, active: bool) -> None:
        if callable(self.on_picking_state):
            try:
                self.on_picking_state(bool(active))
            except Exception:
                pass

    def _log(self, msg: str) -> None:
        print(f"[DepthRecon] {msg}")
        if callable(self.on_log):
            try:
                self.on_log(msg)
            except Exception:
                pass
