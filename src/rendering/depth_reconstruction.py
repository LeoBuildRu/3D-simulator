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

# Stage 3 trend extraction & extrapolation: Gaussian-smooth для тренда
# (через scipy.ndimage) и бигармоническая экстраполяция тренда наружу
# (через skimage). Оба — мягкие зависимости: если их нет, мы откатываемся
# к box-smooth + Jacobi-Laplace соответственно (см. _mask_aware_gaussian
# и _biharmonic_extrapolate).
try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
except ImportError:                       # pragma: no cover
    _scipy_gaussian_filter = None
try:
    from skimage.restoration import inpaint_biharmonic as _skimage_biharmonic
except ImportError:                       # pragma: no cover
    _skimage_biharmonic = None
try:
    from scipy.spatial import Delaunay as _scipy_delaunay
except ImportError:                       # pragma: no cover
    _scipy_delaunay = None

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
    # 0 = срезаем только ту самую ячейку, без пропагации drop'ов на соседей,
    # иначе края рельефа «отъедаются» лишним рядом ячеек и появляются дыры.
    LONGPOLY_PROPAGATE_CELLS = 0
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
    # 80° совпадает с EXTRAP_MAX_ANGLE_DEG — финальная очистка пропускает
    # естественные крутые descent-стенки, но рубит ярко-вертикальные
    # артефакты (борта кузова, окклюзионные стенки).
    STEEP_MAX_ANGLE_DEG = 80.0
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
    # Stage 3 — pattern-based extrapolation. After Stage 2 the relief is
    # bounded to the truck body (point-in-mesh clip by napolnitel); Stage 3
    # extends it outward into a fixed XY rectangle (TARGET_SIZE_M, centred on
    # the napolnitel XY centre) by tile-based pattern synthesis on the
    # high-frequency residual, with a smooth blend of the low-frequency trend
    # toward floor_z over EXTRAP_BLEND_RADIUS_M. The result is a single
    # contiguous heightfield mesh that preserves the visible relief's
    # characteristic bumps / ridges on the extrapolated sides.
    # See _pattern_extrapolate / _quilt_residual.
    ENABLE_EXTRAPOLATION = True
    # Boolean DIFFERENCE (sealed-solid volume) DEACTIVATED but the code
    # — _clip_relief_to_target and friends — is preserved in this file
    # for future use.
    ENABLE_VOLUME = False
    # ==================================================================
    # Per-snapshot fill mask. The masked depth is a sibling of the full depth:
    #   <name>_depth.png       — full depth map (load + body + background),
    #                            used for anchor-point calibration.
    #   <name>_depth_mask.png  — RGBA copy with alpha=0 outside the load (made
    #                            by hand in Photoshop). The alpha channel is
    #                            the inside-mask used for grid sampling; the
    #                            RGB channel carries the depth values for the
    #                            relief grid.
    # When USE_MASKS=True and the mask file exists, anchors sample the full
    # depth while the relief grid samples the masked depth — vertices climbing
    # the body walls are excluded at the source instead of being 3D-clipped.
    # Without the mask file the old single-depth path is used.
    USE_MASKS = True
    MASK_SUFFIX = "_mask.png"
    MASK_ALPHA_MIN = 0.5
    # The mesh-cell rule keeps a cell only when all 4 of its corners are in the
    # mask, which erodes the kept region inward by ~1 grid cell along the whole
    # boundary (plus the grid-node quantization of the edge). Dilate the
    # sampled mask by this many cells first so the surface reaches the true
    # mask edge instead of being trimmed short. 0 = off.
    MASK_DILATE_CELLS = 1
    # --- Mask-edge depth repair (rim de-spiking) ----------------------
    # The masked depth is unreliable in a thin ring just inside the alpha
    # edge: the hand-painted mask inevitably catches a sliver of the body
    # wall or the depth map's soft (anti-aliased) silhouette, so the rim
    # nodes carry depth that doesn't belong to the pile. Left alone they
    # show up as spurious peaks / rises along the border. A plain pixel
    # offset (eroding the mask) is NOT acceptable: on snapshots where the
    # pile crest sits right on the mask edge it would shave off real top
    # vertices.
    #
    # Instead we REPLACE each rim node's depth by a gradient-preserving
    # extrapolation of the trusted interior. For every node within
    # EDGE_EXTRAP_BAND_CELLS of the boundary we fit a local first-order
    # (moving-least-squares) plane z ≈ a·dx + b·dy + c to the CORE nodes
    # around it (those deeper than the band), measuring (dx,dy) from the
    # rim node itself — so the extrapolated value is just c, the interior
    # surface's own slope carried out to the rim. A node on a genuine steep
    # crest keeps climbing at the interior's rate (the crest is preserved);
    # a node on a phantom rim spike is pulled back onto the interior trend.
    # This is the slope-aware extrapolation the naive "average the
    # neighbours' height" approach can't do.
    #
    # EDGE_EXTRAP_BAND_CELLS — width (grid cells) of the rim ring to
    #   re-estimate. Must be ≥ MASK_DILATE_CELLS so the dilated outer ring
    #   (which samples OUTSIDE the original mask, the most contaminated) is
    #   always covered. Bigger = more of the border is rebuilt from the
    #   interior trend (safer against wide contamination, but trusts less
    #   of the measured edge).
    # EDGE_EXTRAP_WINDOW_CELLS — radius (grid cells) of the core
    #   neighbourhood each rim node fits its plane to. Should comfortably
    #   exceed the band so every rim node sees enough core samples; larger
    #   = smoother / more global slope estimate, smaller = more local.
    # EDGE_EXTRAP_IDW_POWER — inverse-distance weighting exponent for the
    #   plane fit; higher keeps the fit tighter to the nearest core nodes.
    # EDGE_EXTRAP_DISC_REACH — beyond the fixed rim band, the repair also
    #   absorbs depth-DISCONTINUITY nodes (the LONGPOLY relative+absolute
    #   criterion, evaluated on the grid depths BEFORE smoothing) that are
    #   connected to the rim, up to this many cells deep. This catches a hard
    #   spike that pokes inward FURTHER than `band` — without it the spike's
    #   inner part is treated as trusted "core", so it both survives and
    #   poisons the fit (the "long peaks the algorithm seemed to skip" case).
    #   An interior crater wall, not connected to the rim, is left intact.
    #   0 disables spike absorption (fixed rim band only). The thresholds are
    #   shared with LONGPOLY_MAX_EDGE_* so this stage and the final cutoff
    #   agree on what a discontinuity is.
    # EDGE_EXTRAP_CLAMP_SLACK — anti-dip guard (LOWER bound only). A first-order
    #   extrapolation of a surface descending toward the rim overshoots
    #   downward, leaving a faint bend where an up-spike used to be; the result
    #   is floored at the core window's min minus this fraction of the window
    #   range. The upper side is left free so a real crest rising onto the edge
    #   is preserved (upward spikes are removed by the absorption, not here).
    #   Lower = firmer floor (less dip); None disables the guard.
    # Set EDGE_EXTRAP=False to disable and keep the raw masked rim depth.
    EDGE_EXTRAP = True
    EDGE_EXTRAP_BAND_CELLS = 3
    EDGE_EXTRAP_WINDOW_CELLS = 8
    EDGE_EXTRAP_IDW_POWER = 2.0
    EDGE_EXTRAP_DISC_REACH = 8
    EDGE_EXTRAP_CLAMP_SLACK = 0.5
    # Extrapolation: after the masked relief is built, extend it to a fixed
    # axis-aligned rectangle in world XY (metres) by mirror-tiling the
    # pattern outside the mask's XY bbox. The trend (plane fit) is added
    # back so the seam stays C0-continuous. Set to None to disable and
    # keep the original mask-only mesh.
    # XY-прямоугольник (метры) куда экстраполируется рельеф (Stage 3).
    # Центрируется на XY-центре napolnitel (см. _napolnitel_xy_center);
    # 4×7 м заведомо больше MAZ (≈ 2.14×5.10), так что мы видим
    # «продолжение» рельефа во всех 4 направлениях вокруг кузова.
    TARGET_SIZE_M = (4.0, 7.0)
    TARGET_RES_PER_M = 35
    # ---- Trend + residual extrapolation -----------------------------
    # Z раскладывается на ДВА слоя и каждый продолжается своим методом:
    #
    #   Z(x, y)  =  TREND(x, y)  +  RESIDUAL(x, y)
    #               глобальная       мелкие бугры,
    #               форма насыпи     гранулярность
    #
    # TREND — Gaussian-сглаженный source с радиусом TREND_SMOOTH_SIGMA_M.
    #   Снаружи source'а продолжается БИГАРМОНИЧЕСКИ (∇⁴Z = 0 с Dirichlet
    #   BC на границе) — это формально-правильный способ «продолжать
    #   рельеф по тем же наклонам». Для плоского source'а это даёт
    #   плоскость; для насыпи — мягко спадающее продолжение насыпи.
    #
    # RESIDUAL — source minus trend. После вычитания тренда это
    #   STATIONARY-поле (mean≈0, дисперсия одинакова), на нём
    #   image-quilting (тот же _quilt_z) работает корректно — больше нет
    #   «высоких» и «низких» патчей, есть только мелкие отклонения,
    #   которые равномерно покрывают всю ext-область шероховатостью.
    #
    # TREND_SMOOTH_SIGMA_M — σ Гауссова сглаживания тренда в МЕТРАХ.
    #   Должно быть больше типичного «бугра» (чтобы тот ушёл в residual)
    #   и меньше масштаба самой насыпи (чтобы тренд её захватил).
    #   0.5 м (= 17 ячеек при TARGET_RES_PER_M=35) — баланс для гравия.
    TREND_SMOOTH_SIGMA_M = 0.5
    # EXTRAP_BIHARMONIC — основной переключатель экстраполяции тренда.
    #   True → бигармоник через skimage (если установлен); если skimage
    #   нет, автоматически откатываемся в итеративный Laplace-solver
    #   (∇²Z = 0). False → сразу Laplace (быстрее, но «плоский» rsult
    #   снаружи — наклоны не продолжаются).
    EXTRAP_BIHARMONIC = True
    # EXTRAP_DECAY_SCALE_M — длина (в метрах), на которой тренд снаружи
    #   anchor'а АДАПТИВНО опускается от Z_boundary к floor_eff
    #   (= min(trend) на anchor). Это и есть «насыпь должна спадать на
    #   другой стороне»:
    #     • в anchor-ячейках высокий Z_b (граница на пике) → cap быстро
    #       спускает наружу = насыпь опускается;
    #     • Z_b близко к floor_eff (плоский кейс) → rate≈0, плоско
    #       остаётся плоско.
    #   ВАЖНО: cap применяется ТОЛЬКО как верхняя граница на тренд
    #   снаружи anchor'а (внутри source НИЧЕГО не модифицируется). Если
    #   биграмоник и так даёт меньше — он остаётся. Это «один из факторов»,
    #   как просил пользователь, а не жёсткий тренд к плоскости.
    EXTRAP_DECAY_SCALE_M = 1.5
    # LAPLACE_MAX_ITER / LAPLACE_TOL — параметры фоллбэка-итерации.
    LAPLACE_MAX_ITER = 4000
    LAPLACE_TOL = 1e-4
    # PATTERN_* — параметры image-quilting'а residual'а. См. _quilt_z.
    PATTERN_PATCH_FRAC = 0.40
    PATTERN_OVERLAP_FRAC = 0.30
    PATTERN_CANDIDATES = 24
    PATTERN_SEED = 0
    # Финальное сглаживание ТОЛЬКО синтезированных ячеек (anchor
    # сохраняется), чтобы убрать остатки видимых швов между патчами
    # residual'а. На trend этот шаг не нужен (он и так гладкий).
    PATTERN_FINAL_SMOOTH = 1
    # Light smoothing of the target grid AFTER extrapolation (mask data is
    # preserved — only extrapolated cells are smoothed). 3 итерации —
    # компромисс: разрывы вдоль кромки закрываются, но рельеф не
    # «размывается» лишним усреднением.
    TARGET_SMOOTH_ITERS = 3
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
    # УСТАРЕЛО: в новой архитектуре экстраполяции не используется
    # (заменено на линейный blend к floor_z через EXTRAP_BLEND_RADIUS_M).
    # Параметр оставлен для обратной совместимости с info-словарём.
    EXTRAP_ANGLE_DEG = 50.0
    EXTRAP_MAX_ANGLE_DEG = 80.0
    # Радиус (в плоскости XY, метрах) плавного перехода Z от граничного
    # seed-значения к floor_z. На границе маски — seed_z; за blend_radius
    # — floor_z плато. Большое значение = более растянутый плавный
    # спуск; малое = резкий обрыв. 0.6 м — комфортный дефолт.
    EXTRAP_BLEND_RADIUS_M = 0.6
    # Inner-ring lift: после построения ext-heightfield'а каждый inner-ring
    # узел (1-cell ring снаружи anchor'а) подтягивается через IDW по
    # k=4 ближайшим src_bnd-вершинам к НАСТОЯЩЕМУ Z source-меша. Без этого
    # bilinear-of-rasterized теряет high-freq на пиках (несколько source-
    # вершин в одной растер-ячейке усредняются), и glue-треугольники
    # между src-меш-вершиной (≈peak) и inner-ring (≈mean) растягиваются
    # вертикально на ≈|residual| — это «зубцы» на back-of-hill швах.
    # Коррекция δ распространяется внутрь ext'а с exp(-d/scale_m) feather'ом
    # — без этого по линии inner-ring была бы 1-клеточная «стенка».
    # 0.3 м (~10 ячеек) — баланс: шов гладкий, ext быстро возвращается к
    # бигармонику+cap. Больше = плавнее, но коррекция тянется дальше внутрь.
    EXTRAP_RIM_LIFT_SCALE_M = 0.3
    # Glue-фильтр (_build_delaunay_glue): максимальная длина СТЫКОВОЧНОГО
    # (src→inner) ребра bridge-треугольника в единицах cell_size_m.
    # cell_size_m ≈ 1/TARGET_RES_PER_M ≈ 2.9 см; factor=10.0 → порог ≈ 29 см.
    # Раньше было 2.5 и фильтр стоял на «max ребре в треугольнике» — этот
    # старый вариант отбрасывал валидные bridge'ы с длинным src-src ребром
    # (где source-граница редкая), оставляя «дырки в виде треугольников»
    # вдоль шва. Новый фильтр игнорирует src-src/inner-inner длину
    # (нерелевантно для шва) и режет только реально длинные стыковочные
    # рёбра. После rim-lift'а ext-поверхность гладко переходит в source-Z,
    # поэтому даже 29-сантиметровый мост ложится плашмя — артефакта тяги
    # нет, лишь снижается «локальное разрешение». Если orphan-узлы всё
    # равно остаются — поднимай ещё (20+), если, наоборот, появляются
    # «паутинные» мосты через пустоту — снижай к 5.0.
    EXTRAP_GLUE_MAX_EDGE_FACTOR = 10.0
    # Densification source-границы перед Delaunay-glue: для каждого граничного
    # ребра source-меша длиннее spacing вставляются промежуточные вершины
    # с линейно-интерполированными XYZ. Это закрывает «провалы» там, где
    # соседние src_bnd-вершины стоят в десятках сантиметров (typically около
    # napolnitel-clip кромок) и без densification ~30-40% inner-ring узлов
    # оставались без bridge'а. spacing = cell_size_m × factor:
    #   1.0 → каждое ребро не длиннее одной ext-ячейки (~2.9 см) — самый плотный
    #         вариант, гарантированно убирает orphan inner-ring;
    #   2.0 → раз в две ячейки, экономнее (меньше extras), но на участках
    #         длиной >5.8 см orphan'ы возможны;
    #   0.5 → ультра-плотно, для тяжёлых случаев. Промежуточные вершины
    #         ДОБАВЛЯЮТСЯ в combined mesh; source-меш всё равно сохраняется
    #         байт-в-байт (extras лежат СТРОГО на исходных граничных рёбрах,
    #         t-junction в 3D совпадает — без видимых щелей).
    EXTRAP_DENSIFY_SPACING_FACTOR = 1.0
    # Buffer-зона (seam hole stitching). После concat source+ext+glue
    # combined-меш ещё может содержать остаточные ДЫРЫ вдоль шва:
    # orphan inner-ring узлы без bridge'а (glue-фильтр отрезал слишком
    # длинные стыковочные рёбра), несклеенные densify-extras и т.п. По
    # логам _build_delaunay_glue это видно как «inner-ring без bridge'а:
    # X/Y». Перепады высот вдоль шва уже задавлены rim-lift'ом — но
    # дыры остаются топологическими (видны как треугольные «окна» в
    # меше). Этот пост-pass находит ВСЕ замкнутые boundary-loop'ы в
    # combined-меше, оставляет ОДИН внешний (= TARGET-прямоугольник,
    # самый большой XY bbox) и каждый внутренний триангулирует buffer-
    # треугольниками (constrained Delaunay + point-in-polygon filter
    # для невыпуклых дыр; fallback на fan для деградации). Аддитивно:
    # source/ext/glue не трогаются, дыры просто закрываются сверху.
    SEAM_HOLE_STITCH = True
    # Unused — extrapolation is no longer in the pipeline.
    EXTRAP_FLOOR_FROM_MIN_M = 0.05
    # Optional absolute Z floor (metres). Takes effect when
    # EXTRAP_FLOOR_FROM_MIN_M is None — useful when you know the truck
    # floor in world Z.
    EXTRAP_FLOOR_M = None
    # Residual texture preservation: the visible relief's high-frequency
    # variation (original Z minus its smoothed trend) is mirror-tiled onto
    # the extrapolated area so the hidden back side keeps the look of the
    # visible front, instead of being a featureless smooth slope. Weight
    # 0 = pure descent (smooth), 1 = full mirror-residual amplitude.
    #
    # 0.0 — mirror-tiling выключен: невидимые зоны заполняются плавным
    # спуском по углу repose. Иначе при «торчащем» пике в дальней зоне он
    # отзеркаливается в ближнюю и создаёт фантомный «бугор» у борта,
    # которого на сцене нет.
    EXTRAP_TEXTURE_WEIGHT = 0.0
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
        self._meta: dict = {}

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
    def set_source(self, depth_path: str, color_path: str = "",
                   meta: dict | None = None) -> None:
        """Point the reconstructor at the active stand snapshot.

        `meta` (опционально) — содержимое серверного meta.json, который
        пришёл вместе с depth-картой (например, {"type":"depth","model":"MAZ"}).
        Сейчас он только запоминается (self._meta) для дальнейшей логики
        (выбор модели и т.п.), но не меняет геометрию реконструкции."""
        self._depth_path = depth_path or ""
        self._color_path = color_path or ""
        self._meta = dict(meta) if isinstance(meta, dict) else {}

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

        depth_full = self._load_depth_norm()
        if depth_full is None:
            self._finish(False, {})
            return False
        H, W = depth_full.shape

        # Companion masked-depth (<depth>_mask.png) if it exists. Anchors are
        # ALWAYS sampled from the full depth (they can land anywhere on the
        # truck), but the relief grid samples this masked map and uses its
        # alpha as the in-region mask — so vertices outside the load (e.g.
        # climbing the body walls) are excluded at the source.
        if self.USE_MASKS:
            masked = self._load_masked_depth(W, H)
        else:
            masked = None
        if masked is not None:
            depth_grid, mask_alpha = masked
            self._log(
                f"Маска наполнения найдена ({os.path.basename(self._mask_path())}): "
                f"калибровка по полной глубине, рельеф — по masked-depth.")
        else:
            depth_grid, mask_alpha = depth_full, None
            if self.USE_MASKS:
                self._log(
                    "Маска наполнения не найдена — обе стадии работают по "
                    "полной карте глубины.")

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
        # Anchors sample the FULL depth — they can land on the body, not the
        # load, where the masked-depth file has alpha=0.
        cfx = np.asarray([f[0] for f in self._films[:n_pts]], dtype=np.float64)
        cfy = np.asarray([f[1] for f in self._films[:n_pts]], dtype=np.float64)
        d_ctrl = np.empty(n_pts, dtype=np.float64)
        z_ctrl = np.empty(n_pts, dtype=np.float64)
        for i in range(n_pts):
            col, row = self._film_to_pixel(cfx[i], cfy[i], W, H, fyh)
            d_ctrl[i] = float(depth_full[row, col])
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

        # --- Region: defined by the alpha channel of the masked depth, NOT by
        # the picked points. The picked points are used ONLY for calibration.
        #   masked-depth present -> the mesh covers exactly the alpha>min
        #                           pixels, at full grid resolution over the
        #                           mask's bounding box;
        #   no masked-depth      -> the whole frame is reconstructed, with the
        #                           flood-fill background dropped.
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
        # Grid sampling uses the MASKED depth so manual Photoshop touchups
        # inside the load region are honoured; anchors above used the full
        # depth instead.
        cols, rows = self._film_to_pixel_vec(FX, FY, W, H, fyh)
        d_grid = depth_grid[rows, cols]
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
            self._log(f"Маска наполнения: регион = {n_mask} узлов "
                      f"(alpha > {self.MASK_ALPHA_MIN:.0%}, +"
                      f"{self.MASK_DILATE_CELLS} ячейка к краю).")
        else:
            inside = np.ones(FX.shape[0], dtype=bool)
            # Flood-fill background removal: a large gray-ish region flooded
            # from the image border (typically the sky/wall above the truck)
            # is dropped before reconstruction. Grid nodes that sample a
            # background pixel are marked out-of-region.
            bg = self._detect_background(depth_full)
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

        # Mask-edge depth repair: re-estimate the thin rim ring just inside
        # the alpha boundary from the interior's slope, so border spikes /
        # rises that don't belong to the pile are pulled onto the surface
        # trend WITHOUT eroding the mask (which would shave off a real crest
        # sitting on the edge). Mask path only — the flood-fill path has no
        # curated alpha edge to protect.
        if (use_mask and self.EDGE_EXTRAP
                and self.EDGE_EXTRAP_BAND_CELLS > 0):
            Z2d = Z_grid.reshape(stride, stride)
            in2d = inside.reshape(stride, stride)
            # Discontinuity thresholds shared with the final LONGPOLY cutoff so
            # the rim repair and the mesh-level cut agree on what a spike is.
            disc_ratio = (self.LONGPOLY_MAX_EDGE_RATIO
                          if self.LONGPOLY_CUTOFF else None)
            disc_abs_m = (self.LONGPOLY_MAX_EDGE_M
                          if self.LONGPOLY_CUTOFF else None)
            Z2d, n_edge = self._extrapolate_edge_z(
                Z2d, in2d,
                self.EDGE_EXTRAP_BAND_CELLS,
                self.EDGE_EXTRAP_WINDOW_CELLS,
                self.EDGE_EXTRAP_IDW_POWER,
                disc_ratio=disc_ratio,
                disc_abs_m=disc_abs_m,
                disc_reach=self.EDGE_EXTRAP_DISC_REACH,
                clamp_slack=self.EDGE_EXTRAP_CLAMP_SLACK)
            Z_grid = Z2d.ravel()
            self._log(
                f"Ремонт кромки: пересчитано {n_edge} приграничных узлов "
                f"(полоса {self.EDGE_EXTRAP_BAND_CELLS} яч. + поглощение пиков "
                f"до {self.EDGE_EXTRAP_DISC_REACH} яч.) экстраполяцией наклона "
                f"внутренней области.")

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

        # ---- Stage 3: pattern-based extrapolation. Source-меш (Stage 1+2)
        #      идёт в результат БЕЗ ИЗМЕНЕНИЙ; снаружи его XY-bbox'а
        #      достраивается «обрамление» (heightfield на регулярной
        #      4×7 м-сетке), заполненное image-quilting'ом из реальных
        #      патчей source'а — паттерн продолжается наружу без правки
        #      основного рельефа. Если extrapolation сработал, он сам
        #      применяет cleanup только к ext-части и просит пропустить
        #      глобальный cleanup (skip_global_cleanup), чтобы source
        #      не пострадал.
        skip_global_cleanup = False
        if self.ENABLE_EXTRAPOLATION and len(faces) > 0:
            res = self._pattern_extrapolate(verts, faces)
            if res is not None:
                verts, faces, skip_global_cleanup = res
            else:
                self._log("Stage 3: pattern-экстраполяция не дала результата — "
                          "оставляю меш по маске.")

        if len(faces) == 0:
            self._log("Нет ячеек для меша (Stage 3 вернул пустой меш).")
            self._finish(False, {})
            return False

        # ---- Final cleanup: strip near-vertical walls (with a metric removal
        #      radius) and tiny disconnected clusters from the displayed mesh.
        #      ВНИМАНИЕ: пропускаем, когда сработал pattern-extrapolation —
        #      source-меш должен дойти до экрана нетронутым.
        if (not skip_global_cleanup
                and (self.STEEP_CUTOFF or self.MIN_CLUSTER_SIZE_M)
                and len(faces) > 0):
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
        """Load the depth map as an HxW float array normalised to [0, 1].

        Supports:
          * 8-bit PNG (legacy stand snapshots) — /255.
          * 16-bit PNG (AnyDepth output, mode 'I' / 'I;16') — /65535.
          * .npy float32 — robust [1, 99] percentile normalisation.
        """
        path = self._depth_path or ""
        if not path:
            return None
        try:
            if path.lower().endswith(".npy"):
                arr = np.load(path).astype(np.float32)
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return None
                lo, hi = np.percentile(finite, [1.0, 99.0])
                if hi - lo < 1e-8:
                    hi = lo + 1e-8
                return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

            from PIL import Image
            im = Image.open(path)
            mode = im.mode
            arr = np.asarray(im)
            if mode in ("I", "I;16", "I;16B", "I;16L"):
                # 16-bit / 32-bit grayscale (AnyDepth кладёт I;16 — 0..65535).
                return arr.astype(np.float32) / 65535.0
            if arr.dtype == np.uint16:
                return arr.astype(np.float32) / 65535.0
            if arr.ndim == 3:
                # RGB/RGBA — берём только luminance.
                arr = np.asarray(im.convert("L"))
            return arr.astype(np.float32) / 255.0
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
        loaded = self._load_masked_depth(W, H)
        if loaded is None:
            return None
        _, alpha = loaded
        return alpha

    def _load_masked_depth(self, W, H):
        """Load `<depth>_mask.png` as a pair (depth_rgb, alpha), each HxW
        float32 in [0, 1] resized to the depth size if needed:
          • depth_rgb — RGB luminance, the manually-curated depth values
                        inside the load region (used for the relief grid);
          • alpha    — the load-region mask (alpha > MASK_ALPHA_MIN is the
                        valid region).
        Returns None when the mask PNG doesn't exist or is unreadable, so the
        caller falls back to the single-depth path."""
        mp = self._mask_path()
        if not mp or not os.path.exists(mp):
            return None
        try:
            from PIL import Image
            im = Image.open(mp).convert("RGBA")
            if im.size != (W, H):
                im = im.resize((W, H), Image.NEAREST)
            arr = np.asarray(im, dtype=np.float32)
            rgb = arr[..., :3].mean(axis=2) / 255.0   # luminance
            alpha = arr[..., 3] / 255.0
            return rgb, alpha
        except Exception as exc:
            self._log(f"Не удалось прочитать masked-depth: {exc}")
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
    def _discontinuity_nodes(Z, inside, ratio, abs_m):
        """Flag grid nodes that touch a depth DISCONTINUITY — the same
        "over-long polygon" criterion the final mesh cutoff (_build_grid_faces)
        uses, but evaluated on the grid depths BEFORE smoothing/unprojection so
        it can drive the edge repair. An edge between two adjacent nodes is a
        discontinuity when |ΔZ| / Zmid > ratio (relative, scale-invariant — a
        smoothly sampled surface has |ΔZ|/Z ≈ the angular pixel pitch, while a
        silhouette / spike spikes far above it) OR |ΔZ| > abs_m metres
        (absolute cap). Both endpoints of every such edge are flagged. Returns
        an (stride,stride) bool grid restricted to `inside` nodes."""
        Z = np.asarray(Z, dtype=np.float64)
        inside = np.asarray(inside, dtype=bool)
        r = float(ratio)
        a = float(abs_m)
        eps = 1e-3
        disc = np.zeros(Z.shape, dtype=bool)
        dH = np.abs(Z[:, 1:] - Z[:, :-1])
        ZmH = np.maximum(0.5 * (Z[:, 1:] + Z[:, :-1]), eps)
        badH = (dH / ZmH > r) | (dH > a)
        disc[:, :-1] |= badH
        disc[:, 1:] |= badH
        dV = np.abs(Z[1:, :] - Z[:-1, :])
        ZmV = np.maximum(0.5 * (Z[1:, :] + Z[:-1, :]), eps)
        badV = (dV / ZmV > r) | (dV > a)
        disc[:-1, :] |= badV
        disc[1:, :] |= badV
        return disc & inside

    def _extrapolate_edge_z(self, Z2d, inside2d, band, win, idw_power=2.0,
                            disc_ratio=None, disc_abs_m=None, disc_reach=0,
                            clamp_slack=None):
        """Re-estimate the depth of grid nodes near the mask boundary by a
        gradient-preserving extrapolation of the trusted interior, and return
        (Z2d_new, n_repaired_nodes).

        Motivation: the masked depth is unreliable in a thin ring just inside
        the alpha edge (the mask catches a sliver of the body wall / the depth
        map's soft silhouette), which leaves spurious peaks and rises along the
        rim. Eroding the mask by a fixed offset is not an option — it would
        also shave off a real pile crest that happens to sit on the edge.

        The repair set is built in two parts:
          • RIM BAND  — inside nodes within `band` cells of the mask exterior
            (catches the gradual soft-silhouette contamination); plus
          • EDGE SPIKES — discontinuity nodes (the LONGPOLY criterion, see
            _discontinuity_nodes) that are connected to the rim through other
            discontinuity nodes, up to `disc_reach` cells deep. This catches a
            hard spike that pokes several cells inward — wider than `band` — so
            it is rebuilt in FULL instead of leaving its inner part behind
            (and contaminating the fit). An interior crater wall, not connected
            to the rim, is NOT absorbed and stays intact.

        Every node in the repair set is re-estimated by a local first-order
        (moving least-squares) plane  z ≈ a·dx + b·dy + c  fitted to the CORE
        nodes around it (inside, not being repaired, not a discontinuity), with
        (dx, dy) measured FROM the node — so the value at the node is simply c,
        the interior surface's own slope carried outward. A node on a genuine
        steep crest keeps climbing at the interior's rate; a phantom rim spike
        is pulled back onto the interior trend.

        `clamp_slack` (None = off) is an anti-dip guard: a first-order
        extrapolation of a surface descending toward the rim overshoots
        downward, so a former up-spike can leave a faint downward bend. The
        result is floored at the core window's minimum minus clamp_slack·range.
        Only the LOWER side is bounded — a genuine crest rising onto the edge
        keeps climbing (upward spikes are removed by the absorption, not here).
        """
        Z = np.asarray(Z2d, dtype=np.float64).copy()
        inside = np.asarray(inside2d, dtype=bool)
        band = int(band)
        win = int(win)
        if band <= 0 or not inside.any():
            return Z, 0

        # Discontinuity (spike) nodes — the LONGPOLY test on the raw grid
        # depths, BEFORE any smoothing, so the repair is driven by the real
        # depth jumps rather than waiting for the final mesh-level cutoff.
        if disc_ratio is not None or disc_abs_m is not None:
            disc = self._discontinuity_nodes(
                Z, inside,
                float("inf") if disc_ratio is None else disc_ratio,
                float("inf") if disc_abs_m is None else disc_abs_m)
        else:
            disc = np.zeros(inside.shape, dtype=bool)

        # Layered Chebyshev distance to the mask exterior, computed up to
        # band + disc_reach (outside = 0, first interior ring = 1, …). The rim
        # band is gdist∈[1, band]; the absorption is allowed to extend out to
        # gdist ≤ band + disc_reach.
        reach = max(0, int(disc_reach))
        maxd = band + reach
        gdist = np.zeros(inside.shape, dtype=np.int32)
        cur = ~inside
        for k in range(1, maxd + 1):
            nb = self._dilate_grid_mask(cur, 1) & inside & (gdist == 0)
            if not nb.any():
                break
            gdist[nb] = k
            cur = cur | nb
        repair = inside & (gdist > 0) & (gdist <= band)

        # Absorb edge-connected spikes: grow the repair set along discontinuity
        # nodes that touch the rim, but ONLY within band+disc_reach cells of the
        # boundary (`within`). The geometric bound stops the flood from walking
        # deep along a wall that merely grazes the rim; the disc-connectivity
        # requirement keeps interior crater walls (not reached from the rim)
        # intact. Together they rebuild a wide rim spike in full without eating
        # real geometry.
        if reach > 0 and disc.any() and repair.any():
            within = inside & (gdist > 0)
            frontier = repair.copy()
            for _ in range(reach):
                grow = (self._dilate_grid_mask(frontier, 1)
                        & disc & within & (~repair))
                if not grow.any():
                    break
                repair |= grow
                frontier = grow

        # Core = trusted interior: inside, not repaired, not a discontinuity.
        core = inside & (~repair) & (~disc)
        if not repair.any() or int(core.sum()) < 3:
            return Z, 0

        Hh, Ww = Z.shape
        er, ec = np.where(repair)
        new = Z.copy()
        p = float(idw_power)
        slack = None if clamp_slack is None else float(clamp_slack)
        n_done = 0
        for r, c in zip(er.tolist(), ec.tolist()):
            r0, r1 = max(0, r - win), min(Hh, r + win + 1)
            c0, c1 = max(0, c - win), min(Ww, c + win + 1)
            cm = core[r0:r1, c0:c1]
            if int(cm.sum()) < 3:
                continue
            zz = Z[r0:r1, c0:c1][cm]
            rr, cc = np.where(cm)
            dx = (cc + c0 - c).astype(np.float64)
            dy = (rr + r0 - r).astype(np.float64)
            d2 = dx * dx + dy * dy
            # Inverse-distance weights (+1 keeps them bounded and favours the
            # nearest core nodes); never zero since core never overlaps repair.
            w = 1.0 / (np.power(d2, p * 0.5) + 1.0)
            # Weighted normal equations for the plane [dx, dy, 1] -> z.
            Sxx = float(np.sum(w * dx * dx)); Sxy = float(np.sum(w * dx * dy))
            Sx = float(np.sum(w * dx));       Syy = float(np.sum(w * dy * dy))
            Sy = float(np.sum(w * dy));       S1 = float(np.sum(w))
            Sxz = float(np.sum(w * dx * zz)); Syz = float(np.sum(w * dy * zz))
            Sz = float(np.sum(w * zz))
            M = np.array([[Sxx, Sxy, Sx],
                          [Sxy, Syy, Sy],
                          [Sx,  Sy,  S1]], dtype=np.float64)
            rhs = np.array([Sxz, Syz, Sz], dtype=np.float64)
            try:
                # Value at the node (dx=dy=0) is the constant term.
                val = float(np.linalg.solve(M, rhs)[2])
            except np.linalg.LinAlgError:
                # Degenerate (collinear core) — fall back to the weighted mean.
                val = Sz / max(S1, 1e-12)
            # Anti-dip guard (LOWER bound only): a first-order extrapolation of
            # a surface that descends toward the rim overshoots downward, so a
            # former up-spike leaves a faint dip. Floor the value at the core
            # window's minimum minus slack·range. The UPPER side is left free
            # so a genuine crest rising onto the edge keeps climbing (upward
            # spikes are already removed by the absorption above, not here).
            if slack is not None:
                lo = float(zz.min())
                rng = float(zz.max() - zz.min())
                val = max(val, lo - slack * rng)
            new[r, c] = val
            n_done += 1
        return new, n_done

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
    def _mask_aware_gaussian(field, valid, sigma_cells):
        """Гауссово сглаживание heightfield'а, игнорирующее invalid-ячейки
        (трюк «нормализованной свёртки»: свёртка значений / свёртка маски).
        Используется для извлечения «тренда» source-рельефа: при σ≈0.5 м
        низкочастотная форма (насыпь) сохраняется, а мелкие бугры
        вычитаются → их потом квилтит residual-стадия.

        sigma_cells — σ в единицах ячеек. Если scipy.ndimage недоступен,
        откатываемся к box-smooth (3×3, ~σ²/0.5 итераций для эквивалента)."""
        f = np.where(valid, field.astype(np.float64), 0.0)
        w = valid.astype(np.float64)
        if _scipy_gaussian_filter is not None:
            f_s = _scipy_gaussian_filter(f, sigma=float(sigma_cells),
                                         mode="nearest")
            w_s = _scipy_gaussian_filter(w, sigma=float(sigma_cells),
                                         mode="nearest")
        else:
            # Box-smooth fallback: эквивалент Гаусса σ через ~σ²/0.5 проходов
            # (приблизительно — для качества лучше иметь scipy).
            iters = max(1, int(round(float(sigma_cells) ** 2 / 0.5)))
            f_s = DepthReconstructor._smooth_grid_z(
                f, np.ones_like(w, dtype=bool), iters)
            w_s = DepthReconstructor._smooth_grid_z(
                w, np.ones_like(w, dtype=bool), iters)
        out = np.where(w_s > 1e-6, f_s / np.maximum(w_s, 1e-12), 0.0)
        return out

    def _biharmonic_extrapolate(self, field, anchor):
        """Продолжает поле наружу anchor'а решением ∇⁴Z = 0 с Dirichlet BC
        (Z|anchor = field|anchor). На наклонной кромке (например, склон
        насыпи) бигармоник сохраняет ГРАДИЕНТ — поле «вытекает» наружу
        по тем же наклонам, без приведения к плоскости и без разрывов.

        Реализация:
          • skimage.restoration.inpaint_biharmonic — если установлено;
          • иначе itertaive Jacobi-Laplace (∇²Z = 0, fallback): даёт
            гладкую харм-интерполяцию граничных значений, но НЕ
            продолжает наклоны — приемлемо для умеренно-плоских случаев.

        field — (H, W) float64; anchor — (H, W) bool (где значения заданы).
        Возвращает заполненный field."""
        field = np.asarray(field, dtype=np.float64)
        anchor = np.asarray(anchor, dtype=bool)
        if not anchor.any():
            self._log("Biharmonic: anchor пуст — заполняю нулями.")
            return np.zeros_like(field)
        if anchor.all():
            return field.copy()
        if self.EXTRAP_BIHARMONIC and _skimage_biharmonic is not None:
            try:
                mask = ~anchor
                # skimage инпейнтит ячейки где mask=True, оставляя остальные.
                out = _skimage_biharmonic(field, mask)
                return np.asarray(out, dtype=np.float64)
            except Exception as exc:
                self._log(f"skimage biharmonic упал: {exc} — фоллбэк "
                          f"в Jacobi-Laplace.")
        # Fallback: итеративный Jacobi-Laplace (5-точечный stencil).
        return self._laplace_extrapolate(
            field, anchor,
            max_iter=int(self.LAPLACE_MAX_ITER),
            tol=float(self.LAPLACE_TOL))

    @staticmethod
    def _laplace_extrapolate(field, anchor, max_iter=4000, tol=1e-4):
        """Заполняет non-anchor ячейки решением ∇²Z = 0 с Dirichlet BC,
        итеративным Jacobi-усреднением 4 соседей. Сходится для любых
        связных регионов; не продолжает наклоны (даёт «harmonic
        interpolation» граничных значений) — fallback на случай отсутствия
        skimage."""
        Z = np.asarray(field, dtype=np.float64).copy()
        a = np.asarray(anchor, dtype=bool)
        if not a.any():
            return Z
        # Инициализация non-anchor средним по anchor — ускоряет сходимость.
        mean_anchor = float(Z[a].mean())
        Z = np.where(a, Z, mean_anchor)
        Hh, Ww = Z.shape
        # Padded buffer для удобного 5-pt stencil'а без if'ов по краям
        # (Neumann BC «zero-gradient» через replicate-padding).
        for it in range(int(max_iter)):
            Zp = np.pad(Z, 1, mode="edge")
            avg = 0.25 * (Zp[:-2, 1:-1] + Zp[2:, 1:-1]
                          + Zp[1:-1, :-2] + Zp[1:-1, 2:])
            Z_new = np.where(a, Z, avg)
            d = float(np.abs(Z_new - Z).max())
            Z = Z_new
            if d < tol:
                break
        return Z

    def _apply_descent_cap(self, trend, anchor, cell_size_m, scale_m,
                           floor_eff=None):
        """Адаптивный верхний cap на тренд СНАРУЖИ anchor'а:
        spускает Z от граничного значения к динамическому floor'у на
        длине ~scale_m. Внутри anchor — НИЧЕГО не трогает.

        floor_eff = min(trend[anchor]) по умолчанию — это «динамический
        пол», совпадающий с самой низкой точкой видимой части рельефа.

        cap = Z_b * (1−t) + floor_eff * t, где t = clip(d/scale, 0, 1)."""
        trend = np.asarray(trend, dtype=np.float64)
        anchor = np.asarray(anchor, dtype=bool)
        if not anchor.any() or anchor.all():
            return trend
        if floor_eff is None:
            floor_eff = float(trend[anchor].min())
        seed_y, seed_x, dist_cells = self._bfs_seeds(anchor)
        sy = np.where(seed_y >= 0, seed_y, 0)
        sx = np.where(seed_x >= 0, seed_x, 0)
        Z_b = trend[sy, sx]
        d_m = dist_cells * float(cell_size_m)
        t = np.clip(d_m / max(float(scale_m), 1e-6), 0.0, 1.0)
        cap = Z_b * (1.0 - t) + float(floor_eff) * t
        return np.where(anchor, trend, np.minimum(trend, cap))

    @staticmethod
    def _erode_grid_z(Z, valid, iters):
        """Mask-aware 3x3 MIN filter on a (stride, stride) depth grid, repeated
        `iters` times. Each valid cell is replaced by the minimum of itself and
        its valid 3x3 neighbours; invalid cells are left untouched and never
        contribute. Morphological erosion of the height-field — used by the
        extrapolation to build a LOWER ENVELOPE of the measured relief (pile
        skirt instead of crown) so the seed for occluded cells is the nearest
        LOW measurement, not whatever single tall point the BFS hit first."""
        Zc = Z.astype(np.float64).copy()
        Hh, Ww = Zc.shape
        for _ in range(int(iters)):
            cur = np.where(valid, Zc, np.inf)
            new = cur.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ys_d, ye_d = max(0, -dy), Hh - max(0, dy)
                    xs_d, xe_d = max(0, -dx), Ww - max(0, dx)
                    ys_s, ye_s = max(0, dy), Hh - max(0, -dy)
                    xs_s, xe_s = max(0, dx), Ww - max(0, -dx)
                    tgt = new[ys_d:ye_d, xs_d:xe_d]
                    src = cur[ys_s:ye_s, xs_s:xe_s]
                    np.minimum(tgt, src, out=tgt)
            # Apply only to valid cells; invalids must not pick up the +inf or
            # a far-away neighbour's value.
            Zc = np.where(valid & np.isfinite(new), new, Zc)
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
    def _bilinear_sample_masked(field, valid, xs, ys, xmin, xmax, ymin, ymax,
                                fill=0.0):
        """Mask-aware bilinear sampling: normalized convolution через bilinear.
        В невалидных ячейках `field` (там, где `valid` = False) стоит ноль —
        обычный bilinear в граничном 2×2 стенциле разбавляет ими настоящие
        значения source'а и занижает Z. Это вычисляет
            sample = bilinear(field*valid) / bilinear(valid)
        что даёт ИСТИННОЕ локальное среднее по валидной части стенциля; на
        крутом подъёме source'а к границе (back side of a hill) разница
        важна — от неё прямо зависит, сидит ли ext-меш на одной высоте с
        source-мешем или проваливается на ≈|residual|.

        Возвращает sample там где bilinear(valid) > 0, fill — иначе."""
        f = np.where(valid, field.astype(np.float64), 0.0)
        w = np.asarray(valid, dtype=np.float64)
        num = DepthReconstructor._bilinear_sample(
            f, xs, ys, xmin, xmax, ymin, ymax)
        den = DepthReconstructor._bilinear_sample(
            w, xs, ys, xmin, xmax, ymin, ymax)
        return np.where(den > 1e-6, num / np.maximum(den, 1e-12),
                        float(fill))

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

    def _napolnitel_floor_z(self):
        """World Z of the napolnitel solid's bottom — used by the extrapolation
        as the plateau the unmeasured terrain rests on. Returns the lowest
        transformed vertex Z plus a small upward margin (so the extrap stays
        STRICTLY inside the body and the point-in-mesh clip keeps the whole
        plateau, instead of trimming the rim where the rectangle's vertices
        would otherwise sit right on / just below the floor). Returns None
        when the napolnitel can't be loaded — the caller then falls back to
        z_min_visible − EXTRAP_FLOOR_FROM_MIN_M."""
        loaded = self._load_tonar_obj()
        if loaded is None:
            return None
        verts, _faces = loaded
        if verts is None or len(verts) == 0:
            return None
        margin = 0.02  # 2 cm above the literal floor — defensive against
                      # point-in-mesh edge cases at the very bottom.
        return float(verts[:, 2].min()) + margin

    def _napolnitel_xy_center(self):
        """XY-центр AABB napolnitel'а в мировых координатах (после
        TONAR_OBJ_TRANSFORM). Используется Stage 3 для размещения
        TARGET_SIZE_M-прямоугольника по центру кузова. None при ошибке."""
        loaded = self._load_tonar_obj()
        if loaded is None:
            return None
        verts, _faces = loaded
        if verts is None or len(verts) == 0:
            return None
        xy = np.asarray(verts, dtype=np.float64)[:, :2]
        cx = float(0.5 * (xy[:, 0].min() + xy[:, 0].max()))
        cy = float(0.5 * (xy[:, 1].min() + xy[:, 1].max()))
        return cx, cy

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

    # ------------------------------------------------------------------
    # Stage 3 — pattern-based extrapolation to TARGET_SIZE_M rectangle
    # ------------------------------------------------------------------
    def _pattern_extrapolate(self, src_verts, src_faces):
        """Two-mesh с Delaunay-glue стыковкой. Source-mesh идёт в результат
        БАЙТ-В-БАЙТ. Снаружи source'а — ext-кадр на регулярной сетке
        TARGET_SIZE_M (cells где НИ ОДИН угол не anchor → 1-cell gap),
        между ними тонкая полоса glue-треугольников через 2D Delaunay,
        отфильтрованных по длине ребра.

        Алгоритм:
          1. Source rasterize → src_field, src_valid (для синтеза).
          2. Target grid TARGET_SIZE_M, центр в napolnitel.
          3. Trend/residual split на SOURCE.
          4. Sample → target. anchor = target cells in src bbox & valid.
          5. Тренд: ∇⁴Z = 0 outside anchor (skimage / Jacobi-Laplace).
          6. Adaptive descent cap (EXTRAP_DECAY_SCALE_M).
          7. Residual: image-quilting _quilt_z на stationary поле.
          8. Z = trend_capped + residual_filled.
          9. Two-mesh + glue:
             • ext-faces: cells где no-anchor-corner;
             • inner-ring: non-anchor узлы 8-смежные с anchor;
             • glue: Delaunay (src-bnd ∪ inner-ring), filter max-edge.
         10. Concat: source + ext + glue, общий remap.

        Возвращает (verts, faces, skip_global_cleanup=True)."""
        src_verts = np.asarray(src_verts, dtype=np.float64)
        src_faces = np.asarray(src_faces, dtype=np.int64)
        if src_verts.shape[0] == 0 or src_faces.shape[0] == 0:
            return None

        # --- 1. Source rasterization (служебная) ---------------------
        used_idx = np.unique(src_faces.reshape(-1))
        used_v = src_verts[used_idx]
        xy = used_v[:, :2]
        xmin_s = float(xy[:, 0].min())
        xmax_s = float(xy[:, 0].max())
        ymin_s = float(xy[:, 1].min())
        ymax_s = float(xy[:, 1].max())
        w_s = max(xmax_s - xmin_s, 1e-6)
        h_s = max(ymax_s - ymin_s, 1e-6)
        src_field, src_valid = self._rasterize_xyz(
            used_v, xmin_s, ymin_s, w_s, h_s)
        if not src_valid.any():
            self._log("Pattern extrapolate: rasterize вернул пустоту — "
                      "оставляю только source.")
            return None

        # --- 2. Target grid centred on napolnitel. -------------------
        center = self._napolnitel_xy_center()
        if center is None:
            cx = 0.5 * (xmin_s + xmax_s)
            cy = 0.5 * (ymin_s + ymax_s)
            self._log("Pattern extrapolate: napolnitel недоступен — "
                      "центр цели взят по bbox source'а.")
        else:
            cx, cy = center
        Tx_m, Ty_m = float(self.TARGET_SIZE_M[0]), float(self.TARGET_SIZE_M[1])
        tx_min = cx - 0.5 * Tx_m
        ty_min = cy - 0.5 * Ty_m
        tx_max = tx_min + Tx_m
        ty_max = ty_min + Ty_m

        res = float(self.TARGET_RES_PER_M)
        tnx = max(2, int(round(Tx_m * res)) + 1)
        tny = max(2, int(round(Ty_m * res)) + 1)

        gx = np.linspace(tx_min, tx_max, tnx)
        gy = np.linspace(ty_min, ty_max, tny)
        GX, GY = np.meshgrid(gx, gy)            # (tny, tnx)

        # --- 3. Trend + residual split на SOURCE. -------------------
        # Тренд — Gaussian σ ≈ TREND_SMOOTH_SIGMA_M метра (с маской
        # source'а — invalid-ячейки не загрязняют). Residual = source −
        # trend (только там где source валиден); это уже stationary
        # mean≈0 поле, на нём image-quilting корректно даёт «равномерную
        # шероховатость» вместо «грядок».
        sigma_cells = float(self.TREND_SMOOTH_SIGMA_M) * res
        trend_src = self._mask_aware_gaussian(
            src_field, src_valid, sigma_cells)
        residual_src = np.where(src_valid, src_field - trend_src, 0.0)

        # --- 4. Sample to target + anchor mask. ---------------------
        # MASK-AWARE bilinear: невалидные ячейки src_field/residual_src = 0,
        # обычный bilinear на границе разбавлял ими настоящие значения и
        # занижал Z на границе anchor'а (≈|residual| / 2 в худшем случае).
        # mask-aware версия делит на bilinear(valid), убирая разбавление.
        valid_f = self._bilinear_sample(
            src_valid.astype(np.float64), GX.ravel(), GY.ravel(),
            xmin_s, xmax_s, ymin_s, ymax_s).reshape(tny, tnx)
        # Полная карта source (НЕ только тренд) — это и будет Dirichlet BC
        # биграмоника: ext-меш C0-сшивается с реальными Z source-меша.
        src_tgt = self._bilinear_sample_masked(
            src_field, src_valid, GX.ravel(), GY.ravel(),
            xmin_s, xmax_s, ymin_s, ymax_s).reshape(tny, tnx)
        # trend_tgt всё ещё нужен — он задаёт «гладкое» направление спуска
        # cap'у (Z_b cap'а считается по trend'у, а не по полной карте, чтобы
        # ext снаружи не уходил «зубцами» вслед за high-freq source'а).
        trend_tgt = self._bilinear_sample(
            trend_src, GX.ravel(), GY.ravel(),
            xmin_s, xmax_s, ymin_s, ymax_s).reshape(tny, tnx)
        in_bbox = ((GX >= xmin_s) & (GX <= xmax_s) &
                   (GY >= ymin_s) & (GY <= ymax_s))
        anchor = in_bbox & (valid_f > 0.5)
        if not anchor.any():
            self._log("Pattern extrapolate: source bbox не пересекся с "
                      "целью — extrapolation пропущен.")
            return None

        # --- 5. Биграмоник на ПОЛНОМ source'е (был на trend'е). ------
        # Раньше Dirichlet = trend (Gauss σ≈0.5 м), и на крутом подъёме
        # source'а к границе тренд занижал boundary Z на ≈|residual| (≈0.5 м
        # по логам). Из-за этого ext-меш начинался ниже source'а и glue-
        # треугольники тянулись вертикально (back-of-hill артефакт). С
        # полной картой как BC ext-меш на границе сидит на тех же Z, что и
        # source-меш — glue работает на плоской высоте.
        base_filled = self._biharmonic_extrapolate(src_tgt, anchor)

        # --- 6. Adaptive DESCENT CAP. ---------------------------------
        # Cap считается ОТДЕЛЬНО через гладкий trend_filled: его Z_b на
        # границе — это сглаженный source (без high-freq), поэтому спуск
        # от него идёт плавно. Если бы Z_b брали из полной карты, ext
        # снаружи получал бы «зубчатый» спуск от high-freq source'а.
        cell_size_m = 1.0 / max(res, 1e-6)
        floor_eff = float(trend_tgt[anchor].min())
        trend_filled = self._biharmonic_extrapolate(trend_tgt, anchor)
        cap_field = self._apply_descent_cap(
            trend_filled, anchor,
            cell_size_m=cell_size_m,
            scale_m=float(self.EXTRAP_DECAY_SCALE_M),
            floor_eff=floor_eff)
        # Применяем cap к полной карте: внутри anchor оставляем base_filled
        # (= src_tgt), снаружи — min(base_filled, cap_field). Так на границе
        # ext держит source'овую высоту, а дальше плавно спускается к floor.
        base_capped = np.where(
            anchor, base_filled, np.minimum(base_filled, cap_field))

        # --- 7. Residual: image-quilting НА НУЛЕ внутри anchor'а. -----
        # Раньше residual_init на anchor'е = residual_tgt (ещё и diluted
        # bilinear'ом — см. п.4), а base_filled на anchor'е был trend_tgt:
        # сумма (trend+residual) ≈ source. Теперь base_filled на anchor'е
        # уже несёт ПОЛНЫЙ source (= trend + residual внутри), поэтому
        # residual на anchor'е должен быть 0, иначе мы добавили бы детали
        # source'а дважды. Квилт всё ещё кидает source-патчи СНАРУЖИ
        # anchor'а — feather overlap'а тянет их к нулю на самой границе,
        # это даёт текстуру в ext'е без скачка на шве.
        residual_init = np.where(anchor, 0.0, np.nan)
        residual_filled = self._quilt_z(
            residual_src, src_valid, residual_init, anchor,
            patch_frac=float(self.PATTERN_PATCH_FRAC),
            overlap_frac=float(self.PATTERN_OVERLAP_FRAC),
            candidates=int(self.PATTERN_CANDIDATES),
            seed=self.PATTERN_SEED)
        if self.PATTERN_FINAL_SMOOTH > 0:
            residual_filled = self._smooth_preserve(
                residual_filled, anchor, int(self.PATTERN_FINAL_SMOOTH))

        # --- 8. Z = base (capped, на anchor = полный source) + residual.
        Z_filled = base_capped + residual_filled

        # --- 8a. INNER-RING LIFT: подтягиваем inner-ring к НАСТОЯЩИМ Z
        # source-меша (через KDTree-IDW по src_bnd-вершинам). Это
        # закрывает фундаментальный разрыв: bilinear-of-rasterized field
        # сглаживает пики source'а (несколько вершин усредняются в одной
        # растер-ячейке), и glue-треугольник между src-меш-вершиной
        # (настоящий Z) и inner-ring (растеризованный/сглаженный Z) на
        # back-of-hill пиках вытягивается вертикально на ≈|residual|.
        # Здесь δ = настоящий_Z(src) − Z_filled(inner_ring) распространяется
        # внутрь ext'а с exp(-d/EXTRAP_RIM_LIFT_SCALE_M) feather'ом,
        # чтобы не возникла 1-клеточная стена.
        a_bool = anchor
        # cell_inside_anchor — потребуется и для cell_keep ниже.
        cell_inside_anchor = (a_bool[:-1, :-1] | a_bool[:-1, 1:] |
                              a_bool[1:, :-1] | a_bool[1:, 1:])
        cell_keep = ~cell_inside_anchor
        inner_ring = self._compute_inner_ring(anchor)
        src_bnd_idx = self._find_boundary_vertices(src_faces)
        Z_filled, n_lifted, max_delta = self._lift_inner_ring_to_source(
            Z_filled, inner_ring, anchor,
            src_verts, src_bnd_idx, GX, GY,
            cell_size_m=cell_size_m,
            feather_scale_m=float(
                getattr(self, "EXTRAP_RIM_LIFT_SCALE_M", 0.3)))

        try:
            t_min = float(trend_src[src_valid].min())
            t_max = float(trend_src[src_valid].max())
            s_min_tgt = float(src_tgt[anchor].min())
            s_max_tgt = float(src_tgt[anchor].max())
            r_amp = float(np.abs(residual_src[src_valid]).max())
            engine = ("skimage-biharmonic"
                      if (self.EXTRAP_BIHARMONIC
                          and _skimage_biharmonic is not None)
                      else "jacobi-laplace")
            n_capped = int(((base_capped < base_filled - 1e-6)
                            & (~anchor)).sum())
            self._log(
                f"Trend σ={sigma_cells:.1f}cell (~"
                f"{self.TREND_SMOOTH_SIGMA_M:g}м), trend Z∈"
                f"[{t_min:.2f}, {t_max:.2f}], "
                f"src_tgt[anchor] Z∈[{s_min_tgt:.2f}, {s_max_tgt:.2f}], "
                f"|residual|max={r_amp:.2f}м; "
                f"{engine} + descent cap (по trend'у) "
                f"scale={self.EXTRAP_DECAY_SCALE_M}м "
                f"(floor_eff={floor_eff:.2f}, cap сработал на "
                f"{n_capped} ячейках); rim-lift: "
                f"{n_lifted} inner-ring узлов подтянуто к src-меш-Z "
                f"(|δ|max={max_delta:.2f}м, feather="
                f"{float(getattr(self, 'EXTRAP_RIM_LIFT_SCALE_M', 0.3)):.2f}м).")
        except Exception:
            pass

        # --- 9. TWO-MESH + GLUE: source preserved byte-by-byte, ext только
        # СНАРУЖИ anchor'а (1-cell gap), Delaunay glue зашивает шов.
        # cell_keep: ячейку держим если НИ ОДИН из 4 углов не anchor.
        # (cell_keep и inner_ring уже посчитаны выше в п.8a — переиспользуем.)
        ext_faces_grid = self._build_regular_grid_faces_masked(
            tnx, tny, cell_keep)
        if ext_faces_grid.shape[0] == 0:
            self._log("Pattern extrapolate: ext пуст (source покрыл всю "
                      "цель) — оставляю только source.")
            return src_verts, src_faces, True

        # inner_ring + src_bnd_idx уже получены в п.8a.
        inner_flat = np.where(inner_ring.ravel())[0]

        # ext_verts: все узлы, используемые ext_faces, PLUS все inner-ring
        # узлы (нужны для glue, даже если ни одна ext-клетка их не юзает).
        ext_verts_full = np.column_stack(
            [GX.ravel(), GY.ravel(), Z_filled.ravel()])
        used_set = np.unique(np.concatenate(
            [ext_faces_grid.reshape(-1), inner_flat]))
        ext_verts = ext_verts_full[used_set]
        remap_grid_to_ext = np.full(tnx * tny, -1, dtype=np.int64)
        remap_grid_to_ext[used_set] = np.arange(used_set.shape[0])
        ext_faces_local = remap_grid_to_ext[ext_faces_grid]

        # Densification src-границы: вставляем промежуточные вершины вдоль
        # длинных граничных рёбер source-меша (XYZ линейно), чтобы Delaunay
        # имел плотный «гребень» для построения мостов к inner-ring'у. Без
        # этого ~38% inner-ring узлов в логах оставались без bridge'а — там,
        # где соседние src_bnd-вершины стояли в >10 см друг от друга
        # (typically около napolnitel-clip кромок). Densified-вершины НЕ
        # принадлежат source-мешу (его сохраняем байт-в-байт), а
        # геометрически они лежат строго на исходном граничном ребре,
        # поэтому t-junction между source-edge и densified-glue-edge
        # совпадает в 3D и видимых щелей не должно создавать.
        src_bnd_edges = self._find_boundary_edges(src_faces)
        densify_factor = float(getattr(
            self, "EXTRAP_DENSIFY_SPACING_FACTOR", 1.0))
        densify_spacing_m = cell_size_m * densify_factor
        extra_xyz, _edges_path = self._densify_boundary(
            src_verts, src_bnd_edges, densify_spacing_m)
        n_extras = int(extra_xyz.shape[0])

        # Glue: 2D Delaunay на (src_bnd ∪ densified-extras ∪ inner_ring).
        # src_bnd_idx уже получен выше в п.8a.
        n_orig_src_bnd = int(src_bnd_idx.shape[0])
        n_src_v = int(src_verts.shape[0])
        n_ext_v = int(ext_verts.shape[0])
        if n_orig_src_bnd == 0 or inner_flat.shape[0] == 0:
            glue_faces_final = np.zeros((0, 3), dtype=np.int64)
            n_glue = 0
            n_src_side = n_orig_src_bnd
        else:
            # «src-сторона» = оригинальные src_bnd + densified extras. Все
            # они лежат на source-границе и для bridge-фильтра считаются src.
            if n_extras > 0:
                src_side_xy = np.concatenate([
                    src_verts[src_bnd_idx, :2],
                    extra_xyz[:, :2],
                ], axis=0)
                src_side_z = np.concatenate([
                    src_verts[src_bnd_idx, 2],
                    extra_xyz[:, 2],
                ], axis=0)
            else:
                src_side_xy = src_verts[src_bnd_idx, :2]
                src_side_z = src_verts[src_bnd_idx, 2]
            inner_xy = np.column_stack(
                [GX.ravel()[inner_flat], GY.ravel()[inner_flat]])
            inner_z = Z_filled.ravel()[inner_flat]
            # cell_size_m уже посчитан выше (для cap'а и rim-lift'а).
            # max_edge_factor=None → берётся из EXTRAP_GLUE_MAX_EDGE_FACTOR.
            glue_local, n_src_side, n_inner = self._build_delaunay_glue(
                src_side_xy, src_side_z, inner_xy, inner_z,
                cell_size_m=cell_size_m, max_edge_factor=None)
            n_glue = int(glue_local.shape[0])
            # Combined → final мэппинг:
            #   [0, n_orig_src_bnd) → src_verts via src_bnd_idx
            #   [n_orig_src_bnd, n_src_side) → extras (в конце combined mesh)
            #   [n_src_side, end) → ext_verts via remap_grid_to_ext
            combined_to_final = np.empty(
                n_src_side + n_inner, dtype=np.int64)
            combined_to_final[:n_orig_src_bnd] = src_bnd_idx
            if n_extras > 0:
                combined_to_final[n_orig_src_bnd:n_src_side] = (
                    n_src_v + n_ext_v + np.arange(n_extras))
            ext_idx_for_inner = remap_grid_to_ext[inner_flat]
            combined_to_final[n_src_side:] = n_src_v + ext_idx_for_inner
            glue_faces_final = combined_to_final[glue_local]
            valid = (glue_faces_final >= 0).all(axis=1)
            glue_faces_final = glue_faces_final[valid]

        # Concat: source (UNTOUCHED) + ext + densified-extras + glue-faces.
        if n_extras > 0:
            all_verts = np.concatenate(
                [src_verts, ext_verts, extra_xyz], axis=0)
        else:
            all_verts = np.concatenate([src_verts, ext_verts], axis=0)
        all_faces = np.concatenate([
            src_faces,
            ext_faces_local + n_src_v,
            glue_faces_final,
        ], axis=0)

        zmin_s = float(src_field[src_valid].min())
        zmax_s = float(src_field[src_valid].max())
        self._log(
            f"Pattern extrapolate (two-mesh): source {src_verts.shape[0]} верш / "
            f"{src_faces.shape[0]} тр. сохранён байт-в-байт; "
            f"ext {ext_verts.shape[0]} верш / {ext_faces_local.shape[0]} тр.; "
            f"densify: +{n_extras} вершин на src-границе "
            f"(spacing={densify_spacing_m*100:.1f} см); "
            f"glue {n_glue} тр. ({n_orig_src_bnd}+{n_extras} src-side + "
            f"{inner_flat.shape[0]} inner-ring узлов). "
            f"source Z∈[{zmin_s:.2f}, {zmax_s:.2f}].")

        # --- 10. SEAM BUFFER: затыкаем остаточные дыры вдоль шва.
        # source+ext+glue может оставить orphan inner-ring узлы (glue-
        # фильтр отрезал длинные стыковочные рёбра) — топологические
        # дыры в виде маленьких треугольных «окон». Находим все
        # boundary-loop'ы в combined-меше; единственный самый большой
        # (TARGET-прямоугольник) — внешний контур, остальные — дыры,
        # которые триангулируем buffer-треугольниками.
        #
        # is_fixed: вершины с этим флагом не двигаются Laplacian-relax'ом
        # ВНУТРИ buffer-зоны:
        #   • [0, n_src_v) — source-меш, сохраняется байт-в-байт;
        #   • [n_src_v + n_ext_v, end) — densify-extras (лежат на src-edges
        #     по построению; сдвинуть их по Z разорвёт согласованность с
        #     source-кромкой).
        # Всё остальное (= ext-grid вершины, включая orphan inner-ring) —
        # moveable: relax сглаживает их Z по соседям в loop'е, чтобы убрать
        # «треугольные гребни» между высоким Z source'а и низким Z ext'а.
        if self.SEAM_HOLE_STITCH:
            n_total = int(all_verts.shape[0])
            is_fixed = np.zeros(n_total, dtype=bool)
            is_fixed[:n_src_v] = True
            if n_extras > 0:
                is_fixed[n_src_v + n_ext_v:] = True
            all_verts, all_faces = self._stitch_seam_holes(
                all_verts, all_faces, is_fixed=is_fixed)

        return all_verts, all_faces, True

    @staticmethod
    def _build_regular_grid_faces_masked(nx, ny, cell_keep):
        """Триангуляция регулярной (nx×ny)-вершинной сетки, но включаются
        только те ячейки, где cell_keep[i, j] = True. Возвращает (M, 3)
        int64 — индексы в расплющенной nx*ny сетке."""
        cx = nx - 1
        cy = ny - 1
        cell_keep = np.asarray(cell_keep, dtype=bool)
        if cx <= 0 or cy <= 0 or not cell_keep.any():
            return np.zeros((0, 3), np.int64)
        ti = np.arange(cy)[:, None]
        si = np.arange(cx)[None, :]
        a = ti * nx + si
        b = a + 1
        c = a + nx
        d = c + 1
        k = cell_keep[:cy, :cx]
        a = a[k]; b = b[k]; c = c[k]; d = d[k]
        f1 = np.stack([a, b, d], axis=1)
        f2 = np.stack([a, d, c], axis=1)
        return np.concatenate([f1, f2], axis=0).astype(np.int64)

    @staticmethod
    def _find_boundary_vertices(faces):
        """Индексы вершин на «голой» грани меша (used by ровно 1 face) —
        внешний контур + кромки внутренних дыр."""
        f = np.asarray(faces, dtype=np.int64)
        if f.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        edges = np.concatenate(
            [f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
        edges_sorted = np.sort(edges, axis=1)
        _, inv, cnt = np.unique(
            edges_sorted, axis=0, return_inverse=True, return_counts=True)
        is_bnd = (cnt[inv] == 1)
        bnd_flat = edges[is_bnd].reshape(-1)
        return np.unique(bnd_flat)

    @staticmethod
    def _find_boundary_edges(faces):
        """Граничные РЁБРА меша (используются ровно одной гранью). Возвращает
        (E, 2) int64 — пары индексов вершин в исходном vertex-наборе. В отличие
        от _find_boundary_vertices это сохраняет связность кромки — нужно для
        densification вдоль длинных граничных сегментов."""
        f = np.asarray(faces, dtype=np.int64)
        if f.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.int64)
        edges = np.concatenate(
            [f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
        edges_sorted = np.sort(edges, axis=1)
        _, inv, cnt = np.unique(
            edges_sorted, axis=0, return_inverse=True, return_counts=True)
        is_bnd = (cnt[inv] == 1)
        return edges[is_bnd]

    @staticmethod
    def _densify_boundary(src_verts, src_bnd_edges, target_spacing_m):
        """Вставить промежуточные вершины вдоль граничных рёбер source-меша:
        для каждого ребра длиной L > target_spacing разбить на ceil(L/spacing)
        равных частей, XY и Z интерполируются линейно. Возвращает:
          extra_xyz — (M, 3) новые вершины (НЕ принадлежат source-мешу,
                      используются только в glue);
          edges_path — список (start_idx, mid_idxs..., end_idx) где start/end
                       индексируют src_verts, mid_idxs индексируют extra_xyz
                       со смещением n_src_v. Сохраняет порядок «по ребру».

        Промежуточные вершины строго лежат на исходном граничном ребре
        source-меша (линейная интерполяция), поэтому t-junction между
        source-edge и densified-glue-edge геометрически совпадает в 3D —
        видимых щелей не должно быть."""
        src_verts = np.asarray(src_verts, dtype=np.float64)
        edges = np.asarray(src_bnd_edges, dtype=np.int64)
        if edges.shape[0] == 0 or target_spacing_m <= 0:
            return (np.zeros((0, 3), dtype=np.float64), [])
        extra_xyz = []
        edges_path = []
        n_src_v = int(src_verts.shape[0])
        next_extra_local = 0
        for (i, j) in edges:
            pa = src_verts[i]
            pb = src_verts[j]
            L = float(np.linalg.norm(pb[:2] - pa[:2]))  # XY-длина
            if L <= target_spacing_m or not np.isfinite(L):
                edges_path.append([int(i), int(j)])
                continue
            n_seg = int(np.ceil(L / target_spacing_m))
            mid_local = []
            for k in range(1, n_seg):
                t = k / n_seg
                p = pa * (1.0 - t) + pb * t
                extra_xyz.append(p)
                mid_local.append(next_extra_local)
                next_extra_local += 1
            # path: src i → extras (со смещением n_src_v) → src j
            path = [int(i)] + [int(n_src_v + m) for m in mid_local] + [int(j)]
            edges_path.append(path)
        return (np.asarray(extra_xyz, dtype=np.float64) if extra_xyz
                else np.zeros((0, 3), dtype=np.float64),
                edges_path)

    @staticmethod
    def _compute_inner_ring(anchor):
        """Bool-маска target-grid'а: non-anchor узлы, 8-смежные с anchor'ом.
        «Внутреннее кольцо» ext'а — узлы, граничащие с source'ом, через
        которые пройдёт glue."""
        a = np.asarray(anchor, dtype=bool)
        dil = a.copy()
        dil[:-1, :] |= a[1:, :]
        dil[1:, :] |= a[:-1, :]
        dil[:, :-1] |= a[:, 1:]
        dil[:, 1:] |= a[:, :-1]
        dil[:-1, :-1] |= a[1:, 1:]
        dil[1:, 1:] |= a[:-1, :-1]
        dil[:-1, 1:] |= a[1:, :-1]
        dil[1:, :-1] |= a[:-1, 1:]
        return dil & ~a

    def _lift_inner_ring_to_source(self, Z_filled, inner_ring, anchor,
                                    src_verts, src_bnd_idx, GX, GY,
                                    cell_size_m, feather_scale_m):
        """Подтянуть inner-ring ext'а к НАСТОЯЩИМ Z source-меша и распространить
        коррекцию внутрь ext'а с экспоненциальным feather'ом.

        Зачем: bilinear-of-rasterized field теряет high-freq на пиках source'а
        (несколько вершин усредняются в одной растер-ячейке `_rasterize_xyz`).
        Glue-треугольник между src_bnd-вершиной (настоящий Z из меша) и
        inner-ring-узлом (растеризованный/сглаженный Z) из-за этого тянется
        вертикально на ≈|residual|. Здесь для каждого inner-ring узла берём
        IDW по k=4 ближайшим src_bnd-вершинам и сдвигаем Z_filled на
        δ = target_z − Z_filled. Дальше внутрь ext'а δ затухает по
        exp(-d/feather_scale_m), чтобы не возникла 1-клеточная «стена».

        Возвращает (Z_corrected, n_inner_ring_nodes, |δ|max)."""
        if src_bnd_idx is None or src_bnd_idx.shape[0] == 0:
            return Z_filled, 0, 0.0
        inner_yy, inner_xx = np.where(inner_ring)
        if inner_yy.size == 0:
            return Z_filled, 0, 0.0
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            self._log("Inner-ring lift: scipy.spatial.cKDTree недоступен — "
                      "пропуск.")
            return Z_filled, int(inner_yy.size), 0.0

        src_bnd_xy = np.asarray(src_verts[src_bnd_idx, :2], dtype=np.float64)
        src_bnd_z = np.asarray(src_verts[src_bnd_idx, 2], dtype=np.float64)
        if src_bnd_xy.shape[0] < 1:
            return Z_filled, int(inner_yy.size), 0.0

        tree = cKDTree(src_bnd_xy)
        inner_pts = np.column_stack(
            [GX[inner_yy, inner_xx], GY[inner_yy, inner_xx]])
        k = int(min(4, src_bnd_xy.shape[0]))
        distances, indices = tree.query(inner_pts, k=k)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        # IDW p=2; eps защищает от деления на ноль если inner совпал с src_bnd.
        eps = 1e-6
        w = 1.0 / np.maximum(distances, eps) ** 2
        w = w / w.sum(axis=1, keepdims=True)
        target_z = (src_bnd_z[indices] * w).sum(axis=1)

        old_inner_z = Z_filled[inner_yy, inner_xx]
        delta_inner = target_z - old_inner_z
        max_delta = float(np.abs(delta_inner).max())

        # BFS от inner-ring: для каждой клетки сетки даёт ближайшую
        # inner-ring клетку и расстояние в шагах.
        seed_y, seed_x, dist_cells = self._bfs_seeds(inner_ring)

        # δ-карта на inner-ring узлах, в остальных — 0.
        delta_at_inner = np.zeros(Z_filled.shape, dtype=np.float64)
        delta_at_inner[inner_yy, inner_xx] = delta_inner
        # δ_seed[cell] = δ ближайшего inner-ring узла.
        sy = np.where(seed_y >= 0, seed_y, 0)
        sx = np.where(seed_x >= 0, seed_x, 0)
        delta_seed = delta_at_inner[sy, sx]
        # exp(-d/feather) — на inner-ring d=0 → decay=1 (полная коррекция),
        # дальше внутрь ext'а плавно сходит к 0.
        d_m = dist_cells * float(cell_size_m)
        finite = np.isfinite(d_m)
        scale = max(float(feather_scale_m), 1e-6)
        decay = np.where(finite, np.exp(-d_m / scale), 0.0)
        correction = delta_seed * decay

        # Anchor не трогаем — там Z уже на src_tgt, поднимать его «к самому
        # себе» через src_bnd-IDW не нужно и может создать ringing.
        Z_corrected = np.where(~anchor, Z_filled + correction, Z_filled)
        return Z_corrected, int(inner_yy.size), max_delta

    def _build_delaunay_glue(self, src_bnd_xy, src_bnd_z,
                              inner_xy, inner_z, cell_size_m,
                              max_edge_factor=None):
        """2D Delaunay на (src-boundary ∪ inner-ring), фильтр по длине
        СТЫКОВОЧНЫХ (src→inner) рёбер ≤ max_edge_factor × cell_size,
        оставляем только «мост»-треугольники (с src-вершиной И inner-).

        Почему фильтр именно по src→inner, а не «max из 3 рёбер»: bridge-
        треугольник вида (src_a, src_b, inner_c) имеет 1 ребро src-src и
        2 ребра src-inner. src-src может быть длинным, если граница
        source'а редкая в этом месте (например, у napolnitel-clip кромки)
        — это совершенно валидный bridge, который заполняет gap. Старый
        фильтр выкидывал такие треугольники, оставляя «дырки в виде
        треугольников» вдоль шва.

        max_edge_factor=None → EXTRAP_GLUE_MAX_EDGE_FACTOR из настроек.
        Возвращает (glue_faces в combined-индексах, n_src_bnd, n_inner)."""
        if max_edge_factor is None:
            max_edge_factor = float(getattr(
                self, "EXTRAP_GLUE_MAX_EDGE_FACTOR", 3.0))
        if _scipy_delaunay is None:
            self._log("Glue: scipy.spatial.Delaunay недоступен — "
                      "стыковочные треугольники не построены (будет gap).")
            return (np.zeros((0, 3), dtype=np.int64),
                    int(src_bnd_xy.shape[0]), int(inner_xy.shape[0]))
        all_xy = np.concatenate([src_bnd_xy, inner_xy], axis=0)
        if all_xy.shape[0] < 3:
            return (np.zeros((0, 3), dtype=np.int64),
                    int(src_bnd_xy.shape[0]), int(inner_xy.shape[0]))
        try:
            tri = _scipy_delaunay(all_xy)
        except Exception as exc:
            self._log(f"Glue: Delaunay упал ({exc}).")
            return (np.zeros((0, 3), dtype=np.int64),
                    int(src_bnd_xy.shape[0]), int(inner_xy.shape[0]))
        s = tri.simplices.astype(np.int64)
        n_src_bnd = int(src_bnd_xy.shape[0])
        # is_src[t, c] — True если c-я вершина t-го треугольника лежит в src.
        is_src = s < n_src_bnd
        # bridge-треугольник: и src, и inner есть.
        n_src_per_tri = is_src.sum(axis=1)
        bridge = (n_src_per_tri >= 1) & (n_src_per_tri <= 2)
        # Фильтр по длине src→inner-рёбер: если любое из них > thr, выкидываем.
        # src-src и inner-inner рёбра не проверяем (длинное src-src — норма
        # для bridge, inner-inner и так короткое — регулярная сетка).
        p = all_xy[s]
        thr = float(cell_size_m) * float(max_edge_factor)
        keep = bridge.copy()
        # Накапливаем длины ВСЕХ src→inner рёбер bridge-треугольников
        # (до фильтра) — для диагностики «почему остаются разрывы».
        bridge_edge_lens_all = []
        for (a, b) in ((0, 1), (1, 2), (2, 0)):
            is_bridge_edge = is_src[:, a] != is_src[:, b]
            elen = np.linalg.norm(p[:, a] - p[:, b], axis=1)
            keep &= ~(is_bridge_edge & (elen > thr))
            mask_be = is_bridge_edge & bridge
            if mask_be.any():
                bridge_edge_lens_all.append(elen[mask_be])
        # Диагностика: статистика длин стыковочных рёбер и сколько отсечено.
        try:
            n_bridge_total = int(bridge.sum())
            n_kept = int(keep.sum())
            n_cut = n_bridge_total - n_kept
            if bridge_edge_lens_all:
                lens = np.concatenate(bridge_edge_lens_all)
                med = float(np.median(lens))
                mx = float(lens.max())
                q90 = float(np.quantile(lens, 0.9))
                over = int((lens > thr).sum())
            else:
                med = mx = q90 = 0.0
                over = 0
            # Сколько inner-ring узлов остались без хотя бы одного bridge'а.
            inner_used = np.unique(s[keep][s[keep] >= n_src_bnd]) - n_src_bnd
            n_inner_total = int(inner_xy.shape[0])
            n_inner_orphan = n_inner_total - int(inner_used.shape[0])
            self._log(
                f"Glue diag: bridges {n_bridge_total} (kept {n_kept}, "
                f"cut {n_cut} по порогу {thr*100:.1f} см); src→inner edge "
                f"медиана {med*100:.1f} см, q90 {q90*100:.1f} см, "
                f"max {mx*100:.1f} см, > порога {over}; inner-ring "
                f"без bridge'а: {n_inner_orphan}/{n_inner_total}.")
            # Глубокий диагноз orphan'ов: где они в XY и насколько далеко
            # от ближайшей src-side точки. По этим цифрам видно, нужно
            # densify (orphan'ы у src-границы) или другая стратегия
            # (orphan'ы вообще не рядом с source-мешем).
            if n_inner_orphan > 0:
                orphan_mask = np.ones(n_inner_total, dtype=bool)
                orphan_mask[inner_used] = False
                orphan_xy = inner_xy[orphan_mask]
                try:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(all_xy[:n_src_bnd])
                    d_orphan, _ = tree.query(orphan_xy, k=1)
                    om_med = float(np.median(d_orphan))
                    om_q50 = om_med
                    om_q90 = float(np.quantile(d_orphan, 0.9))
                    om_q99 = float(np.quantile(d_orphan, 0.99))
                    om_max = float(d_orphan.max())
                    om_min = float(d_orphan.min())
                    # XY bbox of orphan cluster.
                    bx0, by0 = orphan_xy[:, 0].min(), orphan_xy[:, 1].min()
                    bx1, by1 = orphan_xy[:, 0].max(), orphan_xy[:, 1].max()
                    self._log(
                        f"Glue orphan диагностика: расстояние до ближайшей "
                        f"src-side точки — min={om_min*100:.1f} см, "
                        f"med={om_med*100:.1f} см, q90={om_q90*100:.1f} см, "
                        f"q99={om_q99*100:.1f} см, max={om_max*100:.1f} см; "
                        f"orphan XY bbox=({bx0:.2f},{by0:.2f})..({bx1:.2f},"
                        f"{by1:.2f}) м.")
                except ImportError:
                    pass
        except Exception as exc:
            self._log(f"Glue diag упал: {exc}")
        return s[keep], n_src_bnd, int(inner_xy.shape[0])

    # Buffer-зона: число итераций Laplacian-relax'а Z movable-вершин loop'а
    # ПЕРЕД триангуляцией. Z fixed-вершин (source-меш + densify-extras)
    # держится неизменным, movable (= ext-сетка, в т.ч. rim-lift'нутые
    # orphan inner-ring) сглаживаются по соседям loop'а — это убирает
    # «треугольные гребни» между высоким Z source'а (~2.9 м) и низким Z
    # ext'а после descent cap'а (~1.9 м), которые иначе торчат вертикально
    # на дистанции ~30 см. Чем больше итераций, тем плавнее переход (но
    # тем сильнее loop вне «иголки» проседает). 5 — мягкий баланс. Эффект
    # виден только когда loop содержит и fixed, и movable вершины (mixed
    # seam); для чисто-src / чисто-ext дыр relax = no-op.
    SEAM_RELAX_ITERS = 5
    # Center-vertex fan: Z новой центральной вершины =
    #   min(loop_z) + SEAM_CENTER_LIFT_FRAC × (max(loop_z) − min(loop_z))
    # 0.0 → центр строго на минимуме → buffer-треугольники гарантированно
    #       «вдавлены», без «купольных мостов» которые fan-from-vertex-0
    #       клал поверх впадин на вершинах холмов.
    # 0.5 → середина (нейтрально, может слегка выпирать).
    # 1.0 → на максимуме (= old fan, явный купол).
    # Дыры на вершинах холма выглядят как лёгкая «ямочка» — заметно
    # меньше визуально, чем «козырёк». Дыры в плоских областях (range≈0)
    # фильтр не задевает (центр совпадает с уровнем loop'а).
    SEAM_CENTER_LIFT_FRAC = 0.0

    def _stitch_seam_holes(self, verts, faces, is_fixed=None):
        """Buffer-зона: найти ВСЕ замкнутые boundary-loop'ы в combined-меше
        (source + ext + glue), оставить один внешний (= TARGET-прямоугольник,
        самый большой по XY bbox), и каждую внутреннюю дыру триангулировать
        buffer-треугольниками.

        is_fixed: (N,) bool. Если задан, перед триангуляцией каждой дыры
            Z movable-вершин loop'а сглаживается Laplacian-relax'ом
            (SEAM_RELAX_ITERS итераций) — fixed-вершины служат якорями.
            Это убирает «треугольные гребни» возникающие, когда rim-lift
            «поднял» orphan узлы к высокому Z source'а, а glue не построил
            bridge — orphan остался иголкой посреди низкого ext'а.

        Зачем: даже после rim-lift'а и densify-glue может остаться маленькое
        число orphan inner-ring узлов (длинное стыковочное ребро отфильтровано
        порогом cell_size×EXTRAP_GLUE_MAX_EDGE_FACTOR) — они формируют
        треугольные «окна» в меше. Этот шаг ПОЛНОСТЬЮ аддитивен по топологии:
        ни одна старая грань не удаляется, дыры закрываются сверху. Z
        movable-вершин может слегка сдвинуться вниз (см. Laplacian-relax).

        Алгоритм:
          1. Naked-рёбра (cnt==1) → undirected adjacency-граф.
          2. Walk loop'ов: для каждого непосещённого naked-ребра идём по
             соседям, пока не вернёмся в старт. При degree>2 (T-junction)
             выбираем «левый поворот» — это правильный face-tracing для
             planar graph и закрывает loop'ы, которые наивный first-unvisited
             walk пропустит.
          3. Второй проход: если после walk'а остались непосещённые naked-
             рёбра, запускаем walk ещё раз с пустым visited-set'ом (на тех
             же рёбрах). Подбирает loop'ы, на которых первый проход
             свернул не туда.
          4. Внешний loop = с самой большой XY-bbox площадью
             (TARGET-прямоугольник всегда крупнее любого orphan-окна).
          5. Laplacian-relax Z movable-вершин каждой внутренней дыры.
          6. Триангуляция → _triangulate_loop_xy.
          7. Ориентируем добавленные треугольники с +Z-нормалью.

        Возвращает (verts_maybe_relaxed, faces_with_buffer). Старые faces
        сохраняются байт-в-байт; verts может содержать обновлённые Z
        movable-вершин."""
        verts = np.asarray(verts, dtype=np.float64).copy()
        faces = np.asarray(faces, dtype=np.int64)
        if faces.shape[0] == 0:
            return verts, faces

        # 1. Naked-рёбра: ровно одна грань на каждой стороне меша.
        edges = np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
        edges_sorted = np.sort(edges, axis=1)
        uniq, _, cnt = np.unique(
            edges_sorted, axis=0, return_inverse=True, return_counts=True)
        naked = uniq[cnt == 1]
        if naked.shape[0] < 3:
            self._log("Buffer-зона: меш уже замкнут (<3 голых рёбер).")
            return verts, faces

        # 2. Undirected adjacency с сохранением XY-координат — для
        # «left-turn»-walk'а в degree>2 ситуациях нужна геометрия.
        adj: dict[int, list[int]] = {}
        for ab in naked:
            a, b = int(ab[0]), int(ab[1])
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        # 3. Walk loop'ов — два прохода. Первый покрывает большинство;
        # второй пытается закрыть дыры, пропущенные из-за неправильного
        # выбора в T-junction'ах.
        def _walk_loops(naked_edges, ignore_visited=False):
            visited_local: set = set()
            loops_local: list[list[int]] = []
            for ab in naked_edges:
                a0, b0 = int(ab[0]), int(ab[1])
                key0 = frozenset((a0, b0))
                if key0 in visited_local:
                    continue
                loop = [a0, b0]
                visited_local.add(key0)
                prev, cur = a0, b0
                closed = False
                for _ in range(naked_edges.shape[0] * 2 + 4):
                    nxt = self._pick_left_turn(
                        verts, prev, cur, adj.get(cur, ()), visited_local)
                    if nxt is None:
                        break
                    visited_local.add(frozenset((cur, nxt)))
                    if nxt == a0:
                        closed = True
                        break
                    loop.append(nxt)
                    prev, cur = cur, nxt
                if closed and len(loop) >= 3:
                    loops_local.append(loop)
            return loops_local, visited_local

        loops, visited = _walk_loops(naked)
        n_pass1 = len(loops)

        # Второй проход на оставшихся naked-рёбрах (не вошли ни в один
        # закрытый loop первого прохода).
        n_pass2 = 0
        remaining_edges = np.asarray(
            [e for e in naked if frozenset((int(e[0]), int(e[1])))
             not in visited],
            dtype=np.int64)
        if remaining_edges.shape[0] >= 3:
            loops2, _ = _walk_loops(remaining_edges)
            loops.extend(loops2)
            n_pass2 = len(loops2)

        if not loops:
            self._log(
                f"Buffer-зона: walk не закрыл ни одного loop'а из "
                f"{naked.shape[0]} голых рёбер — пропуск.")
            return verts, faces

        # 4. Внешний контур = самый большой по XY bbox площади. TARGET-
        # прямоугольник (4×7 м, центр в napolnitel) заведомо крупнее
        # любого orphan-окна вдоль шва (которое порядка cell_size_m).
        def _bbox_area(loop_idx):
            pts = verts[np.asarray(loop_idx, dtype=np.int64)]
            dx = float(pts[:, 0].max() - pts[:, 0].min())
            dy = float(pts[:, 1].max() - pts[:, 1].min())
            return dx * dy
        bboxes = [_bbox_area(l) for l in loops]
        outer_idx = int(np.argmax(bboxes))
        interior = [l for i, l in enumerate(loops) if i != outer_idx]

        if not interior:
            self._log(
                f"Buffer-зона: дыр нет ({len(loops)} loop, только внешний "
                f"контур площадью {bboxes[outer_idx]:.2f} м²).")
            return verts, faces

        # 4b. Диагностика типа loop'ов (mixed/pure-src/pure-ext) — для
        # понимания распределения, не для фильтрации. Фильтр по типу
        # оказался неработоспособным: walk loop'ов через T-junction'ы
        # систематически разделяет шов на pure-src и pure-ext куски, и
        # mixed-loop'ов почти никогда не образуется (см. логи: 117 src +
        # 56 ext + 0 mixed). Поэтому закрываем ВСЕ interior loop'ы, но
        # триангуляцией center-vertex fan с Z = min(loop_z), чтобы buffer
        # гарантированно «вдавлялся», а не выпирал «куполом» поверх дыры.
        is_fixed_arr = (np.asarray(is_fixed, dtype=bool)
                        if is_fixed is not None else None)
        n_pure_src = n_pure_ext = n_mixed = 0
        if is_fixed_arr is not None:
            for loop in interior:
                la = np.asarray(loop, dtype=np.int64)
                any_fixed = bool(is_fixed_arr[la].any())
                any_mov = bool((~is_fixed_arr[la]).any())
                if any_fixed and any_mov:
                    n_mixed += 1
                elif any_fixed:
                    n_pure_src += 1
                else:
                    n_pure_ext += 1

        # 5. Laplacian-relax Z movable-вершин mixed-дыр. Применяется
        # ПЕРЕД триангуляцией, чтобы buffer-треугольники получили уже
        # плавный Z вдоль loop'а вместо «иголок» от rim-lift'нутых
        # orphan'ов. Для pure-src / pure-ext loop'ов relax = no-op.
        relax_iters = int(self.SEAM_RELAX_ITERS)
        max_delta = 0.0
        n_moved = 0
        if is_fixed_arr is not None and relax_iters > 0:
            for loop in interior:
                d, m = self._relax_loop_z(
                    verts, np.asarray(loop, dtype=np.int64),
                    is_fixed_arr, relax_iters)
                max_delta = max(max_delta, d)
                n_moved += m

        # 6. Триангулируем каждую дыру center-vertex fan'ом. Возвращает
        # extra-вершины (по одной центральной на каждую дыру с n>3) и
        # faces в local-индексах: [0, n_loop) → loop_arr, [n_loop, end)
        # → новые вершины. Делаем global remap, ориентируем +Z, собираем.
        extra_verts_chunks: list[np.ndarray] = []
        new_tris_list: list[np.ndarray] = []
        n_done = 0
        n_failed = 0
        sizes: list[int] = []
        for loop in interior:
            extra, faces_local = self._triangulate_loop_xy(verts, loop)
            if faces_local.shape[0] == 0:
                n_failed += 1
                continue
            n_loop = len(loop)
            loop_arr = np.asarray(loop, dtype=np.int64)
            # Global base index for these new extra vertices = current
            # verts size + size of extras уже накопленных.
            base_new = (int(verts.shape[0])
                        + sum(int(e.shape[0]) for e in extra_verts_chunks))
            # Remap: local idx < n_loop → loop_arr[idx],
            #        local idx >= n_loop → base_new + (idx - n_loop).
            fl = faces_local.astype(np.int64)
            mapped = np.where(
                fl < n_loop,
                loop_arr[np.clip(fl, 0, n_loop - 1)],
                base_new + (fl - n_loop))
            if extra.shape[0] > 0:
                extra_verts_chunks.append(extra)
            new_tris_list.append(mapped)
            n_done += 1
            sizes.append(n_loop)

        if not new_tris_list:
            self._log(
                f"Buffer-зона: {len(interior)} дыр найдено, ни одна не "
                f"зашита (triangulate упал на всех).")
            return verts, faces

        # 7. Concat extra-вершин В verts, затем ориентируем + concat faces.
        if extra_verts_chunks:
            verts = np.concatenate(
                [verts, np.concatenate(extra_verts_chunks, axis=0)], axis=0)
        new_tris = np.concatenate(new_tris_list, axis=0).astype(np.int64)
        # +Z orientation: cross((v1-v0), (v2-v0)).z > 0 — если меньше,
        # переставляем v1↔v2. Источник/ext уже имеют +Z нормали.
        v0 = verts[new_tris[:, 0]]
        v1 = verts[new_tris[:, 1]]
        v2 = verts[new_tris[:, 2]]
        nrm = np.cross(v1 - v0, v2 - v0)
        flip = nrm[:, 2] < 0.0
        if flip.any():
            new_tris[flip] = new_tris[flip][:, [0, 2, 1]]
        all_faces = np.concatenate([faces, new_tris], axis=0)

        sz_min = min(sizes) if sizes else 0
        sz_max = max(sizes) if sizes else 0
        sz_med = int(np.median(sizes)) if sizes else 0
        n_extra_v = sum(int(e.shape[0]) for e in extra_verts_chunks)
        relax_log = (
            f" relax: {n_moved} верш. сдвинуто (|δ|max={max_delta*100:.1f} см)"
            if is_fixed_arr is not None and relax_iters > 0 else "")
        type_log = (
            f" типы: {n_mixed} mixed + {n_pure_src} pure-src + "
            f"{n_pure_ext} pure-ext;"
            if is_fixed_arr is not None else "")
        self._log(
            f"Buffer-зона: {len(loops)} boundary-loop'ов "
            f"(walk pass1={n_pass1} + pass2={n_pass2}; "
            f"1 внешний {bboxes[outer_idx]:.2f} м², "
            f"{len(interior)} внутренних дыр {sz_min}…{sz_max} верш., "
            f"медиана {sz_med});{type_log} "
            f"зашито {n_done} center-vertex fan'ом "
            f"(+{n_extra_v} центр-вершин, "
            f"+{int(new_tris.shape[0])} треугольников; "
            f"Z центров = min(loop_z) + "
            f"{self.SEAM_CENTER_LIFT_FRAC:.2f}×range — "
            f"вдавлено, без купольных мостов).{relax_log}")
        return verts, all_faces

    @staticmethod
    def _pick_left_turn(verts, prev, cur, neighbors, visited):
        """В face-tracing planar graph: при degree>2 в вершине `cur`,
        пришедшей из `prev`, выбираем соседа с МАКСИМАЛЬНЫМ поворотом
        налево (CCW) от направления prev→cur. Это правильный «следующий»
        в обходе грани (по аналогии с алгоритмом face-tracing через
        rotational order вокруг вершины).

        Игнорируем уже посещённые рёбра и сам `prev`. Возвращает int|None.
        Если соседей-кандидатов нет — None (loop оборвался)."""
        cands = []
        for n in neighbors:
            if n == prev:
                continue
            if frozenset((int(cur), int(n))) in visited:
                continue
            cands.append(int(n))
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        # Несколько кандидатов: вычисляем signed angle turn (prev→cur→n)
        # в XY-плоскости, берём кандидата с самым большим положительным
        # углом (наибольший поворот налево = CCW).
        p_prev = verts[int(prev), :2]
        p_cur = verts[int(cur), :2]
        v_in = p_cur - p_prev
        in_len = float(np.linalg.norm(v_in))
        if in_len < 1e-12:
            return cands[0]    # degenerate — пусть будет первый
        best_ang = -math.inf
        best_n = cands[0]
        for n in cands:
            p_n = verts[n, :2]
            v_out = p_n - p_cur
            out_len = float(np.linalg.norm(v_out))
            if out_len < 1e-12:
                continue
            cross = float(v_in[0] * v_out[1] - v_in[1] * v_out[0])
            dot = float(v_in[0] * v_out[0] + v_in[1] * v_out[1])
            ang = math.atan2(cross, dot)   # ∈ (-π, π]; >0 = CCW (left)
            if ang > best_ang:
                best_ang = ang
                best_n = n
        return best_n

    @staticmethod
    def _relax_loop_z(verts, loop_arr, is_fixed, iters):
        """Laplacian-relax по Z вершин loop'а: на каждой итерации каждый
        movable-узел получает Z = среднее двух соседей по loop'у; fixed-
        узлы держатся неизменными как Dirichlet BC. Сходится за O(n)
        итераций к линейной интерполяции между fixed-якорями.

        Мутирует verts[loop_arr, 2] in-place для movable-узлов.
        Возвращает (max_|δZ|, n_moved). Loop без fixed-узлов или без
        movable-узлов — no-op."""
        n = int(loop_arr.shape[0])
        if n < 3 or iters <= 0:
            return 0.0, 0
        mov = ~is_fixed[loop_arr]
        if not mov.any() or mov.all():
            return 0.0, 0
        z = verts[loop_arr, 2].copy()
        z0 = z.copy()
        for _ in range(int(iters)):
            z_left = np.roll(z, 1)
            z_right = np.roll(z, -1)
            z_avg = 0.5 * (z_left + z_right)
            z = np.where(mov, z_avg, z0)
        delta = z - z0
        if mov.any():
            verts[loop_arr[mov], 2] = z[mov]
            return float(np.abs(delta[mov]).max()), int(mov.sum())
        return 0.0, 0

    def _triangulate_loop_xy(self, verts, loop):
        """Center-vertex fan triangulation замкнутого полигона. Возвращает:
          • extra_verts: (M, 3) float64 — новые вершины, M ∈ {0, 1}
              (1 центральная для n>=4; ноль для n==3).
          • faces_local: (T, 3) int64 — индексы:
              [0, n_loop)     → loop_arr[idx]    (исходные вершины loop'а)
              [n_loop, n_loop+M) → extra_verts[idx − n_loop]  (новые)

        Почему center-vertex, а не fan-from-vertex-0:
          • fan-from-0 создаёт длинные «мостовые» треугольники между
            vertex 0 и удалёнными vertex'ами (diameter loop'а). Если loop
            окружает впадину/гребень с переменным Z, такой треугольник
            выпирает над/под настоящей поверхностью «куполом» — visible
            artifact на вершинах холмов.
          • center-vertex fan даёт n коротких треугольников через центр,
            каждое ребро длиной ~radius (вдвое короче). Z центральной
            вершины = min(loop_z) + SEAM_CENTER_LIFT_FRAC × range. С
            SEAM_CENTER_LIFT_FRAC = 0.0 центр строго на минимуме — buffer
            гарантированно вдавлен, без купольных мостов.

        Для невыпуклых loop'ов center-vertex fan может дать треугольники
        с edge'ами через outside; для типичных seam-дыр (3-30 верш.)
        они почти выпуклые и это не критично."""
        loop_arr = np.asarray(loop, dtype=np.int64)
        n = int(loop_arr.shape[0])
        if n < 3:
            return (np.zeros((0, 3), dtype=np.float64),
                    np.zeros((0, 3), dtype=np.int64))
        # Тривиальный случай: один треугольник, без новой вершины.
        if n == 3:
            return (np.zeros((0, 3), dtype=np.float64),
                    np.array([[0, 1, 2]], dtype=np.int64))

        loop_xyz = verts[loop_arr]
        # Защита от вырожденности (collinear loop) — fan-tri будут нулевой
        # площади, пропускаем дыру.
        if (float(np.ptp(loop_xyz[:, 0])) < 1e-9 or
                float(np.ptp(loop_xyz[:, 1])) < 1e-9):
            return (np.zeros((0, 3), dtype=np.float64),
                    np.zeros((0, 3), dtype=np.int64))

        cx = float(loop_xyz[:, 0].mean())
        cy = float(loop_xyz[:, 1].mean())
        z_min = float(loop_xyz[:, 2].min())
        z_max = float(loop_xyz[:, 2].max())
        cz = z_min + float(self.SEAM_CENTER_LIFT_FRAC) * (z_max - z_min)
        extra_verts = np.array([[cx, cy, cz]], dtype=np.float64)

        # Fan: triangle i = (loop_arr[i], loop_arr[(i+1)%n], center).
        # В local-индексах: (i, (i+1)%n, n) — центр стоит на позиции n
        # после loop'а.
        idx_a = np.arange(n, dtype=np.int64)
        idx_b = np.mod(np.arange(1, n + 1, dtype=np.int64), n)
        idx_c = np.full(n, n, dtype=np.int64)
        faces_local = np.stack([idx_a, idx_b, idx_c], axis=1)
        return extra_verts, faces_local

    @staticmethod
    def _points_in_polygon_xy(pts, poly):
        """Ray-casting point-in-polygon, векторизовано по pts. `poly` —
        (M, 2) вершины полигона в порядке обхода (CW или CCW не важно).
        Возвращает (N,) bool. Точки строго на ребре дают неопределённый
        результат — для наших centroid'ов это не критично."""
        pts = np.asarray(pts, dtype=np.float64)
        poly = np.asarray(poly, dtype=np.float64)
        m = poly.shape[0]
        if m < 3 or pts.shape[0] == 0:
            return np.zeros(pts.shape[0], dtype=bool)
        px = pts[:, 0]
        py = pts[:, 1]
        inside = np.zeros(pts.shape[0], dtype=bool)
        for i in range(m):
            j = (i - 1) % m
            xi, yi = float(poly[i, 0]), float(poly[i, 1])
            xj, yj = float(poly[j, 0]), float(poly[j, 1])
            cond1 = (yi > py) != (yj > py)
            dy = yj - yi
            safe = dy if abs(dy) > 1e-12 else 1e-12
            x_int = xi + (py - yi) * (xj - xi) / safe
            cond2 = px < x_int
            inside ^= (cond1 & cond2)
        return inside

    @staticmethod
    def _smooth_preserve(Z, preserve, iters):
        """3×3 box-smooth, который НЕ трогает preserve-ячейки (anchor),
        но использует их как соседей. Применяется к синтезированным
        ячейкам, чтобы загладить только швы между патчами, не размывая
        ни источник, ни сами бугры."""
        Zc = np.asarray(Z, dtype=np.float64).copy()
        pres = np.asarray(preserve, dtype=bool)
        Hh, Ww = Zc.shape
        for _ in range(int(iters)):
            acc = np.zeros_like(Zc)
            wsum = np.zeros_like(Zc)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ys_d, ye_d = max(0, -dy), Hh - max(0, dy)
                    xs_d, xe_d = max(0, -dx), Ww - max(0, dx)
                    ys_s, ye_s = max(0, dy), Hh - max(0, -dy)
                    xs_s, xe_s = max(0, dx), Ww - max(0, -dx)
                    acc[ys_d:ye_d, xs_d:xe_d] += Zc[ys_s:ye_s, xs_s:xe_s]
                    wsum[ys_d:ye_d, xs_d:xe_d] += 1.0
            new = acc / np.maximum(wsum, 1e-12)
            Zc = np.where(pres, Zc, new)
        return Zc

    def _quilt_z(self, Z_src, V_src, Z_tgt, V_tgt,
                 patch_frac=0.40, overlap_frac=0.30,
                 candidates=24, seed=0):
        """Image-quilting (Efros & Freeman) на ПОЛНОЙ высотной карте.

        V_tgt — anchor-ячейки (их Z уже выставлен в Z_tgt и не меняется).
        Остальные ячейки заполняются source-патчами в порядке близости
        к anchor'у (BFS), на каждой позиции выбирается патч из
        `candidates` случайных кандидатов: побеждает тот, у которого
        SSD по уже-расставленному перекрытию минимально. Шов между
        патчами сглаживается линейным feather'ом шириной overlap.

        Никакого приведения к floor_z, никакой собственной модификации
        Z за пределами замен. Возвращает плотный Z (tny, tnx)."""
        Z_src = np.asarray(Z_src, dtype=np.float64)
        V_src = np.asarray(V_src, dtype=bool)
        Z_tgt = np.asarray(Z_tgt, dtype=np.float64).copy()
        V_tgt = np.asarray(V_tgt, dtype=bool)
        sy, sx = Z_src.shape
        Ty, Tx = Z_tgt.shape

        # Patch / overlap size in cells.
        P = max(8, int(round(min(sy, sx) * float(patch_frac))))
        P = min(P, sy, sx)
        O = max(2, int(round(P * float(overlap_frac))))
        O = min(O, P - 1)
        step = max(1, P - O)

        rng = np.random.default_rng(seed)

        # Pre-compute valid source patch top-left starts.
        if sy < P or sx < P:
            self._log("Quilt: источник мельче patch'а — пропуск.")
            # Заполнить пустые ячейки средним по источнику, чтобы не
            # оставлять NaN.
            mean_z = float(Z_src[V_src].mean()) if V_src.any() else 0.0
            return np.where(np.isnan(Z_tgt), mean_z, Z_tgt)

        valid_starts = []
        for sy0 in range(sy - P + 1):
            for sx0 in range(sx - P + 1):
                if V_src[sy0:sy0 + P, sx0:sx0 + P].all():
                    valid_starts.append((sy0, sx0))
        if not valid_starts:
            self._log("Quilt: нет полностью-валидных source-патчей — пропуск.")
            mean_z = float(Z_src[V_src].mean()) if V_src.any() else 0.0
            return np.where(np.isnan(Z_tgt), mean_z, Z_tgt)
        valid_starts = np.asarray(valid_starts, dtype=np.int64)

        # Feather ramp: 1 в центре, плавно ↓ к 0 на ширине O по краям.
        # При overlay'е этот ramp используется только на перекрывающихся
        # ячейках; α≈0 на самом краю означает «держим старое», α≈1 в
        # глубине плитки — «берём новое».
        ramp = np.ones((P, P), dtype=np.float64)
        for k in range(O):
            w_k = (k + 1) / (O + 1)
            ramp[k, :] = np.minimum(ramp[k, :], w_k)
            ramp[P - 1 - k, :] = np.minimum(ramp[P - 1 - k, :], w_k)
            ramp[:, k] = np.minimum(ramp[:, k], w_k)
            ramp[:, P - 1 - k] = np.minimum(ramp[:, P - 1 - k], w_k)

        # Tile top-left positions covering the target.
        tile_ti = list(range(0, Ty, step))
        if tile_ti[-1] + P < Ty:
            tile_ti.append(Ty - P)        # ensure target edge is covered
        tile_tj = list(range(0, Tx, step))
        if tile_tj[-1] + P < Tx:
            tile_tj.append(Tx - P)
        tile_positions = [(ti, tj) for ti in tile_ti for tj in tile_tj]

        # Order tiles by BFS-distance from anchor: process the most-
        # anchored tiles first → каждый следующий патч уже имеет богатую
        # «опору» из уже-расставленных соседей и хорошо склеивается.
        _, _, dist_cells = self._bfs_seeds(V_tgt)

        def tile_dist(pos):
            ti, tj = pos
            i1 = min(ti + P, Ty); j1 = min(tj + P, Tx)
            d = dist_cells[ti:i1, tj:j1]
            d = d[np.isfinite(d)]
            return float(d.min()) if d.size else float("inf")

        tile_positions.sort(key=tile_dist)

        placed = V_tgt.copy()
        out = Z_tgt.copy()
        # На незаполненных ячейках Z_tgt — NaN; заменим временно 0,
        # чтобы арифметика не отравлялась (anchor — настоящие Z).
        out = np.where(np.isnan(out), 0.0, out)

        n_tiles = 0
        for (ti, tj) in tile_positions:
            i0, j0 = ti, tj
            i1 = min(i0 + P, Ty); j1 = min(j0 + P, Tx)
            ph, pw = i1 - i0, j1 - j0
            if ph < 2 or pw < 2:
                continue
            tile_placed = placed[i0:i1, j0:j1]
            if tile_placed.all():
                continue            # уже целиком заполнен — пропуск

            cur_tile = out[i0:i1, j0:j1]
            anchor_count = int(tile_placed.sum())

            # Кандидаты-патчи. Если ничего ещё не расставлено в зоне
            # плитки — берём случайный source-патч; иначе ищем минимум
            # SSD на перекрытии.
            n_cand = min(int(candidates), valid_starts.shape[0])
            idxs = rng.integers(0, valid_starts.shape[0], size=n_cand)
            best_score = np.inf
            best_patch = None
            for k in idxs:
                sy0, sx0 = valid_starts[k]
                sp = Z_src[sy0:sy0 + ph, sx0:sx0 + pw]
                if anchor_count > 0:
                    diff = sp[tile_placed] - cur_tile[tile_placed]
                    score = float((diff * diff).mean())
                else:
                    score = float(rng.random())   # любой кандидат подойдёт
                if score < best_score:
                    best_score = score
                    best_patch = sp
            if best_patch is None:
                continue

            # 1) В «новых» (ещё-незаполненных) ячейках патч кладётся целиком.
            new_mask = ~tile_placed
            out[i0:i1, j0:j1][new_mask] = best_patch[new_mask]
            # 2) В «перекрытии» — feather-blend: α = ramp.
            if anchor_count > 0:
                r = ramp[:ph, :pw]
                a = r[tile_placed]
                old = cur_tile[tile_placed]
                new = best_patch[tile_placed]
                out[i0:i1, j0:j1][tile_placed] = a * new + (1.0 - a) * old
            placed[i0:i1, j0:j1] = True
            n_tiles += 1

        # На случай если что-то осталось NaN (теоретически не должно).
        if not placed.all():
            mean_z = float(Z_src[V_src].mean()) if V_src.any() else 0.0
            out = np.where(placed, out, mean_z)

        self._log(f"Quilt Z: положено {n_tiles} плиток размера "
                  f"{P}×{P} (overlap {O}, step {step}) из "
                  f"{valid_starts.shape[0]} допустимых позиций источника, "
                  f"BFS-порядок от anchor'а.")
        return out

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
