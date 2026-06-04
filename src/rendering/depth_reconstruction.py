# depth_reconstruction.py
# ---------------------------------------------------------------------------
# Local fill-landscape reconstruction from a stand snapshot's depth map.
#
# Flow (driven from the UI):
#   1. The user picks a stand snapshot and lines the live camera up with the
#      reference overlay so the 3D world matches the photo.
#   2. They toggle "point picking" and click the 4 corners of the truck bed
#      (кузов) in the viewport.
#   3. Each clicked screen point is ray-cast against the loaded truck model to
#      recover its true 3D position (4 metric anchor points).
#   4. The depth map (linear, 8-bit grayscale) is calibrated to metric using
#      the 4 anchors — a least-squares fit  Z = A*d + B  maps the normalised
#      depth value d∈[0,1] to a forward (perpendicular) distance from the
#      camera. The fit recovers min/max depth and is sign-agnostic, so it
#      doesn't matter whether "near" is bright or dark.
#   5. A grid is sampled inside the bed quad; every sample is unprojected
#      through the current camera lens at its calibrated depth, giving a 3D
#      surface. That surface is turned into a Panda mesh and added to the
#      scene, anchored to the current camera state.
#
# Assumptions (documented so they're easy to revisit):
#   A1  The 4 points are the bed corners; the mesh covers the region INSIDE
#       that quad (the fill surface), not the whole frame.
#   A2  The depth PNG is linear and normalised by /255.
#   A3  "depth" = forward/perpendicular distance from the camera plane
#       (Panda camera looks down +Y), consistent with linear depth.
#   A4  The photo fills the film rectangle 1:1, i.e. depth-pixel (u,v) maps
#       to film coords fx=2u-1, fy=1-2v. Matching the camera to the overlay
#       (shown KeepAspectRatio) makes this hold when the viewport aspect ≈
#       the photo aspect (1670x942 ≈ 16:9).
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
    Vec3, Material, BitMask32,
    GeomVertexFormat, GeomVertexData, Geom, GeomTriangles, GeomVertexWriter,
)


class DepthReconstructor:
    """Owns the N-point picking interaction and the depth->mesh pipeline."""

    # Any number of points is allowed; this many are required to form a
    # region + calibrate. Picking is finished by the user (RMB / Esc).
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
    # Maximum allowed 3D length (world units / metres) of a mesh triangle's
    # edge. Cells whose neighbouring vertices are farther apart than this are
    # dropped — this culls the long, stretched polygons that bridge depth
    # discontinuities (e.g. fill surface -> background). Tune to taste.
    MAX_EDGE_LEN = 0.7
    # Maximum allowed VERTICAL (global +Z) extent of a triangle edge, metres.
    # Near-vertical polygons (the bed walls / стенки кузова) have a big Δz per
    # edge and get culled, while flat/gently-sloped fill is kept.
    MAX_VERTICAL_DROP = 2.0
    # Clear stray polygons around sharp depth discontinuities: a depth pixel
    # whose value spans more than JUMP_THRESH_M (metres) within a
    # JUMP_RADIUS_PX-pixel window is treated as "near a jump", and any mesh
    # sample landing there is dropped. This removes the floating fragments
    # that appear in the blurred/compressed transition band of a big jump.
    JUMP_RADIUS_PX = 2
    JUMP_THRESH_M = 1.0
    # Light surface smoothing: number of 3x3 mask-aware averaging passes over
    # the grid depths before unprojection. Evens out the terracing caused by
    # 8-bit depth quantization. 0 = off; 1-3 = gentle. Only averages valid
    # (kept) neighbours, so it doesn't bleed across mask/jump boundaries.
    SMOOTH_ITERS = 2
    # Per-snapshot fill mask (<depth>-mask.png, RGBA). When USE_MASKS is on and
    # a mask exists for the snapshot, the mesh is clipped ONLY by the mask
    # (alpha > MASK_ALPHA_MIN) and the geometric cullings above are skipped.
    # If USE_MASKS is on but the mask is missing — or USE_MASKS is off — the
    # geometric cullings (edge length / verticality / jump radius) are used.
    USE_MASKS = True
    MASK_SUFFIX = "-mask.png"
    MASK_ALPHA_MIN = 0.5
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
        self._films: list[tuple[float, float]] = []   # clicked film coords
        self._hits: list[Point3] = []                 # raycast world points
        # Film coords of the last successful pick — reused to auto-reconstruct
        # other snapshots from the SAME fixed stand camera (the bed corners
        # sit at the same screen positions, only the depth map differs).
        self._saved_films: list[tuple[float, float]] = []

        self._mesh_node = None                        # last reconstruction
        self._truck_np = None                         # cached pick target
        self._truck_collider = None                   # CollisionPolygon node
        self._collider_truck_id = None                # whose collider we built

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

    def start_picking(self) -> None:
        if self._picking:
            return
        if not self._depth_path or not os.path.exists(self._depth_path):
            self._log("Нет карты глубины для реконструкции.")
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
        n = len(self._films)
        self.stop_picking()
        if n >= self.MIN_POINTS:
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

        # --- Region = convex hull of the picked points (film space) -------
        hull = self._convex_hull(list(zip(cfx.tolist(), cfy.tolist())))
        if len(hull) < 3:
            self._log("Точки вырождены (нет площади) — нечего реконструировать.")
            self._finish(False, {})
            return False

        # Grid over the hull's bounding box; mask to the hull.
        G = self.GRID
        stride = G + 1
        xs = cfx.min() + (cfx.max() - cfx.min()) * (np.arange(stride) / G)
        ys = cfy.min() + (cfy.max() - cfy.min()) * (np.arange(stride) / G)
        FX, FY = np.meshgrid(xs, ys)          # (stride, stride)
        FX = FX.ravel()
        FY = FY.ravel()
        inside = self._inside_hull_mask(hull, FX, FY)

        # Depth value + locally-corrected metric depth at every grid sample.
        cols, rows = self._film_to_pixel_vec(FX, FY, W, H, fyh)
        d_grid = depth[rows, cols]
        resid = self._idw_residual(FX, FY, cfx, cfy, r_ctrl)
        Z_grid = A * d_grid + B + resid

        # Decide the clipping strategy:
        #   USE_MASKS + mask present  -> use ONLY the snapshot's alpha mask.
        #   USE_MASKS + mask missing  -> fall back to the geometric cullings.
        #   USE_MASKS off             -> geometric cullings.
        mask_alpha = self._load_mask_alpha(W, H) if self.USE_MASKS else None
        use_mask = mask_alpha is not None

        if use_mask:
            keep = mask_alpha[rows, cols] > self.MASK_ALPHA_MIN
            n_before = int(inside.sum())
            inside = inside & keep
            self._log(f"Маска: оставлено {int(inside.sum())} из {n_before} "
                      f"узлов (alpha > {self.MASK_ALPHA_MIN:.0%}); "
                      f"отсечения по длине/вертикали отключены.")
        else:
            # Drop samples in the blurred transition band around a sharp jump.
            forbidden = self._jump_forbidden_mask(depth, A)
            if forbidden is not None:
                near = forbidden[rows, cols]
                n_before = int(inside.sum())
                inside = inside & (~near)
                n_cut = n_before - int(inside.sum())
                if n_cut > 0:
                    self._log(f"У разрывов глубины убрано {n_cut} узлов "
                              f"(радиус {self.JUMP_RADIUS_PX}px, "
                              f"порог {self.JUMP_THRESH_M} м).")

        # Gentle smoothing of the (kept) grid depths to kill the 8-bit
        # quantization terracing before the surface is built.
        if self.SMOOTH_ITERS > 0:
            Z_grid = self._smooth_grid_z(
                Z_grid.reshape(stride, stride),
                inside.reshape(stride, stride),
                self.SMOOTH_ITERS).ravel()

        # Unproject the in-hull samples through the live lens.
        verts = np.zeros((FX.shape[0], 3), dtype=np.float64)
        for k in range(FX.shape[0]):
            if not inside[k]:
                continue
            wp = self._unproject(lens, cam_np, render,
                                 float(FX[k]), float(FY[k]), float(Z_grid[k]))
            verts[k, 0] = wp.x
            verts[k, 1] = wp.y
            verts[k, 2] = wp.z

        # Build triangles. With a mask, no geometric culling (inf limits);
        # otherwise cull by edge length and verticality.
        if use_mask:
            me, mv = float("inf"), float("inf")
        else:
            me, mv = self.MAX_EDGE_LEN, self.MAX_VERTICAL_DROP
        faces, dropped = self._build_grid_faces(G, inside, verts, me, mv)
        if len(faces) == 0:
            self._log("Нет ячеек для меша (узкая область / маска пуста / всё "
                      "отсечено).")
            self._finish(False, {})
            return False
        if dropped and not use_mask:
            self._log(f"Отсечено {dropped} полигонов (длина > "
                      f"{self.MAX_EDGE_LEN} м или вертикаль > "
                      f"{self.MAX_VERTICAL_DROP} м).")
        z_min = float(Z_grid[inside].min())
        z_max = float(Z_grid[inside].max())

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

        info = {"A": A, "B": B, "z_min": z_min, "z_max": z_max,
                "points": int(n_pts),
                "verts": int(verts.shape[0]), "faces": int(len(faces))}
        self._log(f"✅ Реконструкция готова по {n_pts} точкам: "
                  f"{info['faces']} треугольников.")
        self._finish(True, info)
        return True

    # ------------------------------------------------------------------
    def has_saved_points(self) -> bool:
        return len(self._saved_films) >= self.MIN_POINTS

    def clear_saved_points(self) -> None:
        self._saved_films = []

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

    def _vertical_film_span(self, W_img, H_img) -> float:
        """Half-extent (in film fy units) that the photo occupies vertically:
        fyh = window_aspect / photo_aspect. The photo fills the film width,
        so vertically it spans fy in [-fyh, +fyh]."""
        try:
            win = self.panda_app.win
            aw = float(win.get_x_size()) / float(max(1, win.get_y_size()))
        except Exception:
            aw = float(W_img) / float(max(1, H_img))
        ap = float(W_img) / float(max(1, H_img))
        if ap <= 1e-9:
            return 1.0
        return aw / ap

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
    def _convex_hull(points):
        """Andrew's monotone-chain convex hull. Returns hull vertices CCW."""
        pts = sorted(set((round(float(p[0]), 9), round(float(p[1]), 9))
                         for p in points))
        if len(pts) <= 2:
            return pts

        def cross(o, a, b):
            return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    @staticmethod
    def _inside_hull_mask(hull, X, Y):
        """Vectorised point-in-convex-polygon test (CCW hull)."""
        inside = np.ones(X.shape, dtype=bool)
        n = len(hull)
        for k in range(n):
            ax, ay = hull[k]
            bx, by = hull[(k + 1) % n]
            cr = (bx - ax) * (Y - ay) - (by - ay) * (X - ax)
            inside &= (cr >= -1e-9)
        return inside

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
    def _build_grid_faces(G, inside, verts, max_edge, max_vdrop):
        """Vectorised grid triangulation. A cell becomes 2 triangles only if
        all 4 corners are in-hull AND, for every triangle edge, the 3D length
        is <= max_edge AND the vertical (world +Z) drop is <= max_vdrop.

        Returns (faces (T,3) int64, dropped_count)."""
        stride = G + 1
        V = verts.reshape(stride, stride, 3)
        ins = inside.reshape(stride, stride)

        # Edge vectors between neighbouring grid vertices.
        eH = V[:, 1:, :] - V[:, :-1, :]    # (stride, G, 3) horizontal
        eV = V[1:, :, :] - V[:-1, :, :]    # (G, stride, 3) vertical
        eD = V[1:, 1:, :] - V[:-1, :-1, :]  # (G, G, 3) diagonal
        me = float(max_edge)
        mv = float(max_vdrop)

        # Per-edge acceptance: short enough AND not too vertical.
        okH = (np.linalg.norm(eH, axis=2) <= me) & (np.abs(eH[..., 2]) <= mv)
        okV = (np.linalg.norm(eV, axis=2) <= me) & (np.abs(eV[..., 2]) <= mv)
        okD = (np.linalg.norm(eD, axis=2) <= me) & (np.abs(eD[..., 2]) <= mv)

        ti = np.arange(G)[:, None]
        si = np.arange(G)[None, :]
        a = ti * stride + si          # top-left corner index of each cell
        b = a + 1                     # top-right
        c = a + stride                # bottom-left
        d = c + 1                     # bottom-right

        cell_in = (ins[:-1, :-1] & ins[:-1, 1:] &
                   ins[1:, :-1] & ins[1:, 1:])    # (G, G)

        # Triangle 1 (a,b,d): edges a-b, b-d, a-d.
        t1 = cell_in & okH[:-1, :] & okV[:, 1:] & okD
        # Triangle 2 (a,d,c): edges a-d, d-c, a-c.
        t2 = cell_in & okD & okH[1:, :] & okV[:, :-1]

        f1 = np.stack([a[t1], b[t1], d[t1]], axis=1) if t1.any() \
            else np.zeros((0, 3), np.int64)
        f2 = np.stack([a[t2], d[t2], c[t2]], axis=1) if t2.any() \
            else np.zeros((0, 3), np.int64)
        faces = np.concatenate([f1, f2], axis=0).astype(np.int64)

        # In-hull triangles rejected (by edge length or verticality).
        dropped = int(2 * int(cell_in.sum()) - faces.shape[0])
        return faces, dropped

    def _jump_forbidden_mask(self, depth, A):
        """Boolean HxW mask: True where the depth varies by more than
        JUMP_THRESH_M (metric) within a JUMP_RADIUS_PX window — i.e. near a
        sharp discontinuity (plus its blurred transition band)."""
        R = int(self.JUMP_RADIUS_PX)
        if R <= 0 or abs(float(A)) < 1e-9:
            return None
        rng = self._window_range(np.asarray(depth, dtype=np.float32), R)
        return (abs(float(A)) * rng) > float(self.JUMP_THRESH_M)

    @staticmethod
    def _window_range(a, R):
        """Per-pixel (max - min) over a (2R+1)x(2R+1) box, border-replicated.
        Separable: max/min along rows then columns."""
        def shift(arr, s, ax):
            out = np.empty_like(arr)
            if ax == 0:
                if s > 0:
                    out[s:, :] = arr[:-s, :]; out[:s, :] = arr[:1, :]
                else:
                    out[:s, :] = arr[-s:, :]; out[s:, :] = arr[-1:, :]
            else:
                if s > 0:
                    out[:, s:] = arr[:, :-s]; out[:, :s] = arr[:, :1]
                else:
                    out[:, :s] = arr[:, -s:]; out[:, s:] = arr[:, -1:]
            return out

        mx = a.copy()
        mn = a.copy()
        for ax in (0, 1):
            amx = mx.copy()
            amn = mn.copy()
            for s in range(1, R + 1):
                amx = np.maximum(amx, shift(mx, s, ax))
                amx = np.maximum(amx, shift(mx, -s, ax))
                amn = np.minimum(amn, shift(mn, s, ax))
                amn = np.minimum(amn, shift(mn, -s, ax))
            mx, mn = amx, amn
        return mx - mn

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
