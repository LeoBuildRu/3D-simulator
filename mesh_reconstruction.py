# mesh_reconstruction.py
import tkinter as tk
from tkinter import filedialog
from PyQt5 import *
from PyQt5.QtWidgets import *
from panda3d.core import *
import os
import json
import numpy as np
import math
from PIL import Image
import noise
import random
from scipy.spatial import KDTree
import trimesh
import point_cloud_utils as pcu

import cProfile
import pstats
import io
    
class MeshReconstruction:
    def __init__(self, panda_app):
        self.panda_app = panda_app
        self.recon_json_path = ""

        # for testing only
        self.alpha_threshold = 0.5

        # Параметры экстраполяции
        self.extrapolation_enabled = False
        self.target_width = 15.0  # Ширина целевой области в метрах
        self.target_height = 10.0  # Высота целевой области в метрах
        self.grid_resolution = 80  # Разрешение сетки экстраполяции
        
        # Параметры шума Перлина
        self.noise_scale = 0.5
        self.noise_strength = 0.15
        self.noise_octaves = 4
        self.noise_persistence = 0.5
        self.noise_lacunarity = 2.0
        self.noise_seed = random.randint(0, 10000)
        
        # Для хранения мешей
        self.source_mesh_node = None
        self.mesh_node = None
        
        print(f"[DEBUG] Инициализирован MeshReconstruction")

    def browse_recon_json(self):
        file_path = filedialog.askopenfilename(
            title="Select .json config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        self.recon_json_path = file_path

        if file_path:
            return file_path
        return None
        
    def normalize(self, v):
        v = np.array(v, dtype=float)
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v

    def build_local_to_world_matrix_np(self, A, B, C):
        A, B, C = map(np.array, (A, B, C))
        x = self.normalize(B - A)
        tempY = C - A
        proj = np.dot(tempY, x) * x
        y = self.normalize(tempY - proj)
        z = self.normalize(np.cross(x, y))
        m = np.eye(4)
        m[:3, 0] = x
        m[:3, 1] = y
        m[:3, 2] = z
        m[:3, 3] = A
        return m

    def compute_transform_np(self, p1A, p1B, p1C, p2A, p2B, p2C):
        # Pack points into arrays for easier indexing
        tri1 = np.array([p1A, p1B, p1C])  # shape (3, 3)
        tri2 = np.array([p2A, p2B, p2C])  # shape (3, 3)

        # Find which vertex in tri1 is the right-angle vertex
        right_angle_idx = self._find_right_angle_vertex(tri1)

        # Reorder both triangles so that the right-angle vertex is at index 0
        if right_angle_idx != 0:
            # Swap index 0 with right_angle_idx in both triangles
            tri1[[0, right_angle_idx]] = tri1[[right_angle_idx, 0]]
            tri2[[0, right_angle_idx]] = tri2[[right_angle_idx, 0]]

        # Proceed with transform computation
        world_from_local1 = self.build_local_to_world_matrix_np(tri1[0], tri1[1], tri1[2])
        world_from_local2 = self.build_local_to_world_matrix_np(tri2[0], tri2[1], tri2[2])
        return world_from_local1 @ np.linalg.inv(world_from_local2)


    def _find_right_angle_vertex(self, tri):
        """
        Given a triangle with 3 points (3x3 array), find the index of the vertex
        that is (approximately) the right angle by checking which side is longest
        (opposite the right angle).
        
        Returns:
            int: index (0, 1, or 2) of the right-angle vertex.
        """
        # Compute squared distances between all pairs
        d01 = np.sum((tri[0] - tri[1])**2)
        d02 = np.sum((tri[0] - tri[2])**2)
        d12 = np.sum((tri[1] - tri[2])**2)

        # The longest side is opposite the right angle
        if d12 >= d02 and d12 >= d01:
            return 0
        elif d02 >= d12 and d02 >= d01:
            return 1
        else:
            return 2

    def panda_vec3_to_np(self, v):
        return np.array([v.x, v.y, v.z], dtype=float)

    def np_to_panda_point(self, v):
        return LPoint3f(float(v[0]), float(v[1]), float(v[2]))

    def np_to_panda_vec(self, v):
        return LVector3f(float(v[0]), float(v[1]), float(v[2]))

    def viewport_to_world_point(
        self,
        camera,
        u, v,
        distance
    ):
        lens = self.cam_node.getLens()
        nearPoint = Point3()
        farPoint = Point3()
        film_point = Point2((u - 0.5) * 2, (0.5 - v) * 2)
        lens.extrude(film_point, nearPoint, farPoint)     
        fromvec = Vec3(farPoint - nearPoint)   
        fromvec.normalize()

        # mult = 1 / fromvec[1]
        # for i in range(3):
        #     fromvec[i] = fromvec[i] * mult

        dir_world = self.panda_app.render.get_relative_vector(camera, fromvec)

        cam_pos = camera.get_pos()
        world_point = cam_pos + dir_world * distance

        return world_point
    

    def correct_depth(self, depth):
        return depth
        return math.sqrt(depth) 

    
    def map_value(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    def setVertFOV(self, lens, vfov_deg):
        # Convert VFOV → HFOV
        hfov_rad = 2.0 * math.atan(math.tan(math.radians(vfov_deg) / 2.0) * self.aspect_ratio)
        hfov_deg = math.degrees(hfov_rad)

        self.fov_y = vfov_deg

        lens.setFov(hfov_deg, vfov_deg)

    def apply_transform(self, M, camera):
        cam_pos = self.panda_vec3_to_np(camera.get_pos(camera.get_parent()))
        old_pos = cam_pos
        new_pos = M @ np.append(old_pos, 1.0)
        camera.set_pos(self.np_to_panda_point(new_pos[:3]))

        current_mat = camera.get_net_transform().get_mat()
        world_right = self.panda_vec3_to_np(current_mat.xform_vec(LVector3f(1, 0, 0)))
        world_up = self.panda_vec3_to_np(current_mat.xform_vec(LVector3f(0, 1, 0)))
        world_forward = self.panda_vec3_to_np(current_mat.xform_vec(LVector3f(0, 0, 1)))

        R = M[:3, :3]
        new_right = R @ world_right
        new_up = R @ world_up
        new_forward = R @ world_forward

        new_right /= np.linalg.norm(new_right)
        new_up /= np.linalg.norm(new_up)
        new_forward /= np.linalg.norm(new_forward)

        mat3 = Mat3()
        mat3.set_row(0, self.np_to_panda_vec(new_right))
        mat3.set_row(1, self.np_to_panda_vec(new_up))
        mat3.set_row(2, self.np_to_panda_vec(new_forward))

        quat = LQuaternionf()
        quat.set_from_matrix(mat3)

        camera.set_quat(quat)

    def resolve_keypoints(self, scene_3d, points_2d, camera, min_depth, max_depth, debug):
        proj_3d = []

        for i in range(3):
            u = points_2d[i][0]
            v = points_2d[i][1]
            
            x = points_2d[i][2]
            y = points_2d[i][3]
            
            depth = self.correct_depth(self.height_map[y, x])
            z = depth * (max_depth - min_depth) + min_depth

            point = self.viewport_to_world_point(camera, u, v, z)
            proj_3d.append(self.panda_vec3_to_np(point))

        M = self.compute_transform_np(
            scene_3d[0], scene_3d[1], scene_3d[2],
            proj_3d[0], proj_3d[1], proj_3d[2]
        )

        self.apply_transform(M, camera)

        error = 0

        for i in range(self.matching_points):
            u = points_2d[i][0]
            v = points_2d[i][1]

            x = points_2d[i][2]
            y = points_2d[i][3]
            
            depth = self.correct_depth(self.height_map[y, x])
            z = depth * (max_depth - min_depth) + min_depth

            point = self.viewport_to_world_point(camera, u, v, z)
            diff = point - scene_3d[i]
            distance = math.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            if(debug):
                self.add_cube_at(point)
            error += distance

        return error

    def lerp(self, a, b, t):
        return a + t * (b - a)
    
    def add_cube_at(self, pos, color=(1, 0, 0, 1), scale=0.10):
        box = self.panda_app.loader.load_model("models/smiley")
        box.set_pos(pos)
        box.set_scale(scale)
        box.reparent_to(self.panda_app.render)
        return box
    
    def cv_to_panda(self, p):
        """
        Convert point from OpenCV-style camera coordinates
        (X right, Y down, Z forward)
        to Panda3D camera coordinates
        (X right, Y forward, Z up)
        """
        x, y, z = p
        return LVector3f(
            x,   # right -> right
            z,   # forward -> forward
            -y   # up (inverted Y) -> up
        )


    def reconstruct_camera_pos_hpr_fov_depth(self, data):
        key_points = data["keypoints"]
        camera = self.panda_app.camera

        lens = self.panda_app.cam.node().getLens()

        img_w = data["metadata"]["image_size"]["width"]
        img_h = data["metadata"]["image_size"]["height"]

        self.aspect_ratio = img_w/img_h
        
        lerpT = 1.0 / 2

        iterations = 24

        scene_3d = [np.array(p, dtype=float) for p in data["points_3d"]]

        min_scale = 0.0001
        max_scale = 1000

        best_scale_error = 10**10
        best_scale = 1

        if "keypoints_3d" in data:
            print("Using 3d keypoints")

            #cam_pos = self.panda_vec3_to_np(camera.get_pos(camera.get_parent()))
            #quat = camera.get_quat()

            def ApplyScale(scale):
                #camera.set_pos(self.np_to_panda_point(cam_pos))
                #camera.set_quat(quat)

                transformed_keypoints = keypoints_3d.copy()
                for i in range(len(keypoints_3d)):
                    transformed_keypoints[i] = self.panda_app.render.get_relative_point(camera, keypoints_3d[i] * scale)

                M = self.compute_transform_np(
                    scene_3d[0], scene_3d[1], scene_3d[2],
                    transformed_keypoints[0], transformed_keypoints[1], transformed_keypoints[2]
                )

                self.apply_transform(M, camera)

                error = 0
                for i in range(len(transformed_keypoints)):
                    op = self.panda_app.render.get_relative_point(camera, keypoints_3d[i] * scale)
                    sp = scene_3d[i]
                    diff = self.np_to_panda_point(op - sp)
                    error += math.sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z)
                return error

            keypoints_3d = [np.array(p, dtype=float) for p in data["keypoints_3d"]]

            for i in range(len(keypoints_3d)):
                keypoints_3d[i] = self.cv_to_panda(self.np_to_panda_point(keypoints_3d[i]))

            #hardcoded for now
            #self.aspect_ratio = 4/3
            self.setVertFOV(lens, 50.6)

            scaleStart = min_scale
            scaleEnd = max_scale

            for i in range(iterations):
                scaleStep = (scaleEnd - scaleStart) / 2
                
                if scaleStep == 0:
                    break

                scale = scaleStart - scaleStep
                while scale < scaleEnd:
                    scale += scaleStep
                    error = ApplyScale(scale)
                    if error < best_scale_error:
                        best_scale_error = error
                        best_scale = scale

                scaleStart = self.lerp(best_scale, scaleStart, lerpT)
                scaleEnd = self.lerp(best_scale, scaleEnd, lerpT)

            ApplyScale(best_scale)
            print(f"Best error: {best_scale_error} at {best_scale}")
            return

        self.fov_y = lens.getFov()[1]

        points_2d = []

        for i in range(len(key_points)):
            x = key_points[i]["x"]
            y = key_points[i]["y"]

            u = x / img_w
            v = y / img_h

            points_2d.append([u, v, x, y])

        # 99-55 для стационарного решения
        # 0-41 для блендера
        fov_known = True
        known_fov_x = 0
        known_fov_y = 41.1

        self.matching_points = 4

        min_fov = 15
        max_fov = 130
        fov_acc = 1.0

        start_min_depth = 1
        start_max_depth = 200

        bestError = 10**10

        bestMinDepth = 0
        bestMaxDepth = 0
        bestFOV = 0

        render_depth_known = True

        # из блендера
        render_min_depth = 3
        render_max_depth = 6
        render_camera_offset = 5.35774
        render_mesh_offset = 5.4816

        render_camera_offset -= render_mesh_offset

        render_min_depth += render_camera_offset
        render_max_depth += render_camera_offset

        ratio = render_min_depth / render_max_depth

        fov = min_fov - fov_acc
        while fov < max_fov:
            fov += fov_acc
            self.fov_y = fov

            if fov_known:
                if(known_fov_x != 0):
                    lens.setFov(known_fov_x, known_fov_y)
                    self.fov_y = known_fov_y
                else:
                    self.setVertFOV(lens, known_fov_y)
            else:
                self.setVertFOV(lens, fov)

            scaleStart = start_min_depth
            scaleEnd = start_max_depth
            localMaxDepthStart = start_min_depth
            localMaxDepthEnd = start_max_depth

            localBest = 10**10

            for i in range(iterations):
                scaleStep = (localMaxDepthEnd - localMaxDepthStart) / 2
                maxDepthStep = (localMaxDepthEnd - localMaxDepthStart) / 2

                if scaleStep == 0 or maxDepthStep == 0:
                    break
                
                scale = localMaxDepthStart - maxDepthStep
                while scale < localMaxDepthEnd:
                    scale += maxDepthStep
                    min_depth = scaleStart - scaleStep
                    while min_depth < scaleEnd:
                        min_depth += scaleStep

                        if(render_depth_known):
                            min_depth = scale * ratio

                        try:
                            error = self.resolve_keypoints(scene_3d, points_2d, camera, min_depth, scale, False)
                            if(error < localBest):
                                localBest = error
                            if(error < bestError):
                                bestError = error
                                bestFOV = fov
                                bestMinDepth = min_depth
                                bestMaxDepth = scale
                        except:
                            continue

                        if(render_depth_known):
                            break
                
                scaleStart = self.lerp(bestMinDepth, scaleStart, lerpT)
                scaleEnd = self.lerp(bestMinDepth, scaleEnd, lerpT)
                localMaxDepthStart = self.lerp(bestMaxDepth, localMaxDepthStart, lerpT)
                localMaxDepthEnd = self.lerp(bestMaxDepth, localMaxDepthEnd, lerpT)
            
            if fov_known:
                break

            print(f"local best for FOV {fov} is {localBest}")


        self.min_depth = bestMinDepth
        self.max_depth = bestMaxDepth + (bestMaxDepth - bestMinDepth) * 0.3 # чтобы насыпь сходилась с дном, некий магический коэффицент коррекции

        fov = bestFOV
        self.fov_y = fov

        if(known_fov_x != 0):
            lens.setFov(known_fov_x, known_fov_y)
            self.fov_y = known_fov_y
        else:
            self.setVertFOV(lens, known_fov_y)
    
        print(f"camera keypoints final error: {bestError} FOV: {self.fov_y} minDepth: {bestMinDepth} maxDepth: {bestMaxDepth}")
        
        self.resolve_keypoints(scene_3d, points_2d, camera, bestMinDepth, bestMaxDepth, False)

    def load_height_map(self):
        try:
            print(f"[DEBUG] Начало загрузки height map: {self.heightmap_path}")
            img = Image.open(self.heightmap_path)
            print(f"[DEBUG] Изображение загружено: {img.size}, mode: {img.mode}")
            
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                print(f"[DEBUG] Изображение содержит альфа-канал")
                img_rgba = img.convert('RGBA')
                r, g, b, a = img_rgba.split()
                
                alpha = np.array(a, dtype=np.float32) / 255.0
                
                if img.mode != 'L':
                    img_gray = img.convert('L')
                    height_data = np.array(img_gray, dtype=np.float32) / 255.0
                else:
                    height_data = np.array(img, dtype=np.float32) / 255.0
                
                mask = alpha > self.alpha_threshold
                
                self.height_map = height_data * mask
                self.mask = mask  
            else:
                print(f"[DEBUG] Изображение без альфа-канала")
                img_gray = img.convert('L')
                self.height_map = np.array(img_gray, dtype=np.float32) / 255.0
                
                mask = self.height_map > 0.05
                self.mask = mask
            
            # Закрываем изображение для освобождения памяти
            img.close()
            print(f"[DEBUG] Загрузка height map завершена успешно")
            return True
            
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки height map: {e}")
            return False
        
    # ---- fast normal calculator (no numpy) ----
    def _calculate_normals(self, vdata, geom):
        """
        Fast vertex-normal calculation.
        Expects triangles. Decomposes primitives to ensure direct vertex list access.
        """
        # Readers/writers
        vreader = GeomVertexReader(vdata, "vertex")
        nwriter = GeomVertexWriter(vdata, "normal")

        # Read all vertex positions into a list of tuples
        vertices = []
        while not vreader.isAtEnd():
            p = vreader.getData3f()
            vertices.append((p.x, p.y, p.z))

        n_verts = len(vertices)
        # mutable per-vertex normal accumulators
        normals = [[0.0, 0.0, 0.0] for _ in range(n_verts)]

        # Walk all primitives, decompose() to get simple triangle list
        for i in range(geom.getNumPrimitives()):
            prim = geom.getPrimitive(i)
            prim = prim.decompose()  # returns a simple primitive list (triangles)
            # iterate vertices in groups of 3
            nv = prim.getNumVertices()
            j = 0
            while j < nv:
                vi0 = prim.getVertex(j)
                vi1 = prim.getVertex(j + 1)
                vi2 = prim.getVertex(j + 2)
                x0, y0, z0 = vertices[vi0]
                x1, y1, z1 = vertices[vi1]
                x2, y2, z2 = vertices[vi2]

                # edges
                e1x = x1 - x0; e1y = y1 - y0; e1z = z1 - z0
                e2x = x2 - x0; e2y = y2 - y0; e2z = z2 - z0

                # cross product
                nx = e1y * e2z - e1z * e2y
                ny = e1z * e2x - e1x * e2z
                nz = e1x * e2y - e1y * e2x

                # normalize face normal (skip degenerate)
                mag2 = nx*nx + ny*ny + nz*nz
                if mag2 > 0.0:
                    inv_len = 1.0 / (mag2 ** 0.5)
                    nx *= inv_len; ny *= inv_len; nz *= inv_len

                    # accumulate to vertex normals
                    normals[vi0][0] += nx; normals[vi0][1] += ny; normals[vi0][2] += nz
                    normals[vi1][0] += nx; normals[vi1][1] += ny; normals[vi1][2] += nz
                    normals[vi2][0] += nx; normals[vi2][1] += ny; normals[vi2][2] += nz

                j += 3

        # Write normalized normals back into vdata (use writer row-by-row)
        # Reset row pointer for writer (it begins at row 0 by default if newly created)
        nwriter.setRow(0)
        for nx, ny, nz in normals:
            mag2 = nx*nx + ny*ny + nz*nz
            if mag2 > 0.0:
                inv_len = 1.0 / (mag2 ** 0.5)
                nwriter.setData3f(nx * inv_len, ny * inv_len, nz * inv_len)
            else:
                # fallback normal
                nwriter.setData3f(0.0, 0.0, 1.0)


    # ---- optimized mesh builder (calls _calculate_normals) ----
    def create_unified_perlin_mesh_with_lift(self):
        print("[DEBUG] Загрузка height map...")
        if not self.load_height_map():
            print("[ERROR] Не удалось загрузить height map")
            return

        height_map = self.height_map
        mask = self.mask
        correct_depth = self.correct_depth
        viewport_to_world_point = self.viewport_to_world_point
        camera = self.panda_app.camera

        h, w = height_map.shape
        inv_w = 1.0 / w
        inv_h = 1.0 / h
        min_depth = self.min_depth
        depth_range = self.max_depth - min_depth

        fmt = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("unified_perlin_mesh_with_lift", fmt, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")

        # Preallocate writers' row growth (optional)
        # vdata.setNumRows(h * w)  # sometimes useful, but depends on Panda3D version

        # VERTICES
        for y in range(h):
            v = y * inv_h
            for x in range(w):
                u = x * inv_w

                z = correct_depth(height_map[y, x])
                distance = z * depth_range + min_depth

                point = viewport_to_world_point(camera, u, v, distance)

                vertex.addData3f(point)
                normal.addData3f(0.0, 0.0, 1.0)   # placeholder
                texcoord.addData2f(u, v)

        # TRIANGLES (index math: idx = y*w + x)
        tris = GeomTriangles(Geom.UHStatic)
        for y in range(h - 1):
            row = y * w
            next_row = (y + 1) * w
            for x in range(w - 1):
                # early mask check for shared edges
                if not mask[y + 1, x] or not mask[y, x + 1]:
                    continue

                v1 = row + x
                v2 = row + x + 1
                v3 = next_row + x
                v4 = next_row + x + 1

                if mask[y, x]:
                    tris.addVertices(v1, v3, v2)

                if mask[y + 1, x + 1]:
                    tris.addVertices(v2, v3, v4)

        tris.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(tris)

        # Compute normals using the Python routine (fast, in-place)
        self._calculate_normals(vdata, geom)

        node = GeomNode("unified_perlin_mesh_with_lift")
        node.addGeom(geom)
        return node


    def add_extended_mesh_to_scene(self, node):
        if hasattr(self, "mesh_node") and self.mesh_node:
            self.mesh_node.removeNode()
        
        if node:
            self.mesh_node = self.panda_app.render.attachNewNode(node)
            
            material = Material()
            material.setDiffuse((0.8, 0.8, 0.8, 1.0))
            material.setAmbient((0.3, 0.3, 0.3, 1.0))
            material.setSpecular((0.5, 0.5, 0.5, 1.0))
            material.setShininess(50.0)
            self.mesh_node.setTwoSided(True)
            self.mesh_node.setMaterial(material, 1)
            
            self.mesh_node.setShaderAuto()
            self.mesh_node.setPos(0, 0, 0)
            
            if not hasattr(self.panda_app, 'loaded_models'):
                self.panda_app.loaded_models = []
            if not hasattr(self.panda_app, 'model_paths'):
                self.panda_app.model_paths = {}
            
            if self.mesh_node not in self.panda_app.loaded_models:
                self.panda_app.loaded_models.append(self.mesh_node)
                self.panda_app.model_paths[id(self.mesh_node)] = "extended_height_map_mesh"
            
            return self.mesh_node
        
        return None

    def run_2d_to_3d_reconstruction(self):
        json_path = self.recon_json_path
        if not json_path or not os.path.isfile(json_path):
            #QMessageBox.warning(self.panda_app, "Ошибка", "Пожалуйста, выберите корректный JSON-файл.")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.heightmap_path = os.path.dirname(json_path) + "/" + data["metadata"]["image_path"].replace("corrected_", "height_map_").replace(".jpg", ".png")
        self.ply_path =  json_path.replace("input-", "output-").replace(".json", ".ply")
        self.point_cloud = pcu.load_mesh_v(self.ply_path)
        #print(len(self.point_cloud))
        # Загружаем height_map до реконструкции камеры
        if not self.load_height_map(): 
            return
        
        self.cam_node = self.panda_app.cam.node()

        self.reconstruct_camera_pos_hpr_fov_depth(data)

        self.heightmap_path = os.path.dirname(json_path) + "/" + data["metadata"]["mask_path"].replace("corrected_", "height_map_").replace(".jpg", ".png")

        pr = cProfile.Profile()
        pr.enable()
        
        node = self.create_unified_perlin_mesh_with_lift()
        self.add_extended_mesh_to_scene(node)

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
        ps.print_stats(30)   # top 30
        print(s.getvalue())


        