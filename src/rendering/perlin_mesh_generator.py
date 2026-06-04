import sys
import os
import random
import numpy as np

from panda3d.core import (
    Geom, GeomNode, GeomVertexData, GeomVertexFormat, GeomVertexWriter,
    GeomTriangles, NodePath, TextureStage, Texture, Material, Filename,
)


def _texture_filename(path: str) -> Filename:
    """
    Безопасно превратить произвольный путь к файлу текстуры
    (Windows `C:\\...\\diff.jpg`, POSIX-абсолютный или относительный)
    в Panda3D-Filename. Без этого `loader.loadTexture("C:\\...")`
    падает с "Could not load texture", потому что конструктор Filename
    из строки трактует её как уже-Panda-стилевой путь (Unix-слэши).
    """
    return Filename.fromOsSpecific(str(path))

from src.core.TLS_client import TLS_client


class PerlinMeshGenerator:
    """Генератор ландшафта + выполнение boolean"""

    def __init__(self, panda_app, tls_client=None,
                 server_host='78.25.191.12', server_port=9998):
        self.panda_app = panda_app
        self.gui = panda_app.gui

        if tls_client is not None:
            self.tls_client = tls_client
        else:
            self.tls_client = TLS_client(host=server_host, port=server_port)

    # ------------------------------------------------------------------
    # Генерация ландшафта через TLS-сервер (Blender ANT Landscape).
    # Если задан target_model_path, сервер сразу выполняет булеву разность
    # (target − landscape) и возвращает уже итоговый меш — отдельный вызов
    # send_boolean_request больше не нужен.
    # ------------------------------------------------------------------
    def generate_perlin_mesh(self, grid_size=48,
                             target_model_path=None,
                             landscape_offset_z=0.0,
                             target_volume=None):
        if self.gui:
            self.gui.log_message(
                "⏳ Запрос ландшафта (Blender ANT Landscape)..."
            )

        # 1. Размеры из последнего CSG-результата
        csg_info = self.panda_app.csg_results[-1]
        csg_node = csg_info["result_node"]
        pos = csg_node.getPos()
        min_bound, max_bound = csg_node.getTightBounds()

        if min_bound is None or max_bound is None:
            size_x = 10.0
            size_y = 10.0
            size_z = 1.0
        else:
            size_x = max_bound.x - min_bound.x
            size_y = max_bound.y - min_bound.y
            size_z = max_bound.z - min_bound.z

            project_root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ))
            models_dir = os.path.join(project_root, "assets", "models")
            max_volumes = {
                os.path.join(models_dir, "Scania-Napolnitel.gltf"): 24,
                os.path.join(models_dir, "Kamaz-Napolnitel.gltf"): 10,
                os.path.join(models_dir, "Hooper1-Napolnitel.gltf"): 48,
                os.path.join(models_dir, "Hooper2-Napolnitel.gltf"): 88,
            }
            max_coefficient = 48
            min_coefficient = 14
            # Локальный поиск пути к загруженной napolnitel-модели — нужен
            # только для подбора коэффициента из max_volumes ниже. Не путать
            # с параметром target_model_path, который уходит на сервер для
            # булевой разности (его не трогаем).
            local_napolnitel_path = None
            for model in self.panda_app.loaded_models:
                model_id = id(model)
                if model_id in self.panda_app.model_paths:
                    if (self.panda_app.Target_Napolnitel
                            in self.panda_app.model_paths[model_id]):
                        local_napolnitel_path = (
                            self.panda_app.model_paths[model_id]
                        )
                        break
            if local_napolnitel_path in max_volumes:
                max_volume = max_volumes[local_napolnitel_path]
                volume_ratio = self.panda_app.Target_Volume / max_volume
                coefficient = (max_coefficient
                               - (max_coefficient - min_coefficient)
                               * volume_ratio)
                coefficient = max(min_coefficient,
                                  min(max_coefficient, coefficient))
            else:
                coefficient = (max_coefficient + min_coefficient) / 2
            size_z = size_z * coefficient

        # 2. Параметры текстуры
        texture_repeatX = self.panda_app.current_texture_set.get(
            'textureRepeatX', 1.35
        )
        texture_repeatY = self.panda_app.current_texture_set.get(
            'textureRepeatY', 3.2
        )
        strength = self.panda_app.current_texture_set.get('strength', 0.14)
        seed = random.randint(0, 10000)

        # Displace-карта для серверного DISPLACE-модификатора.
        # Берём ОТНОСИТЕЛЬНЫЙ путь из сырого конфига текстурного набора —
        # в current_texture_set он уже подменён на путь локального кэша
        # (см. main_window._materialize_texture_set), а серверу нужен
        # путь относительно его g_textures_dir.
        disp_rel = None
        raw_set = getattr(self.panda_app, 'current_texture_set_raw', None)
        if isinstance(raw_set, dict):
            disp_rel = raw_set.get('displacement') or raw_set.get('height')

        # 3. Параметры Blender ANT Landscape
        tv = float(getattr(self.panda_app, "Target_Volume", 0.0) or 0.0)

        mv = None
        cur_set = getattr(self.panda_app, "current_model_set", None)
        if cur_set:
            try:
                from src.ui.panel_data import get_model_set_config as _gmsc
                _cfg = _gmsc(str(cur_set))
                if _cfg and _cfg.get("max_volume"):
                    mv = float(_cfg["max_volume"])
            except Exception:
                mv = None
        if not mv or mv <= 0:
            mv = max(tv * 1.5, 20.0)

        ratio = max(0.05, min(1.5, tv / mv if mv > 0 else 0.5))
        height_blender = 1.2

        noise_scale_blender = 1.36 + random.uniform(-0.30, 0.30)
        distortion_blender = 1.39 + random.uniform(-0.35, 0.35)

        subdivisions = 48
        mesh_size_x_blender = size_x
        mesh_size_y_blender = size_y
        depth_blender = 8
        edge_falloff = "3"
        edge_level = -0.12
        falloff_x = 3.70
        falloff_y = 4.00

        print(
            f"[Perlin] target_volume={tv:.2f}  max_volume={mv:.2f}  "
            f"ratio={ratio:.2f}  height={height_blender:.3f}  "
            f"noise_scale={noise_scale_blender:.3f}  "
            f"distortion={distortion_blender:.3f}  seed={seed}"
        )

        try:
            landscape_data = self.tls_client.generate_landscape(
                name="Landscape",
                subdivisions=subdivisions,
                subdivisions_x=subdivisions,
                subdivisions_y=subdivisions,
                mesh_size_x=mesh_size_x_blender,
                mesh_size_y=mesh_size_y_blender,
                size_x=1.45,
                size_y=2.23,
                noise_scale=noise_scale_blender,
                noise_type="rocks_noise",
                noise_basis="BLENDER",
                offset_x=-1.05,
                offset_y=0.0,
                height=height_blender,
                height_offset=0.0,
                depth=depth_blender,
                distortion=distortion_blender,
                hard_noise="0",
                maximum=10000.0,
                minimum=-10000.0,
                edge_falloff=edge_falloff,
                edge_level=edge_level,
                falloff_x=falloff_x,
                falloff_y=falloff_y,
                strata_type="0",
                seed=seed,
                texture_repeat_x=texture_repeatX,
                texture_repeat_y=texture_repeatY,
                output_format="ply",
                target_model_path=target_model_path,
                landscape_offset_z=landscape_offset_z,
                target_volume=target_volume,
                displacement_path=disp_rel,
                displacement_strength=strength,
            )
        except Exception as e:
            if self.gui:
                self.gui.log_message(
                    f"❌ Ошибка вызова generate_landscape: {e}"
                )
            raise RuntimeError(
                "Не удалось получить данные от сервера landscape"
            )

        if landscape_data is None:
            raise RuntimeError("Пустой ответ от сервера landscape")

        if len(landscape_data) != 4:
            raise RuntimeError(
                f"Некорректный формат ответа generate_landscape: "
                f"ожидалось 4 элемента, получено {len(landscape_data)}"
            )

        vertices_list, triangles, normals_list, texcoords_list = (
            landscape_data
        )

        if self.gui:
            self.gui.log_message(
                f"✓ Получено {len(vertices_list)} вершин, "
                f"{len(triangles)} треугольников"
            )

        # Если на сервере выполнялась булева разность, иногда она даёт
        # вырожденный результат (пустой/почти пустой меш). Раньше эту
        # проверку делал клиент на trimesh — оставляем её здесь.
        if target_model_path and (
            len(vertices_list) < 32 or len(triangles) < 16
        ):
            raise RuntimeError(
                f"Вырожденный результат булевой операции: "
                f"verts={len(vertices_list)}, tris={len(triangles)}"
            )

        # 4. Построение Panda3D-геометрии
        fmt = GeomVertexFormat.getV3n3t2()
        fmt = GeomVertexFormat.registerFormat(fmt)
        vdata = GeomVertexData("landscape_mesh", fmt, Geom.UHStatic)

        vertex_writer = GeomVertexWriter(vdata, "vertex")
        normal_writer = GeomVertexWriter(vdata, "normal")
        texcoord_writer = GeomVertexWriter(vdata, "texcoord")

        for i in range(len(vertices_list)):
            v = vertices_list[i]
            n = normals_list[i]
            t = texcoords_list[i]
            vertex_writer.addData3f(v[0], v[1], v[2])
            normal_writer.addData3f(n[0], n[1], n[2])
            texcoord_writer.addData2f(t[0], t[1])

        prim = GeomTriangles(Geom.UHStatic)
        for tri in triangles:
            prim.addVertices(int(tri[0]), int(tri[1]), int(tri[2]))
        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        geom_node = GeomNode("landscape_node")
        geom_node.addGeom(geom)
        perlin_np = NodePath(geom_node)

        if self.gui:
            self.gui.log_message("✓ Ландшафт успешно построен")

        return perlin_np

    # ------------------------------------------------------------------
    # Главный метод: запрашиваем у сервера ландшафт, который уже вычтен
    # из target-модели (булева разность выполняется на сервере внутри
    # /generate_landscape). Дальше — только текстуры, позиционирование
    # и расчёт объёма. Никаких локальных trimesh-операций больше нет.
    # ------------------------------------------------------------------
    def generate_perlin_mesh_from_csg(self):
        if self.gui:
            self.gui.log_message("⏳ Запуск Perlin mesh...")

        # --- очистка старых объектов ---
        for attr in ("final_mesh_node", "final_model"):
            obj = getattr(self.panda_app, attr, None)
            if obj is not None:
                if (attr == "final_model"
                        and obj in self.panda_app.loaded_models):
                    self.panda_app.loaded_models.remove(obj)
                obj.removeNode()
                setattr(self.panda_app, attr, None)

        # --- поиск target-модели в сцене (нужен только для последующего
        # hide(); сама геометрия для boolean берётся сервером с диска) ---
        target_model = None
        for model in self.panda_app.loaded_models:
            model_id = id(model)
            if model_id in self.panda_app.model_paths:
                if (self.panda_app.Target_Napolnitel
                        in self.panda_app.model_paths[model_id]):
                    target_model = model
                    break

        if target_model is None:
            print("Target-модель не найдена")
            return False

        if self.gui:
            self.gui.log_message("✓ Target-модель найдена")

        # --- путь к OBJ-наполнителю из конфига для серверной булевой ---
        target_model_path = self._get_target_model_path()
        if not target_model_path:
            if self.gui:
                self.gui.log_message(
                    "❌ В конфиге нет поля target_model для текущего набора"
                )
            return False

        # --- генерация landscape с булевой разностью на сервере ---
        if self.gui:
            self.gui.log_message(
                "⏳ Генерация ландшафта + boolean на сервере "
                "(grid_size=48)..."
            )
        # Z ландшафта подбирается на сервере под Target_Volume.
        # landscape_offset_z остаётся как fallback, если объёма нет.
        target_volume = float(
            getattr(self.panda_app, "Target_Volume", 0.0) or 0.0
        )
        try:
            final_np = self.generate_perlin_mesh(
                grid_size=48,
                target_model_path=target_model_path,
                landscape_offset_z=1.9375,
                target_volume=target_volume if target_volume > 0 else None,
            )
        except Exception as e:
            if self.gui:
                self.gui.log_message(
                    f"❌ Ошибка генерации/boolean на сервере: {e}"
                )
            return False
        if self.gui:
            self.gui.log_message("✓ Итоговый меш получен с сервера")

        # --- финальный меш = то, что пришло с сервера ---
        # generate_perlin_mesh возвращает NodePath без родителя (раньше он
        # шёл только в trimesh-конвертацию). Без reparentTo(render) Panda3D
        # его не рисует, отсюда «пустая сцена» при успешном boolean.
        self.panda_app.final_model = final_np
        self.panda_app.final_model.reparentTo(self.panda_app.render)
        if (self.panda_app.final_model
                not in self.panda_app.loaded_models):
            self.panda_app.loaded_models.append(self.panda_app.final_model)

        # --- текстуры и материал ---
        if self.gui:
            self.gui.log_message("⏳ Настройка текстур и материала...")
        self._apply_textures_and_material(self.panda_app.final_model)

        # --- ВЫВОД ОБЪЁМА В КОНСОЛЬ ---
        print(self.panda_app.calculate_mesh_volume(
            self.panda_app.final_model
        ))

        if self.gui:
            self.gui.log_message("✓ Текстуры применены")

        target_model.hide()

        # Меш приходит с сервера в Blender-координатах после auto-rotation
        # OBJ-импортёра: в Panda3D + render pipeline он оказывается
        # «носом кверху». P=+90 разворачивает обратно в горизонталь
        # с крышей вверх (P=-90 давал тот же поворот, но крышей вниз).
        # Если нос окажется направлен назад — добавить heading 180.
        self.panda_app.final_model.setHpr(0, 90, 0)

        if (hasattr(self.panda_app, "final_model")
                and self.panda_app.final_model):
            volume = self.panda_app.calculate_mesh_volume(
                self.panda_app.final_model
            )
            self.panda_app.update_overlay_info(volume=volume)

        if self.gui:
            self.gui.log_message("✓ Perlin mesh успешно сгенерирован!")

        return True

    # ------------------------------------------------------------------
    # Получение пути к OBJ-наполнителю из конфига моделей.
    # Возвращает относительный путь (как в models_geometry_config.json,
    # поле target_model), который сервер сам резолвит под models_dir.
    # ------------------------------------------------------------------
    def _get_target_model_path(self):
        cur_set = getattr(self.panda_app, "current_model_set", None)
        if not cur_set:
            return None
        try:
            from src.ui.panel_data import get_model_set_config
        except Exception as exc:
            print(f"[Perlin] panel_data import failed: {exc}")
            return None
        try:
            cfg = get_model_set_config(str(cur_set))
        except Exception as exc:
            print(f"[Perlin] get_model_set_config failed: {exc}")
            return None
        if not isinstance(cfg, dict):
            return None
        path = cfg.get("target_model")
        if not path:
            return None
        return str(path)

    # ------------------------------------------------------------------
    # Текстуры и PBR-материал
    # ------------------------------------------------------------------
    def _apply_textures_and_material(self, model_np):
        import os

        texset = self.panda_app.current_texture_set

        diffuse_path = (
            texset.get("diffuse")
            or texset.get("albedo")
            or "assets/textures/stones_8k/rocks_ground_01_diff_8k.jpg"
        )
        normal_path = texset.get(
            "normal",
            "assets/textures/stones_8k/rocks_ground_01_nor_dx_8k.jpg",
        )
        roughness_path = texset.get("roughness")
        metallic_path = texset.get("metallic")

        if not os.path.exists(diffuse_path):
            diffuse_path = "assets/textures/stones_8k/rocks_ground_01_diff_8k.jpg"
        if not os.path.exists(normal_path):
            normal_path = "assets/textures/stones_8k/rocks_ground_01_nor_dx_8k.jpg"

        # PERFORMANCE preset (simplepbr): RP's green-emission + 4-modulate-stage
        # convention renders flat/unlit under simplepbr. Use the PBR-friendly
        # material (base colour + roughness + non-metallic) so the fill mesh
        # is lit by the sun with proper light/shadow contrast.
        if not self.panda_app.use_render_pipeline:
            self.panda_app._apply_pbr_surface(
                model_np, diffuse_path, roughness_path=roughness_path)
            model_np.set_two_sided(True)
            return

        mat = Material()
        mat.set_base_color((1, 1, 1, 1))
        mat.set_emission((0, 1, 0, 0))
        model_np.set_material(mat)

        def setup_tex(tex, srgb=False):
            if srgb:
                tex.set_format(Texture.F_srgb)
            tex.set_minfilter(Texture.FTLinearMipmapLinear)
            tex.set_magfilter(Texture.FTLinear)
            tex.set_wrap_u(Texture.WMRepeat)
            tex.set_wrap_v(Texture.WMRepeat)

        ts_color = TextureStage("0-color")
        ts_color.set_sort(0)
        ts_normal = TextureStage("1-normal")
        ts_normal.set_sort(1)
        ts_metal = TextureStage("2-metallic")
        ts_metal.set_sort(2)
        ts_rough = TextureStage("3-roughness")
        ts_rough.set_sort(3)

        diffuse_tex = self.panda_app.loader.loadTexture(
            _texture_filename(diffuse_path)
        )
        setup_tex(diffuse_tex, srgb=True)
        model_np.set_texture(ts_color, diffuse_tex)

        normal_tex = self.panda_app.loader.loadTexture(
            _texture_filename(normal_path)
        )
        setup_tex(normal_tex)
        model_np.set_texture(ts_normal, normal_tex)

        if metallic_path and os.path.exists(metallic_path):
            metal_tex = self.panda_app.loader.loadTexture(
                _texture_filename(metallic_path)
            )
        else:
            metal_tex = Texture("dummy_metal")
            metal_tex.setup_2d_texture(
                1, 1, Texture.T_unsigned_byte, Texture.F_luminance
            )
            metal_tex.set_ram_image(b"\x00")
        setup_tex(metal_tex)
        model_np.set_texture(ts_metal, metal_tex)

        if roughness_path and os.path.exists(roughness_path):
            rough_tex = self.panda_app.loader.loadTexture(
                _texture_filename(roughness_path)
            )
        else:
            rough_tex = Texture("dummy_rough")
            rough_tex.setup_2d_texture(
                1, 1, Texture.T_unsigned_byte, Texture.F_luminance
            )
            rough_tex.set_ram_image(b"\x80")
        setup_tex(rough_tex)
        model_np.set_texture(ts_rough, rough_tex)

        model_np.set_two_sided(True)

    # ------------------------------------------------------------------
    # Обратная совместимость с main.py
    # ------------------------------------------------------------------
    def create_mesh_from_perlin_data(self):
        return self.generate_perlin_mesh_from_csg()
