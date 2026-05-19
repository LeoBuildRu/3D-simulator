# mesh_reconstruction.py
#
# Тонкий клиент. Раньше тут была вся локальная реконструкция: чтение PLY /
# heightmap, построение меша, отправка булевой разности на TLS-сервер.
# Теперь сервер сам всё считает в C++ (mesh_reconstruction.cpp,
# вызывается из POST /reconstruct_mesh, который форкается из
# Volume_calculator.py после поиска опорных точек). Готовый _result.obj
# уже лежит рядом с JSON на сервере, его имя — поле "result_obj" в JSON.
#
# Этот модуль просто скачивает .obj и кладёт его в сцену Panda3D — ровно
# по той же схеме, что использует браузерный превью BabylonJS.
#
# Публичный API не менялся, поэтому main.py / main_window.py / right_panel.py
# вызывают всё то же:
#
#   MeshReconstruction(panda_app, tls_client=tls_client)
#     .recon_json_path: str
#     .browse_recon_json()
#     .run_2d_to_3d_reconstruction()
#     .run_2d_to_3d_reconstruction_from(json_path, ply_path=None)

import os
import json

try:
    import trimesh
except ImportError as _e:  # pragma: no cover
    trimesh = None
    _TRIMESH_IMPORT_ERROR = _e
else:
    _TRIMESH_IMPORT_ERROR = None

try:
    from tkinter import filedialog
except ImportError:
    filedialog = None


class MeshReconstruction:
    def __init__(self, panda_app, tls_client=None):
        self.panda_app = panda_app
        self.tls_client = tls_client
        self.gui = getattr(panda_app, "gui", None)
        # Совместимость со старым модулем: эти поля читают/пишут UI и main.py.
        self.recon_json_path = ""
        self.ply_path = ""
        self.source_mesh_node = None

    # ------------------------------------------------------------------
    # Логирование — пытаемся в GUI, иначе в stdout (как было).
    # ------------------------------------------------------------------
    def log(self, message: str) -> None:
        if self.gui is not None:
            try:
                self.gui.log_message(message)
                return
            except Exception:
                pass
        print(message)

    # ------------------------------------------------------------------
    # Старая кнопка «выбрать локальный JSON».
    # ------------------------------------------------------------------
    def browse_recon_json(self):
        if filedialog is None:
            self.log("tkinter недоступен — нечем открыть диалог выбора файла")
            return None
        file_path = filedialog.askopenfilename(
            title="Select .json config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        self.recon_json_path = file_path or ""
        return file_path or None

    # ------------------------------------------------------------------
    # Главная точка входа.
    # ------------------------------------------------------------------
    def run_2d_to_3d_reconstruction(self) -> None:
        self.run_2d_to_3d_reconstruction_from(self.recon_json_path)

    def run_2d_to_3d_reconstruction_from(self, json_path: str, ply_path=None) -> None:
        # Чистим предыдущий результат — иначе повторный запуск накапливает
        # меши на сцене (точно как делал старый модуль).
        self._dispose_previous_mesh()

        self.log("🚀 Запуск 2D-3D реконструкции (server-side)")

        if not json_path or not os.path.isfile(json_path):
            self.log(f"❌ JSON не найден: {json_path!r}")
            return

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.log(f"✅ JSON загружен: {json_path}")

        # Сервер пишет в JSON имя готового .obj после успешной реконструкции.
        # Нет поля → серверная стадия ещё не отработала (или не нашла keypoints).
        result_obj = data.get("result_obj")
        if not result_obj:
            self.log(
                "❌ В JSON нет поля 'result_obj'. "
                "Серверный mesh_reconstruction либо ещё не запускался, "
                "либо упал на этапе boolean diff."
            )
            return

        if self.tls_client is None:
            self.log("❌ TLS client не передан в MeshReconstruction — нечем скачать .obj")
            return

        if trimesh is None:
            self.log(f"❌ trimesh не установлен: {_TRIMESH_IMPORT_ERROR}")
            return

        # Кладём .obj рядом с JSON; если уже лежит — не качаем повторно.
        local_obj_path = os.path.join(os.path.dirname(json_path), result_obj)
        if os.path.isfile(local_obj_path):
            self.log(f"📁 Использую локальную копию {result_obj}")
        else:
            self.log(f"⬇️ Скачиваю {result_obj} с сервера...")
            try:
                self.tls_client.download_file(result_obj, local_obj_path)
            except Exception as exc:
                self.log(f"❌ Не удалось скачать {result_obj}: {exc}")
                return

        # Парсим .obj. Сервер пишет plain vertices+triangles (без UV/normals),
        # trimesh нормально это глотает.
        try:
            mesh = trimesh.load(local_obj_path, force="mesh")
        except Exception as exc:
            self.log(f"❌ Не удалось разобрать {result_obj}: {exc}")
            return

        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            self.log("❌ .obj пустой или некорректный")
            return

        self.log(f"✅ Меш загружен: {len(mesh.vertices)} вершин, {len(mesh.faces)} треугольников")

        # trimesh → Panda3D NodePath (вычисляет нормали, формирует Geom).
        node_path = self.panda_app.trimesh_to_panda(mesh)
        if node_path is None or node_path.is_empty():
            self.log("❌ trimesh_to_panda вернул пустой NodePath")
            return

        self.panda_app.final_mesh_node = node_path

        # Текстура для реконструированного меша берётся из groundV2_4k —
        # ровно тот же набор, что использует babylon-viewer.js. Раньше тут
        # вызывался _apply_textures_and_material(), но он применяет текстуры
        # выбранного грузовика (current_texture_set) — на кузов это не то;
        # в сочетании с UV (0,0) из trimesh_to_panda до фикса меш был
        # однотонно-коричневым (один тексел кузова на всю поверхность).
        try:
            self._apply_babylon_ground_material(node_path)
        except Exception as exc:
            self.log(f"⚠️ Текстуры не применились: {exc}")

        # Объём: сервер уже посчитал и записал в JSON; локальный пересчёт оставлен
        # как fallback на случай старых JSON без поля.
        volume = data.get("target_volume")
        if volume is None:
            calc = getattr(self.panda_app, "calculate_mesh_volume", None)
            if callable(calc):
                try:
                    volume = calc(node_path)
                except Exception as exc:
                    self.log(f"⚠️ Не удалось пересчитать объём локально: {exc}")

        update_overlay = getattr(self.panda_app, "update_overlay_info", None)
        if callable(update_overlay) and volume is not None:
            try:
                update_overlay(volume=volume)
            except Exception as exc:
                self.log(f"⚠️ update_overlay_info упал: {exc}")

        self.log(f"✅ Реконструкция завершена, объём ≈ {volume} м³")

    # ------------------------------------------------------------------
    # Внутренние помощники
    # ------------------------------------------------------------------
    def _apply_babylon_ground_material(self, node_path) -> None:
        """Накладывает на NodePath набор текстур groundV2_4k с теми же
        параметрами, что использует babylon-viewer.js
        (см. aspnet-integration/wwwroot/js/babylon-viewer.js, константа
        GROUND_V2_4K и блок создания StandardMaterial groundV2_4k):

          - diffuse  = Ground_basecolor.jpg (sRGB)
          - normal   = Ground_normal.jpg
          - rough    = Ground_roughness.jpg
          - uScale   = 0.7
          - vScale   = 1.8
          - wrap     = repeat по U и V
          - backface = выключен (двусторонний рендер)

        Чтобы RP не ругался «GeomNode has no material», ставим Material
        с base_color = белым и emission = (0,1,0,0) — это RP-кодировка,
        где зелёный канал интерпретируется как сила нормалей (см.
        main.py:_apply_textures_and_material).
        """
        from panda3d.core import Texture, TextureStage, Material, Filename
        import os

        tex_dir       = os.path.join("assets", "textures", "groundV2_4k")
        diffuse_path  = os.path.join(tex_dir, "Ground_basecolor.jpg")
        normal_path   = os.path.join(tex_dir, "Ground_normal.jpg")
        rough_path    = os.path.join(tex_dir, "Ground_roughness.jpg")

        if not os.path.isfile(diffuse_path):
            self.log(f"⚠️ Не найдена {diffuse_path} — меш без текстуры")
            return

        # Babylon-параметры тайлинга, см. GROUND_V2_4K в babylon-viewer.js.
        U_SCALE, V_SCALE = 0.7, 1.8

        loader = self.panda_app.loader

        def _filename(path):
            return Filename.fromOsSpecific(str(path))

        def _make_tex(path: str, srgb: bool = False):
            t = loader.loadTexture(_filename(path))
            if srgb:
                t.setFormat(Texture.F_srgb)
            t.setMinfilter(Texture.FTLinearMipmapLinear)
            t.setMagfilter(Texture.FTLinear)
            t.setWrapU(Texture.WMRepeat)
            t.setWrapV(Texture.WMRepeat)
            return t

        # Слоты как в main.py:_apply_textures_and_material — иначе RP-шейдер
        # не подберёт нужные сэмплеры по их sort-индексам.
        ts_color = TextureStage("0-color");     ts_color.setSort(0)
        node_path.setTexture(ts_color, _make_tex(diffuse_path, srgb=True), 1)
        node_path.setTexScale(ts_color, U_SCALE, V_SCALE)

        if os.path.isfile(normal_path):
            ts_normal = TextureStage("1-normal"); ts_normal.setSort(1)
            node_path.setTexture(ts_normal, _make_tex(normal_path), 1)
            node_path.setTexScale(ts_normal, U_SCALE, V_SCALE)

        # Metallic — заглушка с нулевой металличностью (как в main.py).
        ts_metal = TextureStage("2-metallic"); ts_metal.setSort(2)
        metal_dummy = Texture("dummy_metal")
        metal_dummy.setup_2d_texture(1, 1, Texture.T_unsigned_byte, Texture.F_luminance)
        metal_dummy.set_ram_image(b"\x00")
        metal_dummy.setMinfilter(Texture.FTLinear)
        metal_dummy.setMagfilter(Texture.FTLinear)
        node_path.setTexture(ts_metal, metal_dummy, 1)

        if os.path.isfile(rough_path):
            ts_rough = TextureStage("3-roughness"); ts_rough.setSort(3)
            node_path.setTexture(ts_rough, _make_tex(rough_path), 1)
            node_path.setTexScale(ts_rough, U_SCALE, V_SCALE)

        mat = Material()
        mat.set_base_color((1, 1, 1, 1))
        mat.set_emission((0, 1, 0, 0))  # RP: G = normal strength
        node_path.set_material(mat, 1)
        node_path.set_two_sided(True)   # backFaceCulling = false в Babylon

    def _dispose_previous_mesh(self) -> None:
        """Удаляет с панда-сцены прошлый результат реконструкции, если он есть."""
        for attr in ("final_mesh_node", "mesh_node"):
            old_np = getattr(self.panda_app, attr, None)
            if not old_np:
                continue
            try:
                old_np.removeNode()
            except Exception:
                pass
            setattr(self.panda_app, attr, None)
            try:
                loaded = getattr(self.panda_app, "loaded_models", None)
                if loaded and old_np in loaded:
                    loaded.remove(old_np)
            except Exception:
                pass
