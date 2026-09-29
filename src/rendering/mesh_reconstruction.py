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
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

try:
    from tkinter import filedialog
except ImportError:
    filedialog = None


def uv_scale_for(tex_set) -> tuple:
    """
    Повторы текстуры наполнителя (U, V) из набора текстур.

    Тайлинг берём из текущего набора текстур (textures_napolnitel_config.json
    на сервере), а не из жёстко зашитых Babylon-констант — иначе правка
    textureRepeatX/Y на сервере не влияет на меш реконструкции. Fallback на
    исторические Babylon-значения, если набор ещё не подъехал.
    """
    tex_set = tex_set or {}
    out = []
    for key, fallback in (("textureRepeatX", 0.7), ("textureRepeatY", 1.8)):
        try:
            out.append(float(tex_set.get(key, fallback)))
        except (TypeError, ValueError):
            out.append(fallback)
    return tuple(out)


#: Карты материала наполнителя (diffuse, normal, roughness) — тот же набор
#: groundV2_4k, что у babylon-viewer.js.
_GROUND_TEX_DIR = os.path.join("assets", "textures", "groundV2_4k")
_GROUND_MAPS = tuple(os.path.join(_GROUND_TEX_DIR, name) for name in (
    "Ground_basecolor.jpg", "Ground_normal.jpg", "Ground_roughness.jpg"))


@dataclass
class PreparedMesh:
    """Результат `MeshReconstruction.prepare`: всё, что нужно для сцены."""

    json_path: str
    data: Dict[str, Any]
    obj_path: str
    vertices: np.ndarray        # (N,3) float64
    faces: np.ndarray           # (M,3) int64
    interleaved: np.ndarray     # (N,8) float32: pos, normal, uv


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
        """Синхронный запуск: подготовка и применение подряд в этом потоке.

        Интерфейс по проездам этим больше не пользуется — он зовёт
        `prepare()` из фонового потока и `apply()` из главного (см.
        src/rendering/recon_job.py), чтобы рендер не вставал.
        """
        # Чистим предыдущий результат — иначе повторный запуск накапливает
        # меши на сцене (точно как делал старый модуль).
        self._dispose_previous_mesh()
        tex_set = getattr(self.panda_app, "current_texture_set", None) or {}
        prepared = self.prepare(json_path, uv_scale=uv_scale_for(tex_set))
        if prepared is not None:
            self.apply(prepared)

    # ------------------------------------------------------------------
    # Подготовка: сеть, диск, numpy. Сцену не трогает — можно из потока.
    # ------------------------------------------------------------------
    def prepare(self, json_path: str, uv_scale=(0.7, 1.8)):
        """
        Прочитать JSON, скачать `_result.obj` (если его ещё нет рядом),
        разобрать и посчитать нормали/UV. Возвращает `PreparedMesh` или None
        (причина уже в логе).

        `uv_scale` — повторы текстуры (textureRepeatX/Y выбранного набора):
        RenderPipeline не уважает матрицу TextureStage, поэтому масштаб
        зашивается прямо в texcoord.
        """
        from src.rendering import mesh_io

        self.log("🚀 Запуск 2D-3D реконструкции (server-side)")

        if not json_path or not os.path.isfile(json_path):
            self.log(f"❌ JSON не найден: {json_path!r}")
            return None

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
            return None

        # Кладём .obj рядом с JSON; если уже лежит — не качаем повторно.
        local_obj_path = os.path.join(os.path.dirname(json_path), result_obj)
        if os.path.isfile(local_obj_path):
            self.log(f"📁 Использую локальную копию {result_obj}")
        else:
            if self.tls_client is None:
                self.log("❌ TLS client не передан в MeshReconstruction — нечем скачать .obj")
                return None
            self.log(f"⬇️ Скачиваю {result_obj} с сервера...")
            try:
                self.tls_client.download_file(result_obj, local_obj_path)
            except Exception as exc:
                self.log(f"❌ Не удалось скачать {result_obj}: {exc}")
                return None

        # Сервер пишет plain vertices+triangles (без UV/normals).
        try:
            vertices, faces = mesh_io.load_obj_arrays(local_obj_path)
        except Exception as exc:
            self.log(f"❌ Не удалось разобрать {result_obj}: {exc}")
            return None

        if len(vertices) == 0 or len(faces) == 0:
            self.log("❌ .obj пустой или некорректный")
            return None

        self.log(f"✅ Меш загружен: {len(vertices)} вершин, {len(faces)} треугольников")

        normals = mesh_io.vertex_normals(vertices, faces)
        uv = mesh_io.planar_uv(vertices, *uv_scale)
        # Текстуры материала — в TexturePool заранее: декодирование 4k-карт
        # в apply() стоило бы кадров. Загрузчик Panda отпускает GIL.
        from panda3d.core import Filename
        for path in _GROUND_MAPS:
            if os.path.isfile(path):
                try:
                    self.panda_app.loader.loadTexture(
                        Filename.fromOsSpecific(str(path)))
                except Exception:
                    pass
        return PreparedMesh(
            json_path=json_path,
            data=data,
            obj_path=local_obj_path,
            vertices=vertices,
            faces=faces,
            interleaved=mesh_io.interleave_v3n3t2(vertices, normals, uv),
        )

    # ------------------------------------------------------------------
    # Применение: только главный поток.
    # ------------------------------------------------------------------
    def apply(self, prepared: "PreparedMesh"):
        """Собрать Geom из подготовленных массивов и поставить в сцену.

        Возвращает NodePath меша или None.
        """
        from src.rendering import mesh_io

        self._dispose_previous_mesh()
        data = prepared.data

        node = mesh_io.geom_node_from_arrays(
            "trimesh_result", prepared.interleaved, prepared.faces)
        node_path = self.panda_app.attach_generated_mesh(node)
        if node_path is None or node_path.is_empty():
            self.log("❌ не удалось поставить меш в сцену")
            return None

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
        return node_path

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

        diffuse_path, normal_path, rough_path = _GROUND_MAPS

        if not os.path.isfile(diffuse_path):
            self.log(f"⚠️ Не найдена {diffuse_path} — меш без текстуры")
            return

        # Масштаб тайлинга (textureRepeatX/Y) уже зашит в texcoord при
        # подготовке — см. prepare() и uv_scale_for(): RenderPipeline не
        # уважает матрицу TextureStage, а режим WMRepeat у текстуры даёт
        # нужное число повторов.

        loader = self.panda_app.loader

        # PERFORMANCE preset (simplepbr): recompute upward normals + apply a
        # diffuse PBR material so the reconstructed mesh is lit by the sun
        # (the RP material/stage convention renders it flat/black here).
        if not self.panda_app.use_render_pipeline:
            self.panda_app.relight_generated_mesh(
                node_path, diffuse_path=diffuse_path, roughness_path=rough_path)
            return

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
        # NOTE: setTexScale здесь НЕ вызываем — масштаб уже зашит в UV выше,
        # а RenderPipeline всё равно не уважает texture-matrix у стейджа.
        ts_color = TextureStage("0-color");     ts_color.setSort(0)
        node_path.setTexture(ts_color, _make_tex(diffuse_path, srgb=True), 1)

        if os.path.isfile(normal_path):
            ts_normal = TextureStage("1-normal"); ts_normal.setSort(1)
            node_path.setTexture(ts_normal, _make_tex(normal_path), 1)

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
