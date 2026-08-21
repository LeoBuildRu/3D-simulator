# segmentation_renderer.py
#
# Рендер карты СЕГМЕНТАЦИИ для датасетов.
#
# В отличие от карты глубины (DepthMapRenderer), которая рисует 2D-оверлей
# поверх основного кадра, сегментация должна знать, какой 3D-объект попал в
# каждый пиксель. Поэтому здесь сцена (base.render) перерисовывается в
# отдельный offscreen-буфер через собственную камеру: каждый объект заливается
# своим однотонным цветом, фон — чёрный.
#
# Цвет per-object задаётся через camera TAG STATES — это канонический способ
# Panda3D для масок сегментации/ID-буферов: на seg-камеру вешаются tag-states
# (плоский шейдер + цвет класса), а узлы помечаются тегом нужного класса.
# Tag-state применяется ТОЛЬКО нашей камерой, основное окно не трогается.
#
# Буфер НЕ проходит через постобработку RenderPipeline (tone mapping, bloom,
# sRGB и т.п.), поэтому цвета сохраняются попиксельно точными — это критично
# для масок сегментации.
from panda3d.core import (
    Texture,
    Shader,
    ShaderAttrib,
    ShaderInput,
    RenderState,
    GraphicsOutput,
    GraphicsPipe,
    FrameBufferProperties,
    WindowProperties,
    Camera,
    PerspectiveLens,
    PNMImage,
    LColor,
    LVecBase3f,
)

# ---------------------------------------------------------------------------
# Палитра классов сегментации (R, G, B), 0..255.
# Менять цвета классов — здесь.
# ---------------------------------------------------------------------------
SEG_BACKGROUND = (0, 0, 0)        # фон — чёрный

SEG_COLORS = {
    # груз (сгенерированное наполнение, panda_app.final_model)
    "cargo":  (253, 2, 2),        # красный
    # кузов (борта самосвала, current_cuzov_path)
    "cuzov":  (40, 85, 243),      # синий
    # окружение / насыпь (модель current_other_path)
    "other":  (29, 223, 126),     # зелёный
    # вспомогательная ground_plane (обычно скрыта после CSG); тот же
    # «зелёный окружения», чтобы если вдруг видна — не выбивалась.
    "ground": (29, 223, 126),
    # ткань/тент, свисающая с борта (ClothSimulator) — яркий оранжевый.
    # Отстоит от cargo (253,2,2) на 128 по G, поэтому переживает допуск
    # tol=40, которым маска разбирается в renderer_utils.
    "cloth":  (255, 128, 0),
}

# Заводские цвета — по ним диалог настроек умеет откатывать палитру, даже
# когда SEG_COLORS уже переопределён пользовательским конфигом.
DEFAULT_SEG_BACKGROUND = SEG_BACKGROUND
DEFAULT_SEG_COLORS = dict(SEG_COLORS)

# Человекочитаемые подписи классов для UI (диалог настроек датасета).
SEG_LABELS = {
    "cargo":  ("Груз", "Сгенерированное наполнение кузова"),
    "cuzov":  ("Кузов", "Борта и дно самосвала"),
    "other":  ("Окружение", "Насыпь и прочая геометрия сцены"),
    "ground": ("Земля", "Вспомогательная плоскость (обычно скрыта)"),
    "cloth":  ("Ткань", "Тент/полог, свисающий с борта"),
}

# Доп. палитра для будущей раскраски РАЗНЫХ бортов кузова разными цветами.
# Пока не используется (по ТЗ — опционально), оставлено как справочник.
CUZOV_SIDE_COLORS = [
    (40, 85, 243),
    (50, 183, 250),
    (99, 57, 253),
    (29, 223, 126),
]

# Ключ тега, по которому seg-камера выбирает tag-state для узла.
_SEG_TAG_KEY = "segclass"


_FLAT_VERT = """
#version 330
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""

_FLAT_FRAG = """
#version 330
uniform vec3 segColor;
out vec4 fragColor;
void main() {
    fragColor = vec4(segColor, 1.0);
}
"""


def _to_unit(rgb):
    """(0..255) -> LVecBase3f(0..1)."""
    return LVecBase3f(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


class SegmentationRenderer:
    def __init__(self, base):
        self.base = base
        self.buffer = None
        self.seg_tex = None
        self.seg_cam = None        # Camera node
        self.seg_cam_np = None     # NodePath
        self.seg_lens = None
        self.display_region = None
        self._size = (0, 0)

        # Плоский шейдер.
        self.flat_shader = Shader.make(Shader.SL_GLSL, _FLAT_VERT, _FLAT_FRAG)

        # Tag-state на каждый класс: один ShaderAttrib = (плоский шейдер +
        # цвет класса как shader input). Высокий override, чтобы перебить
        # автошейдеры узлов / simplepbr / RenderPipeline.
        self._tag_states = {}
        for class_name, rgb in SEG_COLORS.items():
            self._tag_states[class_name] = self._make_flat_state(rgb, 20000)

        # initial-state камеры = плоский шейдер + чёрный (класс «фон/прочее»)
        # для всех непомеченных узлов.
        self._bg_state = self._make_flat_state(SEG_BACKGROUND, 10000)

    def _make_flat_state(self, rgb, override):
        attrib = ShaderAttrib.make(self.flat_shader)
        attrib = attrib.set_shader_input(ShaderInput("segColor", _to_unit(rgb)))
        return RenderState.make(attrib, override)

    # ------------------------------------------------------------------
    # Палитра классов: чтение и правка на лету.
    #
    # Цвета живут в модуле (SEG_COLORS / SEG_BACKGROUND), потому что их
    # читает и renderer_utils — он разбирает готовую маску по цветам, чтобы
    # вырезать передний план. Поэтому правим словарь НА МЕСТЕ, а не
    # подменяем его: иначе у renderer_utils остался бы старый объект и
    # вырез переднего плана поехал бы после первой же смены цвета.
    # ------------------------------------------------------------------
    def get_palette(self):
        """Текущая палитра: {"background": (r,g,b), "<класс>": (r,g,b), ...}."""
        palette = {"background": tuple(SEG_BACKGROUND)}
        palette.update({k: tuple(v) for k, v in SEG_COLORS.items()})
        return palette

    def set_class_color(self, class_name, rgb):
        """Сменить цвет одного класса (0..255). Пересобирает tag-state."""
        rgb = tuple(int(max(0, min(255, c))) for c in rgb)
        if class_name == "background":
            return self.set_background_color(rgb)
        if class_name not in SEG_COLORS:
            return False
        SEG_COLORS[class_name] = rgb
        state = self._make_flat_state(rgb, 20000)
        self._tag_states[class_name] = state
        if self.seg_cam is not None:
            self.seg_cam.set_tag_state(class_name, state)
        return True

    def set_background_color(self, rgb):
        """Цвет фона маски: initial-state камеры + clear-цвет буфера."""
        global SEG_BACKGROUND
        rgb = tuple(int(max(0, min(255, c))) for c in rgb)
        SEG_BACKGROUND = rgb
        self._bg_state = self._make_flat_state(rgb, 10000)
        if self.seg_cam is not None:
            self.seg_cam.set_initial_state(self._bg_state)
        color = LColor(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0)
        for target in (self.buffer, self.display_region):
            if target is not None:
                try:
                    target.set_clear_color_active(True)
                    target.set_clear_color(color)
                except Exception:
                    pass
        return True

    def apply_palette(self, palette):
        """Применить палитру целиком; неизвестные ключи игнорируются."""
        for name, rgb in (palette or {}).items():
            if rgb is None:
                continue
            try:
                self.set_class_color(name, rgb)
            except Exception as exc:
                print(f"[Segmentation] цвет класса {name!r} не применён: {exc}")

    # ------------------------------------------------------------------
    # Ленивое создание GL-буфера (на момент первого вызова окно гарантированно
    # инициализировано и имеет реальный размер).
    # ------------------------------------------------------------------
    def _ensure_setup(self):
        win = self.base.win
        w = max(1, win.getXSize())
        h = max(1, win.getYSize())

        if self.buffer is not None and self._size == (w, h):
            return True

        # Размер окна изменился — пересоздаём буфер.
        self._teardown()

        fb_props = FrameBufferProperties()
        fb_props.set_rgba_bits(8, 8, 8, 8)
        fb_props.set_srgb_color(False)   # точные цвета, без sRGB-гаммы
        fb_props.set_depth_bits(24)      # depth нужен для корректного перекрытия
        fb_props.set_multisamples(0)     # без сглаживания — чёткие границы классов

        self.buffer = self.base.graphicsEngine.make_output(
            self.base.pipe,
            "segmentation_buffer",
            -10,
            fb_props,
            WindowProperties.size(w, h),
            GraphicsPipe.BF_refuse_window,
            self.base.win.get_gsg(),
            self.base.win,
        )

        if self.buffer is None:
            # Фолбэк: упрощённый текстурный буфер (обычно тоже даёт
            # линейный RGBA8).
            self.seg_tex = Texture("segmentation_tex")
            self.buffer = self.base.win.make_texture_buffer(
                "segmentation_buffer", w, h, self.seg_tex, to_ram=True
            )
            if self.buffer is None:
                print("[Segmentation] не удалось создать offscreen-буфер")
                return False
        else:
            self.seg_tex = Texture("segmentation_tex")
            self.buffer.add_render_texture(
                self.seg_tex,
                GraphicsOutput.RTM_copy_ram,
                GraphicsOutput.RTP_color,
            )

        bg = LColor(
            SEG_BACKGROUND[0] / 255.0,
            SEG_BACKGROUND[1] / 255.0,
            SEG_BACKGROUND[2] / 255.0,
            1.0,
        )
        self.buffer.set_clear_color_active(True)
        self.buffer.set_clear_color(bg)
        self.buffer.set_clear_depth_active(True)
        self.buffer.set_clear_depth(1.0)

        # Своя камера: плоский initial-state + tag-states по классам.
        self.seg_lens = PerspectiveLens()
        self.seg_cam = Camera("segmentation_camera", self.seg_lens)
        self.seg_cam.set_initial_state(self._bg_state)
        self.seg_cam.set_tag_state_key(_SEG_TAG_KEY)
        for class_name, state in self._tag_states.items():
            self.seg_cam.set_tag_state(class_name, state)
        self.seg_cam_np = self.base.render.attach_new_node(self.seg_cam)

        dr = self.buffer.make_display_region(0, 1, 0, 1)
        dr.set_camera(self.seg_cam_np)
        dr.set_clear_color_active(True)
        dr.set_clear_color(bg)
        self.display_region = dr

        # По умолчанию буфер не активен — включаем только на момент захвата.
        self.buffer.set_active(False)
        self._size = (w, h)
        return True

    def _teardown(self):
        if self.seg_cam_np is not None:
            self.seg_cam_np.remove_node()
            self.seg_cam_np = None
        self.seg_cam = None
        if self.buffer is not None:
            self.base.graphicsEngine.remove_window(self.buffer)
            self.buffer = None
        self.seg_tex = None
        self.seg_lens = None
        self.display_region = None
        self._size = (0, 0)

    # ------------------------------------------------------------------
    # Сопоставление 3D-узлов сцены классам сегментации.
    # Возвращает список (NodePath, class_name).
    # ------------------------------------------------------------------
    def _resolve_seg_nodes(self):
        app = self.base
        items = []

        def add(node, class_name):
            if node is not None and not node.is_empty():
                items.append((node, class_name))

        # Груз — сгенерированное наполнение.
        add(getattr(app, "final_model", None), "cargo")

        # Кузов / окружение — ищем по сохранённым путям среди loaded_models.
        cuzov_path = getattr(app, "current_cuzov_path", None)
        other_path = getattr(app, "current_other_path", None)
        model_paths = getattr(app, "model_paths", {}) or {}
        loaded = getattr(app, "loaded_models", []) or []

        for node in loaded:
            if node is None or node.is_empty():
                continue
            path = model_paths.get(id(node))
            if path is None:
                continue
            if cuzov_path and path == cuzov_path:
                add(node, "cuzov")
            elif other_path and path == other_path:
                add(node, "other")

        # Ground-plane (обычно скрыта после CSG; hide() всё равно исключит её
        # из рендера, тег задаём на всякий случай).
        add(getattr(app, "ground_plane", None), "ground")

        # Ткань/тент. Тег вешается на корень полотна; перекрытие груза и
        # кузова разрешает depth-буфер, так что отдельного порядка не нужно.
        cloth_sim = getattr(app, "cloth_simulator", None)
        if cloth_sim is not None:
            add(getattr(cloth_sim, "node", None), "cloth")

        return items

    def _hidden_nodes(self):
        """Узлы, которые надо временно скрыть, чтобы не пачкать маску
        (частицы-листья и т.п.)."""
        out = []
        particles = getattr(self.base, "particles", None)
        if particles is not None:
            node = getattr(particles, "node", None)
            if node is not None and not node.is_empty():
                out.append(node)
        return out

    # ------------------------------------------------------------------
    # Снять кадр сегментации. Возвращает PNMImage (размер = размер окна)
    # или None при ошибке. Камера/линза синхронизируются с основной камерой.
    # ------------------------------------------------------------------
    def capture(self):
        if not self._ensure_setup():
            return None

        # 1) Синхронизируем позу и линзу с основной камерой. base.cam — это
        # реальный узел камеры с линзой (base.camera — лишь его родитель),
        # поэтому мировую позу берём с base.cam относительно render.
        main_cam = self.base.cam
        self.seg_cam_np.set_pos(main_cam.get_pos(self.base.render))
        self.seg_cam_np.set_hpr(main_cam.get_hpr(self.base.render))

        main_lens = main_cam.node().get_lens()
        if hasattr(main_lens, "get_fov"):
            self.seg_lens.set_fov(main_lens.get_fov())
        if hasattr(main_lens, "get_near") and hasattr(main_lens, "get_far"):
            self.seg_lens.set_near_far(main_lens.get_near(), main_lens.get_far())
        self.seg_lens.set_aspect_ratio(main_lens.get_aspect_ratio())

        # 2) Помечаем узлы тегами классов (видит только seg-камера).
        seg_items = self._resolve_seg_nodes()
        tagged = []
        for node, class_name in seg_items:
            node.set_tag(_SEG_TAG_KEY, class_name)
            tagged.append(node)

        # 3) Прячем мешающие узлы (частицы).
        hidden = []
        for node in self._hidden_nodes():
            if not node.is_hidden():
                node.hide()
                hidden.append(node)

        img = PNMImage()
        ok = False
        try:
            # 4) Рендерим один кадр в буфер и читаем его.
            self.buffer.set_active(True)
            self.base.graphicsEngine.render_frame()

            if self.seg_tex is not None and self.seg_tex.has_ram_image():
                ok = self.seg_tex.store(img)
            if not ok:
                ok = self.buffer.get_screenshot(img)
        finally:
            self.buffer.set_active(False)
            # 5) Восстанавливаем сцену.
            for node in hidden:
                node.show()
            for node in tagged:
                node.clear_tag(_SEG_TAG_KEY)

        return img if ok else None
