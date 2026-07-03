# gemini_postprocess.py
#
# Постобработка отрендеренных кадров датасета через Google Gemini image API
# ("Nano Banana" / gemini-2.5-flash-image). Делает кадры «сложнее» для обучения
# ИИ, не трогая 3D-мир:
#   * передний план (кузов + груз) — выветривание ПОВЕРХНОСТИ внутри силуэта:
#     ржавчина, разные цвета/краска, вмятины/швы пластин как затенение, кабели,
#     разнофракционный груз (куски бетона/металла/цветные обломки к песку);
#   * фон — генерируется заново (промзона, склад, металл, металлический асфальт).
#
# ВАЖНО про выравнивание: этот модуль возвращает только КАРТИНКИ. Гарантию, что
# силуэт переднего плана попиксельно совпадает с картой глубины/маской (GT),
# обеспечивает вызывающая сторона (renderer_utils) жёстким матированием по маске
# сегментации. Промпты дополнительно просят Gemini не менять контуры.
#
# Зависимости: только requests (уже в requirements) + PIL/numpy (уже
# используются в renderer_utils). Ошибки/таймауты никогда не роняют рендер —
# методы возвращают None, вызывающая сторона откатывается на обычное поведение.

import os
import io
import json
import time
import base64
import random

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "gemini.json")
_BG_CACHE_DIR = os.path.join(
    _PROJECT_ROOT, "assets", "backgrounds", "_gemini_cache")

_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)


# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = {
    "enabled": True,
    # Провайдер постобработки: "huggingface" (по умолчанию, работает в EU) или
    # "gemini" (нужен не-EEA регион + биллинг).
    "provider": "huggingface",
    # Модель Gemini для генерации/редактирования изображений.
    "model": "gemini-2.5-flash-image",
    # --- Hugging Face ---
    "hf_token": "",
    "hf_base": "https://router.huggingface.co/hf-inference/models/",
    "hf_text_model": "black-forest-labs/FLUX.1-schnell",   # text-to-image (фон)
    "hf_edit_model": "black-forest-labs/FLUX.1-Kontext-dev",  # image-to-image
    "hf_strength": 0.55,        # сила img2img (меньше — ближе к оригиналу)
    # --- DeepInfra (делает и фон, и img2img; принимает base64; $5/мес free) ---
    "di_token": "",
    "di_base": "https://api.deepinfra.com/v1/inference/",
    "di_text_model": "black-forest-labs/FLUX-1-schnell",     # text-to-image (фон)
    "di_edit_model": "black-forest-labs/FLUX.1-Kontext-dev",  # image-to-image edit
    # --- OpenAI GPT Image (платный; редактирует ВЕСЬ кадр за один запрос) ---
    "openai_api_key": "",
    "openai_model": "gpt-image-1",
    "openai_size": "1536x1024",   # ближайший к 16:9 landscape (потом ресайз)
    "openai_quality": "medium",   # low | medium | high (влияет на цену/качество)
    # Таймаут одного HTTP-запроса, сек.
    "timeout": 60,
    # Вероятность переиспользовать недавно сгенерированный фон из кэша вместо
    # нового запроса (экономия квоты). 0 — всегда генерировать новый.
    "background_reuse_prob": 0.5,
    # Сколько последних фонов держать в кэше.
    "background_cache_size": 24,
    # Применять выветривание переднего плана к доле кадров (1.0 — ко всем).
    "foreground_prob": 1.0,
    # Режим выветривания переднего плана (кузов/груз):
    #   "procedural" — бесплатно, оффлайн, numpy/PIL (по умолчанию);
    #   "ai"         — через провайдера img2img (нужен платный/квотный ключ);
    #   "off"        — не трогать передний план (только фон + свет).
    "foreground_mode": "procedural",
}


def load_gemini_config():
    """Прочитать config/gemini.json; ключ — оттуда либо из env GEMINI_API_KEY.

    Возвращает dict с полями _DEFAULT_CONFIG + api_key (может быть пустым).
    """
    cfg = dict(_DEFAULT_CONFIG)
    try:
        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f) or {})
    except Exception as exc:
        print(f"[Gemini] не удалось прочитать {_CONFIG_PATH}: {exc}")
    if not cfg.get("api_key"):
        cfg["api_key"] = os.environ.get("GEMINI_API_KEY", "")
    return cfg


# ---------------------------------------------------------------------------
# Банки фраз для сборки разнообразных промптов ("сложные случаи").
# Не все элементы попадают в каждый кадр — см. build_*_prompt.
# ---------------------------------------------------------------------------
_BODY_DAMAGE = [
    "shallow dents and creases pressed into the metal body panels",
    "slightly bent and buckled side panels of the dump body",
    "one body corner sticking out and deformed",
    "steel plates of the body slightly separated at the seams, edges lifted a bit",
    "wavy, warped sheet metal along the top rail of the body",
    "a torn and folded patch of metal on the side wall",
]
_CABLES = [
    "a few loose cables and wires draped along the side of the body",
    "a hanging cable dangling down over the body wall",
    "frayed rubber hoses hanging from the top edge of the body",
]
_RUST = [
    "rust streaks running down from the top rail",
    "patches of orange-brown rust and corrosion spreading from welds and bolts",
    "heavy rust stains bleeding down the side walls",
    "flaking paint revealing rusty metal underneath",
]
_PAINT = [
    "faded and sun-bleached paint",
    "the body repainted an uneven color with visible brush marks",
    "scratched and chipped paint with bare metal showing",
    "the body a different color than usual (repainted, mismatched panels)",
    "dirty, dusty, mud-splattered body surface",
]
_CARGO_FRACTIONS = [
    "chunks of broken concrete of various sizes mixed into the sand pile",
    "pieces of scrap metal and rebar sticking out of the material",
    "colorful debris and broken bricks scattered across the cargo",
    "large rocks and gravel of mixed sizes among the fine sand",
    "a coarse mix of rubble: concrete blocks, metal scraps and colored fragments",
    "irregular lumps of different colors and shapes throughout the load",
]

# Фоны — то, что реально видно с камеры, закреплённой на кузове самосвала:
# карьеры, перевалочные/погрузочные станции, дороги, другие самосвалы, кучи
# материала, асфальт, заборы. Всё серое, пыльное, промышленное.
_BACKGROUNDS = [
    "an open-pit mine with terraced grey rock walls and dusty haul roads",
    "a quarry with big grey heaps of crushed stone, sand and gravel",
    "a material transfer and loading station with conveyors and hoppers",
    "a stockpile yard with large mounds of grey bulk material",
    "a dusty haul road with other worn dump trucks and an excavator",
    "an aggregate depot with piles of rubble, sand and broken concrete",
    "a grey asphalt yard with parked dump trucks and worn pavement",
    "an industrial loading area behind a rusty wire-mesh fence",
    "a cement and gravel plant with grey silos, pipes and steel structures",
    "a truck weighbridge station with barriers, fencing and grey asphalt",
]
# Металл на фоне (требование ТЗ) — приглушённый, ржавый, не блестящий.
_METAL_SURFACES = [
    "dull corrugated steel sheets and rusty metal cladding",
    "a row of weathered grey steel shipping containers",
    "rusty galvanized metal panels and steel fencing",
]
# Асфальт почти металлического серого цвета (требование ТЗ).
_ASPHALT_METALLIC = [
    "worn dark asphalt of an almost metallic grey colour",
    "a dull grey asphalt lot with a faint metallic sheen",
]

# Освещение — тусклое, промышленное, как у CCTV (не «красивое»).
_LIGHTING_HINT = [
    "flat overcast dusty light",
    "grey hazy daylight",
    "dull diffuse industrial light",
    "murky low-contrast afternoon light",
]


def _sample(lst, n):
    n = min(n, len(lst))
    return random.sample(lst, n) if n > 0 else []


def _pick_tier():
    return random.choices(
        ["clean", "light", "heavy"], weights=[0.20, 0.40, 0.40])[0]


def _foreground_features(tier):
    """Описание повреждений КУЗОВА (без груза) по уровню сложности."""
    parts = []
    if tier == "clean":
        parts += _sample(_PAINT, 1)
    elif tier == "light":
        parts += _sample(_RUST, 1)
        parts += _sample(_PAINT, 1)
        if random.random() < 0.5:
            parts += _sample(_BODY_DAMAGE, 1)
    else:  # heavy
        parts += _sample(_BODY_DAMAGE, random.randint(1, 2))
        parts += _sample(_RUST, random.randint(1, 2))
        parts += _sample(_PAINT, 1)
        if random.random() < 0.7:
            parts += _sample(_CABLES, 1)
    return "; ".join(parts) if parts else "light dust and wear"


def _cargo_clause(tier):
    """Как поменять НАПОЛНЕНИЕ (сейчас модель часто оставляет песок как есть)."""
    n = 1 if tier == "clean" else random.randint(1, 2)
    frac = "; ".join(_sample(_CARGO_FRACTIONS, n))
    return (
        "Do NOT leave the load as plain sand — replace the bulk material in "
        "the truck body with a clearly different, coarser mixed load: " + frac
        + ". Vary its colour and texture so it no longer looks like clean sand."
    )


def _pick_bg_scene():
    """Сцена фона (то, что видно с камеры на кузове). Иногда металл/асфальт."""
    roll = random.random()
    if roll < 0.20:
        scene = random.choice(_METAL_SURFACES)
    elif roll < 0.38:
        scene = random.choice(_ASPHALT_METALLIC)
    else:
        scene = random.choice(_BACKGROUNDS)
        if random.random() < 0.4:
            scene += ", with " + random.choice(
                _METAL_SURFACES + _ASPHALT_METALLIC)
    return scene


def build_foreground_prompt():
    """Промпт для img2img-провайдеров (выветривание переднего плана внутри
    силуэта). Возвращает (prompt_text, tier)."""
    tier = _pick_tier()
    features = _foreground_features(tier)
    if random.random() < 0.7:
        features += "; " + "; ".join(_sample(_CARGO_FRACTIONS, 1))
    prompt = (
        "You are editing a photo of a dump truck body loaded with bulk "
        "material, for a computer-vision training dataset. "
        "Make the following surface changes look photorealistic: "
        f"{features}. "
        "STRICT CONSTRAINTS: keep the exact outline, silhouette, pose, scale "
        "and position of the truck body and the cargo pile pixel-identical to "
        "the input. Do NOT add, remove or move any geometry beyond the "
        "existing shape; do NOT change the camera or perspective. Keep the "
        "background exactly as it is. Output the full image at the same "
        "resolution."
    )
    return prompt, tier


# Общий стиль CCTV — не «красиво», приглушённо, серо, похоже на ржавый кузов.
_CCTV_STYLE = (
    "Overall look: this is a still frame from a cheap fixed CCTV / "
    "surveillance camera mounted on a dump truck body — NOT an artistic or "
    "attractive photo. Muted, desaturated, low-contrast, grey and rusty "
    "industrial tones, flat overcast dusty light, slightly grainy and soft. "
    "No bright or vivid colours. Make the background tones RESEMBLE the grey "
    "rusty steel body and the grey bulk material (low colour and brightness "
    "contrast between truck, load and background) so the scene is visually "
    "hard to segment."
)


def build_combined_prompt(shadow=False):
    """Промпт для whole-image режима (OpenAI): за ОДИН запрос — выветривание
    кузова, смена наполнения и новый фон, всё в стиле CCTV с кузова самосвала.
    Возвращает (prompt_text, tier)."""
    tier = _pick_tier()
    body = _foreground_features(tier)
    cargo = _cargo_clause(tier)
    scene = _pick_bg_scene()
    prompt = (
        "This is a still frame from a low-quality fixed CCTV camera mounted "
        "on the body of a dump truck, looking down at the truck's own load "
        "with the industrial surroundings in the background. Edit the frame "
        "into a harder, realistic training sample. "
        f"On the truck body make these look photorealistic: {body}. "
        f"{cargo} "
        "Replace the background around and behind the truck with "
        f"{scene}. "
        "Keep the truck body and its load in the EXACT same outline, "
        "position, scale and perspective as the input — do not move, rotate, "
        "resize or re-frame them; only their surface and the surroundings "
        "change. "
        + _CCTV_STYLE +
        " Output the full edited image at the same resolution."
    )
    if shadow:
        prompt += (
            " Lighting: a hard-edged diagonal cast shadow falls across the "
            "scene, splitting the truck body and its load into a sunlit half "
            "and a clearly shaded half."
        )
    return prompt, tier


def build_background_prompt():
    """Промпт для standalone text-to-image фона (HF/DeepInfra), стиль CCTV."""
    scene = _pick_bg_scene()
    lighting = random.choice(_LIGHTING_HINT)
    return (
        f"A still frame from a low-quality fixed CCTV surveillance camera at "
        f"{scene}, {lighting}. Muted, desaturated, low-contrast grey and "
        "rusty industrial tones, no bright colours, slightly grainy, not "
        "artistic. No people, no text. Empty utilitarian scene suitable as a "
        "backdrop."
    )


# ---------------------------------------------------------------------------
# Процессор
# ---------------------------------------------------------------------------
class GeminiPostProcessor:
    def __init__(self, config=None):
        self.config = config or load_gemini_config()
        self._session = None
        self._last_prompts = {}   # для записи в метаданные (debug)

    # ---- доступность ----
    def available(self):
        if not self.config.get("enabled", True):
            return False
        if not self.config.get("api_key"):
            return False
        try:
            import requests  # noqa: F401
        except Exception:
            return False
        return True

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    # ---- низкоуровневый вызов ----
    def _call_generate(self, text, images=None):
        """Один запрос generateContent. images — список PIL.Image (inline).

        Возвращает PIL.Image из ответа или None при любой ошибке.
        """
        try:
            from PIL import Image  # noqa: F401
        except Exception as exc:
            print(f"[Gemini] PIL недоступен: {exc}")
            return None

        parts = [{"text": text}]
        for img in (images or []):
            b64 = self._img_to_b64(img)
            if b64 is None:
                return None
            parts.append(
                {"inline_data": {"mime_type": "image/png", "data": b64}})

        body = {"contents": [{"parts": parts}]}
        url = _API_URL.format(model=self.config.get(
            "model", "gemini-2.5-flash-image"))
        try:
            resp = self._get_session().post(
                url,
                params={"key": self.config["api_key"]},
                json=body,
                timeout=float(self.config.get("timeout", 60)),
            )
        except Exception as exc:
            print(f"[Gemini] сетевая ошибка: {exc}")
            return None

        if resp.status_code != 200:
            print(f"[Gemini] HTTP {resp.status_code}: {resp.text[:300]}")
            return None

        try:
            data = resp.json()
            cand = data["candidates"][0]
            out_parts = cand["content"]["parts"]
        except Exception as exc:
            print(f"[Gemini] не разобрал ответ: {exc}")
            return None

        for part in out_parts:
            # REST может отдавать snake_case или camelCase.
            inline = part.get("inline_data") or part.get("inlineData")
            if inline and inline.get("data"):
                return self._b64_to_img(inline["data"])
        print("[Gemini] в ответе нет изображения (возможно, только текст/отказ)")
        return None

    @staticmethod
    def _img_to_b64(img):
        try:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            print(f"[Gemini] кодирование изображения не удалось: {exc}")
            return None

    @staticmethod
    def _b64_to_img(b64):
        try:
            from PIL import Image
            raw = base64.b64decode(b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            print(f"[Gemini] декодирование изображения не удалось: {exc}")
            return None

    # ---- высокоуровневые операции ----
    def weather_foreground(self, pil_color, pil_mask=None):
        """Выветрить поверхность переднего плана. Возвращает PIL.Image или None.

        pil_mask (если задан) передаётся вторым reference-изображением, чтобы
        «заземлить» силуэт (маска сегментации: цветные кузов/груз, чёрный фон).
        """
        if not self.available():
            return None
        if random.random() > float(self.config.get("foreground_prob", 1.0)):
            return None
        prompt, tier = build_foreground_prompt()
        self._last_prompts["foreground"] = prompt
        self._last_prompts["foreground_tier"] = tier
        images = [pil_color]
        if pil_mask is not None:
            prompt += (
                " A segmentation mask is also provided (second image): the "
                "colored regions mark the truck body and cargo whose outlines "
                "must be preserved exactly."
            )
            images.append(pil_mask)
        out = self._call_generate(prompt, images)
        if out is not None and out.size != pil_color.size:
            try:
                from PIL import Image
                out = out.resize(pil_color.size, Image.LANCZOS)
            except Exception:
                return None
        return out

    def weather_full_scene(self, pil_color, pil_mask=None):
        """Режим single_call: фон + выветривание за ОДИН запрос.

        Возвращает PIL.Image (полный кадр) или None.
        """
        if not self.available():
            return None
        prompt, tier = build_combined_prompt()
        self._last_prompts["combined"] = prompt
        self._last_prompts["foreground_tier"] = tier
        images = [pil_color]
        if pil_mask is not None:
            prompt += (
                " A segmentation mask is also provided (second image): the "
                "colored regions mark the truck body and cargo whose outlines "
                "must be preserved exactly."
            )
            images.append(pil_mask)
        out = self._call_generate(prompt, images)
        if out is not None and out.size != pil_color.size:
            try:
                from PIL import Image
                out = out.resize(pil_color.size, Image.LANCZOS)
            except Exception:
                return None
        return out

    def generate_background(self, width, height):
        """Сгенерировать (или переиспользовать из кэша) фон нужного размера.

        Возвращает PIL.Image (RGB) или None. Кэш экономит квоту.
        """
        if not self.available():
            return None
        try:
            from PIL import Image
        except Exception:
            return None

        # Иногда переиспользуем недавний фон из кэша.
        reuse_p = float(self.config.get("background_reuse_prob", 0.5))
        cached = self._list_cache()
        if cached and random.random() < reuse_p:
            path = random.choice(cached)
            try:
                return Image.open(path).convert("RGB").resize(
                    (width, height), Image.LANCZOS)
            except Exception:
                pass  # битый файл — сгенерируем новый

        prompt = build_background_prompt()
        self._last_prompts["background"] = prompt
        img = self._call_generate(prompt, images=None)
        if img is None:
            # Откат: если кэш не пуст — берём оттуда.
            if cached:
                try:
                    return Image.open(random.choice(cached)).convert(
                        "RGB").resize((width, height), Image.LANCZOS)
                except Exception:
                    return None
            return None

        self._save_to_cache(img)
        try:
            return img.resize((width, height), Image.LANCZOS)
        except Exception:
            return img

    # ---- кэш фонов ----
    def _list_cache(self):
        try:
            return [
                os.path.join(_BG_CACHE_DIR, f)
                for f in os.listdir(_BG_CACHE_DIR)
                if f.lower().endswith(".png")
            ]
        except OSError:
            return []

    def _save_to_cache(self, img):
        try:
            os.makedirs(_BG_CACHE_DIR, exist_ok=True)
            fname = f"bg_{int(time.time()*1000)}_{random.randint(0,9999)}.png"
            img.save(os.path.join(_BG_CACHE_DIR, fname), format="PNG")
            # Ротация: держим не больше N последних.
            limit = int(self.config.get("background_cache_size", 24))
            files = sorted(
                self._list_cache(), key=lambda p: os.path.getmtime(p))
            for old in files[:-limit] if len(files) > limit else []:
                try:
                    os.remove(old)
                except OSError:
                    pass
        except Exception as exc:
            print(f"[Gemini] не удалось сохранить фон в кэш: {exc}")

    def last_prompts(self):
        return dict(self._last_prompts)


# ---------------------------------------------------------------------------
# Hugging Face провайдер (по умолчанию). Тот же интерфейс, что у Gemini:
# available / generate_background / weather_foreground / weather_full_scene /
# last_prompts / config — renderer_utils работает с любым провайдером.
# ---------------------------------------------------------------------------
class HuggingFaceProcessor:
    def __init__(self, config=None):
        self.config = config or load_gemini_config()
        self._session = None
        self._last_prompts = {}
        # Если провайдер hf-inference не поддерживает img2img для выбранной
        # модели — отключаем выветривание на сессию, чтобы не слать заведомо
        # падающие запросы на каждом кадре.
        self._edit_disabled = False

    def available(self):
        if not self.config.get("enabled", True):
            return False
        if not self.config.get("hf_token"):
            return False
        try:
            import requests  # noqa: F401
        except Exception:
            return False
        return True

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _post(self, model, payload, accept="image/png"):
        """POST к hf-inference. Возвращает PIL.Image или None."""
        base = self.config.get(
            "hf_base", "https://router.huggingface.co/hf-inference/models/")
        url = base + model
        headers = {
            "Authorization": f"Bearer {self.config['hf_token']}",
            "Accept": accept,
        }
        try:
            resp = self._get_session().post(
                url, headers=headers, json=payload,
                timeout=float(self.config.get("timeout", 60)))
        except Exception as exc:
            print(f"[HF] сетевая ошибка ({model}): {exc}")
            return None
        ct = resp.headers.get("content-type", "")
        if resp.status_code != 200:
            print(f"[HF] HTTP {resp.status_code} ({model}): {resp.text[:300]}")
            return None
        if "image" not in ct:
            print(f"[HF] ожидали изображение, пришло {ct}: {resp.text[:200]}")
            return None
        try:
            from PIL import Image
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as exc:
            print(f"[HF] не декодировал изображение: {exc}")
            return None

    def generate_background(self, width, height):
        if not self.available():
            return None
        prompt = build_background_prompt()
        self._last_prompts["background"] = prompt
        payload = {
            "inputs": prompt,
            "parameters": {"width": int(width), "height": int(height)},
        }
        return self._post(
            self.config.get("hf_text_model", "black-forest-labs/FLUX.1-schnell"),
            payload)

    def weather_foreground(self, pil_color, pil_mask=None):
        if not self.available() or self._edit_disabled:
            return None
        if random.random() > float(self.config.get("foreground_prob", 1.0)):
            return None
        prompt, tier = build_foreground_prompt()
        self._last_prompts["foreground"] = prompt
        self._last_prompts["foreground_tier"] = tier
        b64 = GeminiPostProcessor._img_to_b64(pil_color)
        if b64 is None:
            return None
        payload = {
            "inputs": b64,     # image-to-image: base64 исходной картинки
            "parameters": {
                "prompt": prompt,
                "strength": float(self.config.get("hf_strength", 0.55)),
            },
        }
        model = self.config.get(
            "hf_edit_model", "black-forest-labs/FLUX.1-Kontext-dev")
        out = self._post(model, payload)
        if out is None:
            # img2img недоступен на этом провайдере — выключаем на сессию.
            self._edit_disabled = True
            print("[HF] img2img недоступен (hf-inference) — выветривание "
                  "переднего плана отключено; фон продолжит генерироваться.")
        if out is not None and out.size != pil_color.size:
            try:
                from PIL import Image
                out = out.resize(pil_color.size, Image.LANCZOS)
            except Exception:
                return None
        return out

    def weather_full_scene(self, pil_color, pil_mask=None):
        # HF img2img и так меняет весь кадр; отдельного single_call не требуется.
        return self.weather_foreground(pil_color, pil_mask)

    def last_prompts(self):
        return dict(self._last_prompts)


# ---------------------------------------------------------------------------
# DeepInfra провайдер: делает и text-to-image (фон), и image-to-image edit
# (выветривание кузова/груза). Принимает исходную картинку как base64 data-URI
# (локальный рендер слать напрямую, без публичного хостинга). $5 бесплатных
# кредитов ежемесячно. Тот же интерфейс, что у остальных провайдеров.
# ---------------------------------------------------------------------------
class DeepInfraProcessor:
    def __init__(self, config=None):
        self.config = config or load_gemini_config()
        self._session = None
        self._last_prompts = {}
        self._edit_disabled = False

    def available(self):
        if not self.config.get("enabled", True):
            return False
        if not self.config.get("di_token"):
            return False
        try:
            import requests  # noqa: F401
        except Exception:
            return False
        return True

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _post(self, model, payload):
        """POST к DeepInfra inference. Возвращает PIL.Image или None."""
        base = self.config.get("di_base",
                               "https://api.deepinfra.com/v1/inference/")
        url = base + model
        headers = {"Authorization": f"Bearer {self.config['di_token']}"}
        try:
            resp = self._get_session().post(
                url, headers=headers, json=payload,
                timeout=float(self.config.get("timeout", 60)))
        except Exception as exc:
            print(f"[DeepInfra] сетевая ошибка ({model}): {exc}")
            return None
        if resp.status_code != 200:
            print(f"[DeepInfra] HTTP {resp.status_code} ({model}): "
                  f"{resp.text[:300]}")
            return None
        try:
            data = resp.json()
        except Exception as exc:
            print(f"[DeepInfra] не разобрал ответ: {exc}")
            return None
        return self._parse_image(data)

    @staticmethod
    def _parse_image(data):
        """Достать картинку из разных форматов ответа DeepInfra."""
        from PIL import Image
        # Возможные поля: images[], image_url, output[]
        candidates = []
        for key in ("images", "output"):
            v = data.get(key)
            if isinstance(v, list):
                candidates.extend(v)
        for key in ("image_url", "image"):
            v = data.get(key)
            if isinstance(v, str):
                candidates.append(v)
        for c in candidates:
            if not isinstance(c, str):
                continue
            try:
                if c.startswith("data:"):
                    b64 = c.split(",", 1)[1]
                    raw = base64.b64decode(b64)
                    return Image.open(io.BytesIO(raw)).convert("RGB")
                if c.startswith("http"):
                    import requests
                    r = requests.get(c, timeout=60)
                    if r.status_code == 200:
                        return Image.open(io.BytesIO(r.content)).convert("RGB")
                # голый base64
                raw = base64.b64decode(c)
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue
        print(f"[DeepInfra] в ответе не найдено изображение: keys={list(data)[:6]}")
        return None

    def generate_background(self, width, height):
        if not self.available():
            return None
        prompt = build_background_prompt()
        self._last_prompts["background"] = prompt
        payload = {"prompt": prompt, "width": int(width), "height": int(height)}
        return self._post(
            self.config.get("di_text_model", "black-forest-labs/FLUX-1-schnell"),
            payload)

    def weather_foreground(self, pil_color, pil_mask=None):
        if not self.available() or self._edit_disabled:
            return None
        if random.random() > float(self.config.get("foreground_prob", 1.0)):
            return None
        prompt, tier = build_foreground_prompt()
        self._last_prompts["foreground"] = prompt
        self._last_prompts["foreground_tier"] = tier
        b64 = GeminiPostProcessor._img_to_b64(pil_color)
        if b64 is None:
            return None
        payload = {
            "prompt": prompt,
            "image": "data:image/png;base64," + b64,
        }
        model = self.config.get("di_edit_model",
                                "black-forest-labs/FLUX.1-Kontext-dev")
        out = self._post(model, payload)
        if out is None:
            self._edit_disabled = True
            print("[DeepInfra] img2img недоступен — выветривание переднего "
                  "плана отключено на сессию; фон продолжит генерироваться.")
        elif out.size != pil_color.size:
            try:
                from PIL import Image
                out = out.resize(pil_color.size, Image.LANCZOS)
            except Exception:
                return None
        return out

    def weather_full_scene(self, pil_color, pil_mask=None):
        return self.weather_foreground(pil_color, pil_mask)

    def last_prompts(self):
        return dict(self._last_prompts)


# ---------------------------------------------------------------------------
# OpenAI GPT Image провайдер. Редактирует ВЕСЬ кадр за один запрос
# (/v1/images/edits, multipart): повреждения кузова + разнофракционный груз +
# новый фон одним промптом. Работает на PNG-байтах (без локальной обработки
# изображения) — наличие метода edit_whole сигналит renderer_utils
# использовать whole-image путь без матирования.
# ---------------------------------------------------------------------------
class OpenAIProcessor:
    def __init__(self, config=None):
        self.config = config or load_gemini_config()
        self._session = None
        self._last_prompts = {}

    def available(self):
        if not self.config.get("enabled", True):
            return False
        if not self.config.get("openai_api_key"):
            return False
        try:
            import requests  # noqa: F401
        except Exception:
            return False
        return True

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def edit_whole(self, png_bytes, shadow=False):
        """Отредактировать весь кадр. Вход/выход — PNG-байты (или None).
        shadow=True — добавить косую тень, рассекающую кузов+груз пополам."""
        if not self.available():
            return None
        prompt, tier = build_combined_prompt(shadow=shadow)
        self._last_prompts["combined"] = prompt
        self._last_prompts["foreground_tier"] = tier
        self._last_prompts["provider"] = "openai"
        url = "https://api.openai.com/v1/images/edits"
        headers = {"Authorization": f"Bearer {self.config['openai_api_key']}"}
        files = {"image": ("frame.png", png_bytes, "image/png")}
        data = {
            "model": self.config.get("openai_model", "gpt-image-1"),
            "prompt": prompt,
            "size": self.config.get("openai_size", "1536x1024"),
            "n": "1",
        }
        quality = self.config.get("openai_quality")
        if quality:
            data["quality"] = quality
        try:
            resp = self._get_session().post(
                url, headers=headers, files=files, data=data,
                timeout=float(self.config.get("timeout", 180)))
        except Exception as exc:
            print(f"[OpenAI] сетевая ошибка: {exc}")
            return None
        if resp.status_code != 200:
            print(f"[OpenAI] HTTP {resp.status_code}: {resp.text[:400]}")
            return None
        try:
            b64 = resp.json()["data"][0]["b64_json"]
            return base64.b64decode(b64)
        except Exception as exc:
            print(f"[OpenAI] не разобрал ответ: {exc} :: {resp.text[:200]}")
            return None

    def last_prompts(self):
        return dict(self._last_prompts)


def get_image_processor(config=None):
    """Фабрика провайдера постобработки по config['provider']."""
    cfg = config or load_gemini_config()
    provider = str(cfg.get("provider", "huggingface")).lower()
    if provider == "openai":
        return OpenAIProcessor(cfg)
    if provider == "gemini":
        return GeminiPostProcessor(cfg)
    if provider == "deepinfra":
        return DeepInfraProcessor(cfg)
    return HuggingFaceProcessor(cfg)
