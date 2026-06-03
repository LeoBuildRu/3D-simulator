# graphics_settings.py
# ---------------------------------------------------------------------------
# Graphics-preset management for the 3D simulator.
#
# The simulator can render with two different engines:
#
#   * tobspr RenderPipeline (deferred, heavy) — used by the "ultra" and
#     "medium" presets. "medium" points RP at a trimmed config directory
#     (fewer plugins, smaller shadow maps) via MountManager.config_dir.
#   * panda3d-simplepbr (lightweight forward PBR with a sun + shadows) —
#     used by the "performance" preset, which runs on Intel iGPUs.
#
# Which engine to start with is decided BEFORE the Panda3D window exists
# (RenderPipeline must be constructed before ShowBase), so the active
# preset is persisted to config/graphics.json and resolved at startup.
# On the very first launch (no config yet) we auto-detect the GPU and fall
# back to "performance" on integrated Intel graphics, "ultra" otherwise.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import subprocess
import sys

# src/core/graphics_settings.py -> src/core -> src -> <project root>
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "graphics.json")

DEFAULT_PRESET = "ultra"

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------
# engine            : "render_pipeline" | "simplepbr"
# name              : human-readable Russian label for the UI combo
# rp_config_dir     : (render_pipeline only) abs path to the RP config dir,
#                     or None to use the default render_pipeline/config
# shadow_resolution : (simplepbr only) directional shadow map size in px
# msaa              : (simplepbr only) MSAA sample count (0 = off)
# max_lights        : (simplepbr only) max simultaneous lights
# enable_fog        : (simplepbr only) add light distance fog
# ---------------------------------------------------------------------------
PRESETS: dict = {
    "ultra": {
        "name": "Ультра (RenderPipeline)",
        "engine": "render_pipeline",
        "rp_config_dir": None,
    },
    "medium": {
        "name": "Среднее (RenderPipeline)",
        "engine": "render_pipeline",
        "rp_config_dir": os.path.join(PROJECT_ROOT, "render_pipeline",
                                      "config_medium"),
    },
    "performance": {
        "name": "Производительность (simplepbr)",
        "engine": "simplepbr",
        "shadow_resolution": 2048,
        "msaa": 0,
        "max_lights": 4,
        "enable_fog": True,
    },
}

# Stable display order for the UI combo.
PRESET_ORDER = ("ultra", "medium", "performance")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def get_preset(name: str) -> dict:
    """Return the preset definition for `name`, falling back to the default."""
    return PRESETS.get(name, PRESETS[DEFAULT_PRESET])


def load_saved() -> str | None:
    """
    Return the preset key stored in config/graphics.json, or None if the
    file does not exist / is unreadable / holds an unknown preset.
    """
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    preset = data.get("preset") if isinstance(data, dict) else None
    return preset if preset in PRESETS else None


def save(name: str) -> None:
    """Persist the chosen preset key to config/graphics.json."""
    if name not in PRESETS:
        name = DEFAULT_PRESET
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump({"preset": name}, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"[graphics] failed to save preset '{name}': {exc}")


# ---------------------------------------------------------------------------
# GPU auto-detection (first run only)
# ---------------------------------------------------------------------------
def detect_gpu_is_integrated() -> bool:
    """
    Best-effort check whether the machine only has integrated graphics
    (Intel iGPU / Microsoft Basic adapter) and no discrete NVIDIA/AMD card.

    Done without a GL context (we decide the engine before the window
    exists) by querying Win32_VideoController via PowerShell. Any failure
    returns False, so we never accidentally downgrade a capable machine.
    """
    if sys.platform != "win32":
        return False

    try:
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object -ExpandProperty Name",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:  # noqa: BLE001 - detection must never crash startup
        print(f"[graphics] GPU detection failed: {exc}")
        return False

    names = [ln.strip().lower() for ln in completed.stdout.splitlines()
             if ln.strip()]
    if not names:
        return False

    # A discrete card present -> not integrated-only.
    for n in names:
        if "nvidia" in n or "geforce" in n or "radeon" in n or "amd " in n:
            return False

    # Otherwise: integrated only if we actually see an Intel / basic adapter.
    for n in names:
        if "intel" in n or "microsoft basic" in n or "uhd" in n or "iris" in n:
            return True

    return False


def resolve_startup_preset() -> str:
    """
    Resolve the preset to start with:
      * a previously saved valid preset wins;
      * otherwise (first run) auto-detect — "performance" on integrated
        Intel graphics, "ultra" on a discrete card — and persist it.
    """
    saved = load_saved()
    if saved is not None:
        return saved

    chosen = "performance" if detect_gpu_is_integrated() else DEFAULT_PRESET
    print(f"[graphics] first run, auto-selected preset: '{chosen}'")
    save(chosen)
    return chosen
