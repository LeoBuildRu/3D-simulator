import sys
import os
from cx_Freeze import setup, Executable

# -----------------------------------------------------
# ПАПКА ПРОЕКТА
# -----------------------------------------------------
current_dir = os.path.abspath(os.getcwd())
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "render_pipeline"))

# -----------------------------------------------------
# INCLUDE FILES — кладём всё, что нужно EXE
# -----------------------------------------------------
include_files = [
    ("models", "models"),
    ("textures", "textures"),
    ("PLY_examples", "PLY_examples"),
    ("height_examples", "height_examples"),
    ("fonts", "fonts"),

    ("models_config.yaml", "models_config.yaml"),
    ("textures_config.yaml", "textures_config.yaml"),

    ("mesh_distribution.py", "mesh_distribution.py"),
    ("falling_particles.py", "falling_particles.py"),

    # RenderPipeline целиком
    ("render_pipeline", "render_pipeline"),

    # RenderPipeline (обязательные папки)
    ("render_pipeline/config", "lib/config"),
    ("render_pipeline/effects", "lib/effects"),
    ("render_pipeline/data", "lib/data"),
    ("render_pipeline/rpplugins", "lib/rpplugins"),
]

# 🔥 ВАЖНО: включаем исходники Warp (иначе Inspect ломается)
# cx_Freeze обычно кладёт только .pyc → Warp не работает без .py
import warp
warp_path = os.path.dirname(warp.__file__)
include_files.append((warp_path, "warp"))

# -----------------------------------------------------
# PACKAGES
# -----------------------------------------------------
packages = [
    "yaml", "trimesh", "numpy", "scipy", "PIL", "tkinter",
    "panda3d", "direct", "noise", "warp", "point_cloud_utils",
    "requests", "win32gui", "win32con",
    "gltf", "simplepbr", "pyembree",
    "csg", "scipy.spatial", "scipy.ndimage",
    "pygame", "pyglm", "pygltflib", "pyrr",
    "ezdxf", "manifold3d", "matplotlib", "moderngl",
    "networkx", "stl", "opensimplex", "packaging",
    "pandas", "pybind11", "OpenGL", "OpenGL_accelerate",
    "pyopengltk", "python_utils", "rtree", "skimage",
    "tifffile", "typing_extensions", "wrapt",

    # RenderPipeline
    "render_pipeline",
    "render_pipeline.rpcore",
    "render_pipeline.rpplugins",
]

# -----------------------------------------------------
# EXCLUDES
# -----------------------------------------------------
excludes = [
    "unittest", "test",
    "PyQt5.QtQml", "PyQt5.QtQuick",

    "render_pipeline.rplibs.yaml.yaml_py2",
    "render_pipeline.rplibs.yaml.yaml_py2.*",
    "render_pipeline.rplibs.yaml.yaml_py3",
]

# -----------------------------------------------------
# BUILD OPTIONS
# -----------------------------------------------------
build_exe_options = {
    "packages": packages,
    "excludes": excludes,
    "include_files": include_files,

    # ❗ ВАЖНО: Оставляем .py-файлы, иначе Warp ломается
    "optimize": 0,

    # ❗ Ничего не упаковываем в ZIP → исходники доступны Warp
    "zip_include_packages": "",
    "zip_exclude_packages": "*",

    # Для Windows (MSVC runtime)
    "include_msvcr": True,
}

# -----------------------------------------------------
# EXECUTABLE
# -----------------------------------------------------
executables = [
    Executable(
        script="main.py",
        base=None,
        target_name="3D_Simulator.exe"
    )
]

# -----------------------------------------------------
# SETUP
# -----------------------------------------------------
setup(
    name="3D Simulator",
    version="1.0",
    description="3D Visualization Tool",
    options={"build_exe": build_exe_options},
    executables=executables
)