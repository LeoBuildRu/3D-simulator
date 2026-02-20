import sys
import os
from cx_Freeze import setup, Executable

current_dir = os.path.abspath(os.getcwd())
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "render_pipeline"))

# -----------------------------------------------------------------------------
# INCLUDE FILES — кладём всё в папки рядом с EXE
# -----------------------------------------------------------------------------
include_files = [
    ("models", "models"),
    ("textures", "textures"),
    ("lidar_example", "lidar_example"),

    ("models_config.yaml", "models_config.yaml"),
    ("textures_config.yaml", "textures_config.yaml"),

    # RenderPipeline целиком (на всякий случай)
    ("render_pipeline", "render_pipeline"),

    # RenderPipeline для MountManager (ОБЯЗАТЕЛЬНЫЕ ПАПКИ)
    ("render_pipeline/config", "lib/config"),
    ("render_pipeline/effects", "lib/effects"),
    ("render_pipeline/data", "lib/data"),
    ("render_pipeline/rpplugins", "lib/rpplugins"),
]

# ЕСЛИ хочешь копировать rpplugins в lib (необязательно, но можно)
include_files += [
    ("render_pipeline/rpplugins", "lib/rpplugins"),
]


# -----------------------------------------------------------------------------
# PACKAGES — оставляем всё как у тебя
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# EXCLUDES — оставляем
# -----------------------------------------------------------------------------
excludes = [
    "unittest", "test",
    "PyQt5.QtQml", "PyQt5.QtQuick",

    "render_pipeline.rplibs.yaml.yaml_py2",
    "render_pipeline.rplibs.yaml.yaml_py2.*",
    "render_pipeline.rplibs.yaml.yaml_py3",
]


# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------
build_exe_options = {
    "packages": packages,
    "excludes": excludes,
    "include_files": include_files,
    "zip_include_packages": "",
    "zip_exclude_packages": "*",
    "include_msvcr": True,
}


# -----------------------------------------------------------------------------
# EXECUTABLE
# -----------------------------------------------------------------------------
executables = [
    Executable(
        script="main.py",
        base=None,
        target_name="3D_Simulator.exe"
    )
]


# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
setup(
    name="3D Simulator",
    version="1.0",
    description="3D Visualization Tool",
    options={"build_exe": build_exe_options},
    executables=executables
)