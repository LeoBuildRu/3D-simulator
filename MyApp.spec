# -*- mode: python ; coding: utf-8 -*-

import os
import panda3d
import warp
from PyInstaller.utils.hooks import collect_submodules

# ----------------------------------------------------------------------
# 1. Определяем путь к установке Panda3D
# ----------------------------------------------------------------------
panda_root = os.path.dirname(panda3d.__file__)
print(">>> panda_root =", panda_root)

# ----------------------------------------------------------------------
# 2. Скрытые импорты
# ----------------------------------------------------------------------
panda_hidden = collect_submodules("panda3d") + collect_submodules("direct")
rp_hidden = (
    collect_submodules("rpcore")
    + collect_submodules("rpplugins")
    + collect_submodules("rplibs")
)

hidden = (
    panda_hidden
    + rp_hidden
    + [
        "PyQt5.sip",
        "noise",
        "trimesh",
        "scipy",
        "PIL",
        "yaml",
        "point_cloud_utils",
        "scipy.special._cdflib",
        "win32gui",
        "win32con",
    ]
)

# ----------------------------------------------------------------------
# 3. Собираем все DLL Panda3D
# ----------------------------------------------------------------------
binaries = []

def collect_dlls_from_dir(path):
    if not os.path.isdir(path):
        print(f"    {path} not found")
        return
    print(f"    Scanning {path}")
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.dll'):
                full_path = os.path.join(root, file)
                binaries.append((full_path, '.'))
                print(f"      Added {file}")

# Сканируем возможные места хранения DLL
collect_dlls_from_dir(panda_root)
collect_dlls_from_dir(os.path.join(panda_root, "plugins"))
collect_dlls_from_dir(os.path.join(panda_root, "bin"))
collect_dlls_from_dir(os.path.join(panda_root, "libs"))

# ----------------------------------------------------------------------
# 4. Бинарные файлы Warp
# ----------------------------------------------------------------------
warp_bin_path = os.path.join(os.path.dirname(warp.__file__), "bin")
if os.path.isdir(warp_bin_path):
    for f in os.listdir(warp_bin_path):
        binaries.append((os.path.join(warp_bin_path, f), "warp/bin"))

# ----------------------------------------------------------------------
# 5. Статические данные (ресурсы проекта и конфиги)
# ----------------------------------------------------------------------
datas = [
    # Папки и файлы вашего проекта – все копируются в корень (target = ".")
    ("models", "models"),                     # вся папка models со всем содержимым
    ("textures", "textures"),                  # вся папка textures со всем содержимым
    ("render_pipeline", "render_pipeline"),    # папка render_pipeline
    ("models_config.yaml", "."),                # конфиг моделей – в корень
    ("textures_config.yaml", "."),              # конфиг текстур – в корень
    ("mesh_distribution.py", "."),              # дополнительные скрипты – в корень
    ("falling_particles.py", "."),
    # Конфигурация Panda3D (папка etc внутри panda_root) – копируется в корень как etc/
    (os.path.join(panda_root, "etc"), "etc"),
]

# ----------------------------------------------------------------------
# 6. Анализ и сборка
# ----------------------------------------------------------------------
a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    runtime_hooks=[],
    noarchive=False,
    module_collection_mode={"warp": "py"},
)

pyz = PYZ(a.pure, optimize=0)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MyApp",
    debug=True,
    console=True,
    upx=False,
    strip=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="MyApp",
)