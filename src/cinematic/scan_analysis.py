# -*- coding: utf-8 -*-
"""
Запуск анализа проезда (scan_worker.py) отдельным процессом.

Воркеру нужен интерпретатор с direct_mesh и volume_calculator (Python 3.11:
torch + Depth Anything 3, open3d, scipy). В утилите их нет и не нужно: расчёт
в своём процессе ещё и не делит GIL с рендером — кадры во время анализа не
проседают.

Интерпретатор ищется: переменная окружения IQOKO_CINE_PYTHON, затем известные
места установки Python 3.11. Нет его — `available()` скажет почему, и
кинематограф пойдёт без снимка и без поиска кузова.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import threading
from typing import Callable, Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER = os.path.join(PROJECT_ROOT, "src", "cinematic", "scan_worker.py")

_PY_CANDIDATES = [
    os.environ.get("IQOKO_CINE_PYTHON", ""),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", "Python311", "python.exe"),
    r"C:\Program Files\Python311\python.exe",
]


def find_python() -> Optional[str]:
    for p in _PY_CANDIDATES:
        if p and os.path.isfile(p):
            return p
    for p in glob.glob(os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs",
                                    "Python", "Python31*", "python.exe")):
        return p
    return None


def available() -> Dict[str, object]:
    py = find_python()
    return {"python": py, "worker": os.path.isfile(WORKER),
            "reason": "" if py else "не найден Python 3.11 с direct_mesh (IQOKO_CINE_PYTHON)"}


class Analysis:
    """Результат одной части анализа: словарь массивов npz."""

    def __init__(self, path: str):
        with np.load(path, allow_pickle=False) as z:
            self.data = {k: z[k] for k in z.files}
        import json
        try:
            self.notes = json.loads(str(self.data.get("notes", "[]")))
        except Exception:
            self.notes = []

    def get(self, key, default=None):
        return self.data.get(key, default)

    def has(self, *keys) -> bool:
        return all(k in self.data for k in keys)


def run_worker(parts: str, ply: str, json_path: str, out: str, *,
               photo: Optional[str] = None, napolnitel: Optional[str] = None,
               mesh_before: Optional[str] = None, wall_mask: Optional[str] = None,
               on_stage: Optional[Callable[[str], None]] = None,
               cancel: Optional[threading.Event] = None) -> Optional[Analysis]:
    """
    Выполнить часть анализа (блокирует; звать из фонового потока).
    Результат кэшируется рядом с PLY: повторный показ того же проезда не
    пересчитывается, пока входные файлы не изменились.
    """
    py = find_python()
    if not py:
        return None
    inputs = [p for p in (ply, json_path, photo, napolnitel, mesh_before, wall_mask) if p]
    if os.path.isfile(out) and all(os.path.getmtime(out) >= os.path.getmtime(p)
                                   for p in inputs if os.path.isfile(p)) \
            and os.path.getmtime(out) >= os.path.getmtime(WORKER):
        try:
            return Analysis(out)
        except Exception:
            pass
    cmd: List[str] = [py, "-u", WORKER, "--parts", parts, "--ply", ply, "--json", json_path,
                      "--out", out]
    for flag, val in (("--photo", photo), ("--napolnitel", napolnitel),
                      ("--mesh-before", mesh_before), ("--wall-mask", wall_mask)):
        if val and os.path.isfile(val):
            cmd += [flag, val]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=env, creationflags=flags, cwd=PROJECT_ROOT)
    for raw in proc.stdout:
        line = raw.decode("utf-8", "replace").rstrip()
        if cancel is not None and cancel.is_set():
            proc.kill()
            return None
        if line.startswith("@@STAGE "):
            if on_stage:
                on_stage(line[8:])
        elif line.startswith("[cine-worker]"):
            print(line)
    proc.wait()
    if proc.returncode != 0 or not os.path.isfile(out):
        print(f"[cine] анализ ({parts}) завершился с кодом {proc.returncode}")
        return None
    return Analysis(out)
