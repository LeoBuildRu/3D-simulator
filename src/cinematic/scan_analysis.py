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


class WorkerServer:
    """
    Постоянный процесс анализа (scan_worker.py --serve): torch, Depth Anything
    на GPU и open3d грузятся и прогреваются один раз, при старте утилиты, —
    а не при каждом проезде (это были секунды ожидания посреди сцены).
    Запросы выполняются по одному; ход работы пересылается вызывающему.
    """

    _instance: Optional["WorkerServer"] = None

    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.ready = threading.Event()
        self.failed = False
        self.warm_seconds = None
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._result = None
        self._on_stage = None
        self._seq = 0

    @classmethod
    def get(cls) -> Optional["WorkerServer"]:
        """Запущенный (или запускаемый) сервер; None — Python 3.11 нет."""
        if cls._instance is None or not cls._instance.alive:
            if not find_python():
                return None
            srv = cls()
            srv._start()
            cls._instance = srv
        return cls._instance

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _start(self) -> None:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen([find_python(), "-u", WORKER, "--serve"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env=env, creationflags=flags,
                                     cwd=PROJECT_ROOT)
        threading.Thread(target=self._reader, daemon=True, name="cine-worker-io").start()

    def _reader(self) -> None:
        for raw in self.proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip()
            if line.startswith("@@READY"):
                try:
                    self.warm_seconds = float(line.split()[1])
                except (IndexError, ValueError):
                    pass
                print(f"[cine] анализ прогрет за {self.warm_seconds} с")
                self.ready.set()
            elif line.startswith("@@STAGE "):
                cb = self._on_stage
                if cb:
                    cb(line[8:])
            elif line.startswith("@@DONE") or line.startswith("@@FAIL"):
                self._result = line.startswith("@@DONE")
                if not self._result:
                    print(f"[cine] {line}")
                self._done.set()
            elif line.startswith("[cine-worker]"):
                print(line)
        # процесс кончился
        self.failed = True
        self.ready.set()
        self._result = False
        self._done.set()

    def request(self, argv: List[str], on_stage=None, cancel=None, timeout=600.0) -> bool:
        """Выполнить один запуск воркера (блокирует; из фонового потока)."""
        self.ready.wait(timeout=180)
        if not self.alive:
            return False
        with self._lock:
            self._seq += 1
            self._done.clear()
            self._result = None
            self._on_stage = on_stage
            import json as _json
            msg = _json.dumps({"id": self._seq, "argv": argv}) + chr(10)
            try:
                self.proc.stdin.write(msg.encode("utf-8"))
                self.proc.stdin.flush()
            except OSError:
                return False
            waited = 0.0
            while not self._done.wait(0.2):
                waited += 0.2
                if (cancel is not None and cancel.is_set()) or waited > timeout:
                    return False
            self._on_stage = None
            return bool(self._result)


def warm_up() -> None:
    """Запустить постоянный анализ заранее (при старте утилиты)."""
    WorkerServer.get()


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
    argv: List[str] = ["--parts", parts, "--ply", ply, "--json", json_path, "--out", out]
    for flag, val in (("--photo", photo), ("--napolnitel", napolnitel),
                      ("--mesh-before", mesh_before), ("--wall-mask", wall_mask)):
        if val and os.path.isfile(val):
            argv += [flag, val]
    srv = WorkerServer.get()
    if srv is not None and not srv.failed:
        if srv.request(argv, on_stage=on_stage, cancel=cancel) and os.path.isfile(out):
            return Analysis(out)
        if cancel is not None and cancel.is_set():
            return None
        print("[cine] постоянный анализ не справился — одноразовый запуск")
    cmd: List[str] = [py, "-u", WORKER] + argv
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
