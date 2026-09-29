# -*- coding: utf-8 -*-
"""
Данные кинематографичной реконструкции и их фоновая загрузка.

Всё грузится параллельно, сценарий ждёт только то, что нужно его текущему
этапу (флаги `*_ready`):

  поток A: JSON -> PLY -> снимок станции (photo_ready)
           -> анализ «scene»: фото-меш, облако, поиск кузова (scene_ready)
  поток B: штатная выборка утилиты (_fetch_reconstruction: наборы текстур и
           модели, готовый меш наполнителя) (fetch_ready)
           -> серверные артефакты (_napolnitel, _mesh_before, _wall_mask)
           -> анализ «fill»: карты высот этапов (fill_ready)
  поток C: кузова-кандидаты для подбора: .bam кузова + точки проёма
           (bodies_ready)

Сеть и диск отпускают GIL, анализ идёт в отдельном процессе — кадры во время
загрузки не проседают.
"""

from __future__ import annotations

import ast
import json
import os
import random
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import scan_analysis

RECON_DIR = os.path.join(tempfile.gettempdir(), "vizutil_recon")


@dataclass
class Body:
    key: str
    name: str
    points_3d: np.ndarray            # (4,3) углы проёма в системе модели
    width: float
    length: float
    correct: bool
    path: str = ""
    model: Any = None                # NodePath, загруженный в фоне


@dataclass
class CineData:
    rec: Any
    stem: str = ""
    json: Dict[str, Any] = field(default_factory=dict)
    json_path: str = ""
    ply_path: str = ""
    photo_path: str = ""
    scene: Optional[scan_analysis.Analysis] = None
    fill: Optional[scan_analysis.Analysis] = None
    fetched: Optional[dict] = None
    bodies: List[Body] = field(default_factory=list)
    status: str = ""
    error: str = ""

    json_ready: bool = False
    ply_done: bool = False           # PLY получен (или его нет) — файлы на диске
    photo_ready: bool = False
    scene_ready: bool = False
    fetch_ready: bool = False
    fill_ready: bool = False
    bodies_ready: bool = False

    # ------------------------------------------------------------------ #
    @property
    def direction_yaw(self) -> bool:
        return bool(self.json.get("direction_yaw_applied"))

    @property
    def fill_ratio(self) -> Optional[float]:
        vol = self.json.get("target_volume")
        mv = None
        f = self.fetched or {}
        cfg = f.get("model_cfg") or {}
        mv = cfg.get("max_volume")
        try:
            return float(vol) / float(mv) if vol is not None and mv else None
        except (TypeError, ValueError, ZeroDivisionError):
            return None


def parse_points(val) -> Optional[np.ndarray]:
    if val is None:
        return None
    if isinstance(val, str):
        try:
            val = ast.literal_eval(val)
        except Exception:
            return None
    try:
        a = np.asarray(val, np.float64)
        return a if a.shape == (4, 3) else None
    except Exception:
        return None


def rect_size(p: np.ndarray):
    """(ширина, длина) прямоугольника из четырёх углов по часовой."""
    a = np.linalg.norm(p[1] - p[0])
    b = np.linalg.norm(p[2] - p[1])
    return (min(a, b), max(a, b))


class Loader:
    """Фоновая загрузка всего, что нужно сценарию."""

    def __init__(self, window, rec):
        self.win = window
        self.app = window.panda_app
        self.data = CineData(rec=rec, stem=os.path.splitext(rec.name)[0])
        self.cancel = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="cine")

    def start(self) -> CineData:
        os.makedirs(RECON_DIR, exist_ok=True)
        self._pool.submit(self._guard, self._scene_chain)
        self._pool.submit(self._guard, self._fetch_chain)
        self._pool.submit(self._guard, self._bodies_chain)
        return self.data

    def shutdown(self) -> None:
        self.cancel.set()
        self._pool.shutdown(wait=False)

    def _guard(self, fn):
        try:
            fn()
        except Exception as exc:
            traceback.print_exc()
            self.data.error = self.data.error or f"{type(exc).__name__}: {exc}"

    def _say(self, text: str) -> None:
        self.data.status = text

    def _download(self, name: str) -> Optional[str]:
        local = os.path.join(RECON_DIR, name)
        if os.path.isfile(local) and os.path.getsize(local) > 0:
            return local
        try:
            self.app.tls_client.download_file(name, local)
            return local if os.path.isfile(local) else None
        except Exception as exc:
            print(f"[cine] {name}: {exc}")
            return None

    # ------------------------------------------------------------------ #
    def _scene_chain(self):
        d = self.data
        rec = d.rec
        self._say("получение проезда…")
        d.json_path = self.win._resolve_recon_json(rec) or ""
        if not d.json_path:
            d.error = "не удалось получить JSON проезда"
            d.ply_done = True
            return
        with open(d.json_path, "r", encoding="utf-8") as fh:
            d.json = json.load(fh)
        d.json_ready = True
        ply = d.json.get("ply_file") or rec.ply_file
        if ply:
            d.ply_path = self.win._resolve_recon_ply(rec, ply, d.json_path) or ""
        d.ply_done = True
        self._say("снимок станции…")
        d.photo_path = self._download(d.stem + "_topdown.jpg") or ""
        d.photo_ready = True
        if not d.ply_path:
            d.error = "нет облака точек (PLY)"
            d.scene_ready = True
            return
        info = scan_analysis.available()
        if not info["python"]:
            print(f"[cine] анализ недоступен: {info['reason']}")
            d.scene_ready = True
            return
        d.scene = scan_analysis.run_worker(
            "scene", d.ply_path, d.json_path,
            os.path.join(RECON_DIR, d.stem + ".cine_scene.npz"),
            photo=d.photo_path or None, on_stage=self._say, cancel=self.cancel)
        d.scene_ready = True

    def _fetch_chain(self):
        d = self.data
        # JSON и PLY качает цепочка снимка — две записи в один файл испортили бы его
        while not d.ply_done and not self.cancel.is_set():
            self.cancel.wait(0.05)
        out = self.win._fetch_reconstruction(d.rec)
        d.fetched = out
        d.fetch_ready = True
        if isinstance(out, dict) and "error" in out:
            d.error = str(out["error"])
            d.fill_ready = True
            return
        # серверные артефакты этапов наполнителя
        arts = {suf: self._download(d.stem + suf)
                for suf in ("_napolnitel.obj", "_wall_mask.obj", "_mesh_before.obj")}
        if arts["_napolnitel.obj"] and d.ply_path and scan_analysis.available()["python"]:
            d.fill = scan_analysis.run_worker(
                "fill", d.ply_path, d.json_path,
                os.path.join(RECON_DIR, d.stem + ".cine_fill.npz"),
                napolnitel=arts["_napolnitel.obj"], mesh_before=arts["_mesh_before.obj"],
                wall_mask=arts["_wall_mask.obj"], cancel=self.cancel)
        d.fill_ready = True

    def _bodies_chain(self):
        d = self.data
        from src.ui.panel_data import get_model_set_config, load_model_sets
        while not d.json_ready and not self.cancel.is_set() and not d.error:
            self.cancel.wait(0.1)
        model_name = d.json.get("model") or d.rec.model
        correct_key = self.win._find_model_key_by_name(model_name) if model_name else None
        keys = [k for k, _ in load_model_sets()]
        pool = []
        for k in keys:
            cfg = get_model_set_config(k) or {}
            pts = parse_points(cfg.get("points_3d"))
            if pts is None or not cfg.get("cuzov"):
                continue
            pool.append((k, cfg, pts))
        rng = random.Random(d.stem)
        others = [p for p in pool if p[0] != correct_key]
        rng.shuffle(others)
        # непохожие по длине — подбор выглядит осмысленно
        chosen = []
        for p in others:
            w, l = rect_size(p[2])
            if all(abs(l - rect_size(c[2])[1]) > 0.6 for c in chosen):
                chosen.append(p)
            if len(chosen) == 2:
                break
        correct = next((p for p in pool if p[0] == correct_key), None)
        # Случайные кандидаты больше не показываются — только нужный кузов.
        entries = [(correct, True)] if correct else []
        cache = self.app.get_cache_dir()
        for (key, cfg, pts), ok in entries:
            local = os.path.join(cache, os.path.basename(cfg["cuzov"]))
            if not os.path.isfile(local):
                try:
                    self.app.tls_client.download_model_file(key, "cuzov", local)
                except Exception as exc:
                    print(f"[cine] кузов {key}: {exc}")
                    continue
            w, l = rect_size(pts)
            body = Body(key=key, name=str(cfg.get("model") or key), points_3d=pts,
                        width=w, length=l, correct=ok, path=local)
            try:
                from panda3d.core import Filename
                body.model = self.app.loader.load_model(Filename.from_os_specific(local),
                                                        noCache=True)
            except Exception as exc:
                print(f"[cine] кузов {key} не загружен: {exc}")
                continue
            d.bodies.append(body)
        d.bodies_ready = True
