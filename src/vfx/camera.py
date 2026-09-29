# -*- coding: utf-8 -*-
"""
Кинокамера.

Пока идёт сцена, камерой пользователя (`base.camera`) и её линзой управляет
риг: позиция, точка взгляда, вектор «вверх», фокусное расстояние и сдвиг
плёнки. Всё интерполируется, поэтому переход из ракурса станции (снимок) в
обычный облёт непрерывен — нужен только общий язык: «откуда, куда, где верх,
какой объектив».

Поверх позы — «дыхание» ручной камеры (сумма медленных синусов), по
умолчанию едва заметное.

По окончании `release()` возвращает линзу, позу и управление FlyCamera.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from panda3d.core import LVecBase2f, Point3, Vec2, Vec3

from . import easing as ease
from .sequencer import Cmd


def _v(p) -> Vec3:
    return Vec3(float(p[0]), float(p[1]), float(p[2]))


class Shot:
    """Поза камеры: где стоит, куда смотрит, где верх, объектив."""

    __slots__ = ("pos", "target", "up", "fov", "offset")

    def __init__(self, pos, target, up=(0, 0, 1), fov: float = 50.0,
                 offset=(0.0, 0.0)):
        self.pos = _v(pos)
        self.target = _v(target)
        self.up = _v(up)
        self.fov = float(fov)            # вертикальный угол, градусы
        self.offset = Vec2(*offset)      # сдвиг плёнки, доли высоты кадра

    def copy(self) -> "Shot":
        return Shot(self.pos, self.target, self.up, self.fov, self.offset)

    @staticmethod
    def mix(a: "Shot", b: "Shot", t: float) -> "Shot":
        up = a.up * (1 - t) + b.up * t
        if up.length_squared() < 1e-8:
            up = b.up
        up.normalize()
        return Shot(a.pos + (b.pos - a.pos) * t, a.target + (b.target - a.target) * t,
                    up, a.fov + (b.fov - a.fov) * t, a.offset + (b.offset - a.offset) * t)


def _catmull(p0, p1, p2, p3, t):
    # векторы Panda умножаются на число только справа
    t2, t3 = t * t, t * t * t
    return (p1 * 2.0 + (p2 - p0) * t + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
            + (p1 * 3.0 - p0 - p2 * 3.0 + p3) * t3) * 0.5


class _MoveCmd(Cmd):
    def __init__(self, rig, shots, duration, curve):
        self.rig, self.shots = rig, shots
        self.duration = max(1e-6, duration)
        self.curve = curve
        self.t = 0.0

    def start(self, seq):
        self.shots = [self.rig.shot.copy()] + list(self.shots)

    def step(self, dt):
        self.t += dt
        k = min(1.0, self.t / self.duration)
        self.rig.shot = self.rig.sample_path(self.shots, self.curve(k))
        return k >= 1.0


class _OrbitCmd(Cmd):
    def __init__(self, rig, center, radius, height, a0, a1, duration, curve, fov,
                 target_offset):
        self.rig = rig
        self.center = _v(center)
        self.radius, self.height = radius, height
        self.a0, self.a1 = a0, a1
        self.duration = max(1e-6, duration)
        self.curve, self.fov = curve, fov
        self.target_offset = _v(target_offset)
        self.t = 0.0

    def shot_at(self, k):
        a = math.radians(self.a0 + (self.a1 - self.a0) * k)
        r = self.radius if not callable(self.radius) else self.radius(k)
        h = self.height if not callable(self.height) else self.height(k)
        pos = self.center + Vec3(math.cos(a) * r, math.sin(a) * r, h)
        fov = self.fov if self.fov is not None else self.rig.shot.fov
        return Shot(pos, self.center + self.target_offset, (0, 0, 1), fov)

    def step(self, dt):
        self.t += dt
        k = min(1.0, self.t / self.duration)
        self.rig.shot = self.shot_at(self.curve(k))
        return k >= 1.0


class CinematicCamera:
    """Риг поверх `base.camera` и `base.camLens`."""

    def __init__(self, base, space=None):
        self.base = base
        #: система координат, в которой заданы позы (узел абстрактного
        #: мира); None — render
        self.space = space
        self.shot = Shot((0, -10, 5), (0, 0, 0))
        self.shake = 0.0                  # амплитуда «дыхания», метры
        self.photo: Optional[dict] = None # объектив станции (см. photo_lens)
        self.lens_mix = 0.0               # 1 — объектив станции, 0 — обычный
        self._saved = None
        self._task = None
        self._time = 0.0

    # ------------------------------------------------------------------ #
    def capture_user(self) -> Shot:
        """Текущая поза пользователя как Shot (в системе `space`)."""
        cam = self.base.camera
        ref = self.space or self.base.render
        pos = ref.get_relative_point(cam, Point3(0, 0, 0))
        fwd = ref.get_relative_vector(cam, Vec3(0, 1, 0))
        up = ref.get_relative_vector(cam, Vec3(0, 0, 1))
        return Shot(pos, pos + fwd * 10.0, up, self.base.camLens.get_fov()[1])

    def acquire(self) -> None:
        """Забрать камеру у пользователя (поза и линза запоминаются)."""
        if self._saved is not None:
            return
        base = self.base
        lens = base.camLens
        self._saved = dict(
            parent=base.camera.get_parent(), mat=base.camera.get_mat(),
            fov=LVecBase2f(lens.get_fov()), film=LVecBase2f(lens.get_film_size()),
            offset=LVecBase2f(lens.get_film_offset()), near=lens.get_near(),
            far=lens.get_far(), focal=lens.get_focal_length())
        fly = getattr(base, "fly_cam", None)
        if fly is not None:
            self._saved["fly_frozen"] = fly.is_frozen()
            fly.set_frozen(True)
        self.shot = self.capture_user()
        self._task = base.taskMgr.add(self._apply, "vfx_camera", sort=45)

    def release(self, keep_pose: bool = True) -> None:
        """Вернуть камеру. keep_pose — остаться там, где закончилась сцена."""
        if self._saved is None:
            return
        base = self.base
        s = self._saved
        if self._task is not None:
            base.taskMgr.remove(self._task)
            self._task = None
        lens = base.camLens
        lens.set_film_offset(s["offset"])
        lens.set_fov(s["fov"])
        lens.set_near_far(s["near"], s["far"])
        if not keep_pose:
            base.camera.reparent_to(s["parent"])
            base.camera.set_mat(s["mat"])
        else:
            # поза из пространства сцены — в систему пользователя
            mat = base.camera.get_mat(s["parent"])
            base.camera.reparent_to(s["parent"])
            base.camera.set_mat(mat)
            base.camera.set_r(0)
        fly = getattr(base, "fly_cam", None)
        if fly is not None:
            fly.set_frozen(bool(s.get("fly_frozen", False)))
        self._saved = None

    # ------------------------------------------------------------------ #
    def photo_lens(self, f: float, cx: float, cy: float, width: int, height: int) -> None:
        """
        Объектив станции (пиксельные f, cx, cy снимка width x height). Камера
        сцены — обскура с тем же центром кадра; её фокус f (см. photo_shot)
        может быть короче, чтобы в кадр вошёл весь «выпрямленный» снимок.
        """
        self.photo = dict(f=f, cx=cx, cy=cy, w=width, h=height)

    def photo_shot(self, R_wc: np.ndarray, centre: np.ndarray, f_view=None) -> Shot:
        """
        Поза камеры станции: R_wc — столбцы = оси камеры OpenCV (x вправо,
        y вниз, z вперёд) в мировой системе, centre — её центр. f_view —
        фокус камеры сцены в пикселях снимка (по умолчанию как у станции).
        """
        fwd = R_wc[:, 2]
        up = -R_wc[:, 1]
        p = self.photo or dict(f=1000, h=1000)
        fov = math.degrees(2 * math.atan(p["h"] / 2 / (f_view or p["f"])))
        off = (0.0, 0.0)
        if self.photo:
            off = ((p["cx"] - (p["w"] - 1) / 2) / p["h"], -(p["cy"] - (p["h"] - 1) / 2) / p["h"])
        return Shot(centre, centre + fwd * 10.0, up, fov, off)

    # ------------------------------------------------------------------ #
    def sample_path(self, shots: Sequence[Shot], k: float) -> Shot:
        """Сглаженный путь через позы (Catmull-Rom по позиции и цели)."""
        n = len(shots)
        if n == 1:
            return shots[0].copy()
        if n == 2:
            return Shot.mix(shots[0], shots[1], k)
        seg = min(n - 2, int(k * (n - 1)))
        t = k * (n - 1) - seg
        p = [shots[max(0, min(n - 1, seg + i))] for i in (-1, 0, 1, 2)]
        pos = _catmull(*(s.pos for s in p), t)
        tgt = _catmull(*(s.target for s in p), t)
        base = Shot.mix(p[1], p[2], t)
        return Shot(pos, tgt, base.up, base.fov, base.offset)

    def move(self, shots, duration: float, curve=ease.in_out_cubic) -> Cmd:
        """Команда: проехать через позу/позы за duration секунд."""
        if isinstance(shots, Shot):
            shots = [shots]
        return _MoveCmd(self, shots, duration, curve)

    def orbit(self, center, radius, height, a0, a1, duration, curve=ease.in_out_sine,
              fov=None, target_offset=(0, 0, 0)) -> Cmd:
        """Команда: облёт вокруг center по дуге a0 -> a1 (градусы)."""
        return _OrbitCmd(self, center, radius, height, a0, a1, duration, curve, fov,
                         target_offset)

    def cut(self, shot: Shot) -> None:
        self.shot = shot.copy()

    # ------------------------------------------------------------------ #
    def _apply(self, task):
        from direct.task.Task import Task
        from panda3d.core import ClockObject
        self._time += ClockObject.get_global_clock().get_dt()
        s = self.shot
        base = self.base
        ref = self.space or base.render
        pos = Vec3(s.pos)
        tgt = Vec3(s.target)
        if self.shake > 0:
            t = self._time
            n = Vec3(math.sin(t * 0.71) + 0.5 * math.sin(t * 1.93 + 1.3),
                     math.sin(t * 0.53 + 2.1) + 0.5 * math.sin(t * 1.37),
                     math.sin(t * 0.87 + 0.7) * 0.6)
            pos += n * self.shake
            tgt += n * self.shake * 0.3
        cam = base.camera
        cam.reparent_to(ref)
        cam.set_pos(pos)
        cam.look_at(Point3(tgt), s.up)
        lens = base.camLens
        # вертикальный угол; горизонтальный — по соотношению сторон окна
        win = base.win
        aspect = win.get_x_size() / max(1, win.get_y_size())
        fov_v = max(1.0, min(170.0, s.fov))
        fov_h = math.degrees(2 * math.atan(math.tan(math.radians(fov_v) / 2) * aspect))
        lens.set_fov(LVecBase2f(fov_h, fov_v))
        lens.set_film_offset(LVecBase2f(s.offset.x * lens.get_film_size()[1],
                                        s.offset.y * lens.get_film_size()[1]))
        return Task.cont
