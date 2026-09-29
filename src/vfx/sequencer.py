# -*- coding: utf-8 -*-
"""
Сценарий на корутинах.

Режиссура пишется обычным генератором, который отдаёт команды::

    def intro(seq):
        yield wait(0.5)
        yield tween(2.0, lambda v: node.set_alpha_scale(v), 0, 1, ease.out_cubic)
        yield until(lambda: data.ready)          # ждать данных без таймлайна
        a = seq.spawn(orbit())                   # параллельная дорожка
        yield tween(1.0, ...)
        yield a                                  # дождаться дорожки
        yield sub_scene()                        # вложенный генератор

`yield None` — просто следующий кадр. Команда, завершившаяся посреди кадра,
сразу отдаёт управление следующей: цепочка мгновенных команд не теряет
кадров.

Время идёт по реальным кадрам, но шаг ограничен (`max_dt`): просевший кадр
не проглатывает кусок анимации. Для записи роликов и проверок есть
`fixed_dt` — каждый кадр продвигает сценарий ровно на этот шаг.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Iterable, List, Optional

from . import easing as ease


class Cmd:
    """Команда сценария. `step` возвращает True, когда команда завершена."""

    def start(self, seq: "Sequencer") -> None:
        pass

    def step(self, dt: float) -> bool:
        return True


class _Wait(Cmd):
    def __init__(self, seconds: float):
        self.left = float(seconds)

    def step(self, dt):
        self.left -= dt
        return self.left <= 0


class _Until(Cmd):
    def __init__(self, predicate: Callable[[], bool], timeout: Optional[float]):
        self.predicate = predicate
        self.left = timeout

    def step(self, dt):
        if self.predicate():
            return True
        if self.left is not None:
            self.left -= dt
            return self.left <= 0
        return False


def _mix(a, b, t):
    """Число, кортеж/список или вектор Panda — поэлементно."""
    if isinstance(a, (int, float)):
        return a + (b - a) * t
    vals = [x + (y - x) * t for x, y in zip(a, b)]
    if isinstance(a, (list, tuple)):
        return vals
    try:
        return type(a)(*vals)
    except Exception:
        return vals


class _Tween(Cmd):
    def __init__(self, duration, fn, a=0.0, b=1.0, curve=ease.in_out_cubic):
        self.duration = max(1e-6, float(duration))
        self.fn, self.a, self.b, self.curve = fn, a, b, curve
        self.t = 0.0

    def start(self, seq):
        self.fn(_mix(self.a, self.b, self.curve(0.0)))

    def step(self, dt):
        self.t += dt
        k = min(1.0, self.t / self.duration)
        self.fn(_mix(self.a, self.b, self.curve(k)))
        return k >= 1.0


class _Call(Cmd):
    def __init__(self, fn):
        self.fn = fn

    def start(self, seq):
        self.fn()


def wait(seconds: float) -> Cmd:
    return _Wait(seconds)


def until(predicate: Callable[[], bool], timeout: Optional[float] = None) -> Cmd:
    return _Until(predicate, timeout)


def tween(duration, fn, a=0.0, b=1.0, curve=ease.in_out_cubic) -> Cmd:
    """Каждый кадр зовёт fn(значение) — от a до b по кривой curve."""
    return _Tween(duration, fn, a, b, curve)


def call(fn) -> Cmd:
    return _Call(fn)


# --------------------------------------------------------------------------- #

class Track(Cmd):
    """Исполняемая корутина. Сама — команда: `yield track` ждёт её конца."""

    def __init__(self, gen, name: str = ""):
        self.gen = gen
        self.name = name or getattr(gen, "__name__", "track")
        self.cur: Optional[Cmd] = None
        self.done = False
        self.error: Optional[BaseException] = None
        self.seq: Optional[Sequencer] = None

    def cancel(self) -> None:
        if not self.done:
            self.done = True
            try:
                self.gen.close()
            except Exception:
                pass

    # как команда внутри другой дорожки — просто ждём завершения
    def step(self, dt):
        return self.done

    def _advance(self, dt: float) -> None:
        budget = 64                      # защита от бесконечных мгновенных цепочек
        while not self.done and budget > 0:
            budget -= 1
            if self.cur is None:
                try:
                    item = next(self.gen)
                except StopIteration:
                    self.done = True
                    return
                except Exception as exc:
                    self.error = exc
                    self.done = True
                    print(f"[vfx] дорожка «{self.name}» упала:")
                    traceback.print_exc()
                    return
                if item is None:
                    return                   # просто следующий кадр
                if hasattr(item, "__next__"):
                    item = self.seq.spawn(item)
                elif isinstance(item, (list, tuple)):
                    item = _All([self.seq.spawn(g) if hasattr(g, "__next__") else g
                                 for g in item])
                self.cur = item
                if not isinstance(item, Track):
                    try:
                        item.start(self.seq)
                    except Exception as exc:
                        self.error = exc
                        print(f"[vfx] команда в дорожке «{self.name}» не стартовала:")
                        traceback.print_exc()
                        self.cancel()
                        return
                dt_here = dt
                dt = 0.0                     # время кадра тратится один раз
                if self._step_cur(dt_here):
                    self.cur = None
                    continue
                return
            if self._step_cur(dt):
                self.cur = None
                dt = 0.0
                continue
            return

    def _step_cur(self, dt: float) -> bool:
        """Шаг текущей команды; упавшая команда гасит только эту дорожку."""
        try:
            return self.cur.step(dt)
        except Exception as exc:
            self.error = exc
            print(f"[vfx] команда в дорожке «{self.name}» упала:")
            traceback.print_exc()
            self.cancel()
            return True


class _All(Cmd):
    def __init__(self, items: List[Cmd]):
        self.items = items

    def start(self, seq):
        for it in self.items:
            if not isinstance(it, Track):
                it.start(seq)

    def step(self, dt):
        done = True
        for it in self.items:
            if isinstance(it, Track):
                done &= it.done
            elif not getattr(it, "_finished", False):
                it._finished = it.step(dt)
                done &= it._finished
        return done


class Sequencer:
    """Крутит дорожки каждый кадр (задача Panda)."""

    def __init__(self, base, max_dt: float = 1 / 15, task_sort: int = -40):
        self.base = base
        self.max_dt = max_dt
        self.fixed_dt: Optional[float] = None
        self.speed = 1.0
        self.time = 0.0
        self.tracks: List[Track] = []
        self._on_frame: List[Callable[[float, float], None]] = []
        self._task = base.taskMgr.add(self._update, "vfx_sequencer", sort=task_sort)

    def spawn(self, gen, name: str = "") -> Track:
        """Запустить дорожку: генератор или одиночную команду."""
        if not hasattr(gen, "__next__"):
            cmd = gen

            def _single():
                yield cmd
            gen = _single()
        tr = Track(gen, name)
        tr.seq = self
        self.tracks.append(tr)
        tr._advance(0.0)
        return tr

    run = spawn

    def every_frame(self, fn: Callable[[float, float], None]) -> None:
        """fn(time, dt) каждый кадр — для униформ, зависящих от времени."""
        self._on_frame.append(fn)

    def cancel_all(self) -> None:
        for tr in self.tracks:
            tr.cancel()
        self.tracks.clear()

    def destroy(self) -> None:
        self.cancel_all()
        self._on_frame.clear()
        if self._task is not None:
            self.base.taskMgr.remove(self._task)
            self._task = None

    @property
    def busy(self) -> bool:
        return any(not t.done for t in self.tracks)

    def _update(self, task):
        from direct.task.Task import Task
        if self.fixed_dt is not None:
            dt = self.fixed_dt
        else:
            from panda3d.core import ClockObject
            dt = min(self.max_dt, ClockObject.get_global_clock().get_dt())
        dt *= self.speed
        self.time += dt
        for fn in list(self._on_frame):
            try:
                fn(self.time, dt)
            except Exception:
                traceback.print_exc()
        for tr in list(self.tracks):
            if not tr.done:
                tr._advance(dt)
        self.tracks = [t for t in self.tracks if not t.done]
        return Task.cont
