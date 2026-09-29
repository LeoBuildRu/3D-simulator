# -*- coding: utf-8 -*-
"""Кривые сглаживания: t в [0, 1] -> [0, 1] (у back/elastic — с выходом за)."""

from __future__ import annotations

import math


def linear(t: float) -> float:
    return t


def in_quad(t):
    return t * t


def out_quad(t):
    return 1 - (1 - t) * (1 - t)


def in_out_quad(t):
    return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2


def in_cubic(t):
    return t ** 3


def out_cubic(t):
    return 1 - (1 - t) ** 3


def in_out_cubic(t):
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def in_out_quint(t):
    return 16 * t ** 5 if t < 0.5 else 1 - (-2 * t + 2) ** 5 / 2


def out_quint(t):
    return 1 - (1 - t) ** 5


def in_expo(t):
    return 0.0 if t <= 0 else 2 ** (10 * t - 10)


def out_expo(t):
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def in_out_expo(t):
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    return 2 ** (20 * t - 10) / 2 if t < 0.5 else (2 - 2 ** (-20 * t + 10)) / 2


def in_out_sine(t):
    return -(math.cos(math.pi * t) - 1) / 2


def out_back(t, s=1.70158):
    return 1 + (s + 1) * (t - 1) ** 3 + s * (t - 1) ** 2


def out_elastic(t):
    if t <= 0 or t >= 1:
        return float(t >= 1)
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi / 3)) + 1


def smoothstep(a: float, b: float, x: float) -> float:
    if b == a:
        return float(x >= b)
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t
