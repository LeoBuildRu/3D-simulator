# camera_controller.py
# ---------------------------------------------------------------------------
# Editor-style fly camera for the embedded Panda3D viewport.
#
# Bindings:
#     W / A / S / D     — strafe along the camera's local forward / right axes
#     Q / E             — fall / rise along world +Z
#     Shift (any side)  — sprint (4× movement speed)
#     RMB hold + mouse  — look (yaw / pitch); cursor is hidden while held,
#                         pitch is clamped to ±89°, roll is forced to 0
#
# Implementation notes:
#   * Input goes through Panda3D's own messenger (`app.accept`). The embedded
#     Panda HWND receives WM_KEY/WM_MOUSE messages whenever it has focus —
#     usually the moment the user clicks anywhere on the viewport.
#   * Mouse-look uses raw pixel deltas via `win.getPointer(0)` (NOT the
#     normalized `mouseWatcherNode.getMouseX/Y`) so that look sensitivity is
#     in proper degrees-per-pixel and independent of viewport size / aspect.
#   * `app.disable_mouse()` MUST have been called by the host (main.py does
#     this) so Panda's default trackball doesn't fight us.
# ---------------------------------------------------------------------------

from __future__ import annotations

import math

from panda3d.core import Vec3, WindowProperties, ClockObject


class FlyCamera:
    """Per-frame fly camera driven from Panda3D's input system."""

    # ---- Tunables ----------------------------------------------------
    MOVE_SPEED   = 10.0   # world units / sec
    SPRINT_MULT  = 4.0
    LOOK_SENS    = 0.18   # degrees per pixel of mouse delta
    PITCH_LIMIT  = 89.0   # ± degrees

    # Keys we care about — both bare names and `-up` events get bound.
    _KEYS = ("w", "a", "s", "d", "q", "e",
             "shift", "lshift", "rshift")

    # ------------------------------------------------------------------
    def __init__(
        self,
        app,
        start_pos: Vec3 = Vec3(0.0, -120.0, 25.0),
        start_hpr: Vec3 = Vec3(0.0, -12.0, 0.0),
    ):
        self.app = app
        self.cam = app.camera

        # Place the camera somewhere where the ground plane is actually
        # visible (it's a 2000×2000 quad at z≈0).
        self.cam.set_pos(start_pos)
        self.cam.set_hpr(start_hpr)

        self._keys: set[str] = set()
        self._looking = False
        self._last_mx: int | None = None
        self._last_my: int | None = None

        # ---- Bind keys ----------------------------------------------
        for k in self._KEYS:
            app.accept(k,         self._on_key, [k, True])
            app.accept(f"{k}-up", self._on_key, [k, False])

        # ---- Right mouse button toggles look mode -------------------
        app.accept("mouse3",    self._begin_look)
        app.accept("mouse3-up", self._end_look)

        # ---- Per-frame update ---------------------------------------
        self._task = app.taskMgr.add(self._update, "fly_cam_update")

    # ==================================================================
    # Input handlers
    # ==================================================================
    def _on_key(self, key: str, down: bool) -> None:
        if down:
            self._keys.add(key)
        else:
            self._keys.discard(key)

    def _shift_held(self) -> bool:
        return bool(self._keys & {"shift", "lshift", "rshift"})

    # ------------------------------------------------------------------
    def _begin_look(self) -> None:
        self._looking = True
        self._last_mx = None
        self._last_my = None

        props = WindowProperties()
        props.setCursorHidden(True)
        try:
            self.app.win.requestProperties(props)
        except Exception:
            pass

    def _end_look(self) -> None:
        self._looking = False

        props = WindowProperties()
        props.setCursorHidden(False)
        try:
            self.app.win.requestProperties(props)
        except Exception:
            pass

    # ==================================================================
    # Freeze / unfreeze (for stationary / on-board camera modes)
    # ==================================================================
    def set_frozen(self, frozen: bool) -> None:
        """
        When frozen, the per-frame _update is a no-op so external code
        (camera-mode buttons in the telemetry widget) can pin the
        camera at a fixed pose without the fly-cam fighting back.
        Also clears any held keys + look state so unfreezing doesn't
        snap the camera.
        """
        self._frozen = bool(frozen)
        if frozen:
            self._keys.clear()
            self._looking = False
            try:
                from panda3d.core import WindowProperties as _WP
                props = _WP()
                props.setCursorHidden(False)
                if self.app.win is not None:
                    self.app.win.requestProperties(props)
            except Exception:
                pass

    def is_frozen(self) -> bool:
        return bool(getattr(self, "_frozen", False))

    # ==================================================================
    # Frame tick
    # ==================================================================
    def _update(self, task):
        if getattr(self, "_frozen", False):
            return task.cont
        dt = ClockObject.getGlobalClock().getDt()

        # ---- Look ---------------------------------------------------
        if self._looking and self.app.win is not None:
            try:
                ptr = self.app.win.getPointer(0)
                if ptr.getInWindow():
                    mx, my = ptr.getX(), ptr.getY()
                    if self._last_mx is not None:
                        dx = mx - self._last_mx
                        dy = my - self._last_my

                        self.cam.set_h(self.cam.get_h() - dx * self.LOOK_SENS)

                        new_p = self.cam.get_p() - dy * self.LOOK_SENS
                        if new_p >  self.PITCH_LIMIT: new_p =  self.PITCH_LIMIT
                        if new_p < -self.PITCH_LIMIT: new_p = -self.PITCH_LIMIT
                        self.cam.set_p(new_p)

                        # Fly cams shouldn't tilt.
                        self.cam.set_r(0.0)

                    self._last_mx = mx
                    self._last_my = my
                else:
                    # Pointer left the window — reset deltas so we don't
                    # snap when it re-enters.
                    self._last_mx = None
                    self._last_my = None
            except Exception:
                # Some embedded configurations don't expose getPointer —
                # never let that crash the frame loop.
                pass

        # ---- Move ---------------------------------------------------
        if self._keys:
            heading_rad = math.radians(self.cam.get_h())
            pitch_rad   = math.radians(self.cam.get_p())

            # Local forward (W) — combines yaw and pitch so flying forward
            # while looking down dives toward the ground (feels right).
            forward = Vec3(
                -math.sin(heading_rad) * math.cos(pitch_rad),
                 math.cos(heading_rad) * math.cos(pitch_rad),
                 math.sin(pitch_rad),
            )
            # Strafe is purely horizontal so A/D doesn't drift you up/down.
            right = Vec3(
                 math.cos(heading_rad),
                 math.sin(heading_rad),
                 0.0,
            )
            world_up = Vec3(0.0, 0.0, 1.0)

            delta = Vec3(0.0, 0.0, 0.0)
            if "w" in self._keys: delta += forward
            if "s" in self._keys: delta -= forward
            if "d" in self._keys: delta += right
            if "a" in self._keys: delta -= right
            if "e" in self._keys: delta += world_up
            if "q" in self._keys: delta -= world_up

            if delta.length_squared() > 0.0:
                delta.normalize()
                speed = self.MOVE_SPEED * (
                    self.SPRINT_MULT if self._shift_held() else 1.0
                )
                self.cam.set_pos(self.cam.get_pos() + delta * speed * dt)

        return task.cont
