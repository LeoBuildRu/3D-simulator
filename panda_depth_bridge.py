# panda_depth_bridge.py
# ---------------------------------------------------------------------------
# Bridge: Panda3D depth buffer  ->  PyQt6 DepthMapPreview widget.
#
# Strategy:
#   1. Create an offscreen buffer with a depth texture attached.
#   2. Enable set_keep_ram_image(True) so the depth texture is mirrored into
#      CPU-accessible memory after each render pass.
#   3. Every frame, add a task to taskMgr that grabs Texture.getRamImage()
#      and hands the bytes off to the preview widget via a Qt signal.
#
# Notes:
#   * We keep all Qt interactions on the GUI thread by emitting a signal from
#     the task (the task itself runs inside Panda's loop, which in our setup
#     is stepped from the Qt main loop via QTimer — so we're already on the
#     GUI thread anyway).  The signal/slot pattern gives us thread-safety for
#     free if that ever changes.
#   * The depth texture is F_depth_component; we convert to 8-bit grayscale
#     (or to a false-color heat map if you want richer visualisation — see
#     `_to_heatmap` below).
# ---------------------------------------------------------------------------

from __future__ import annotations

import struct

import numpy as np
from panda3d.core import (
    Texture, GraphicsOutput, FrameBufferProperties, WindowProperties,
    GraphicsPipe, NodePath, OrthographicLens, Camera,
)
from direct.task import Task


class DepthBridge:
    """
    Pipes the Panda3D depth pass into a PyQt6 DepthMapPreview widget.

    Parameters
    ----------
    panda_app : ShowBase
        Your running Panda3D application instance.
    preview   : DepthMapPreview
        The Qt widget that will display frames.
    width, height : int
        Resolution of the depth capture.  Keep it modest (e.g. 512×288
        for a 16:9 preview) — scaling up buys nothing on the GUI side.
    stride    : int
        Only push every N-th frame to keep UI load low.  2 or 3 is fine.
    colormap  : str
        "gray" — raw grayscale (fastest).
        "turbo" — false-color heat map (nicer visualisation).
    """

    def __init__(self, panda_app, preview, *,
                 width: int = 512, height: int = 288,
                 stride: int = 2, colormap: str = "turbo"):
        self.app = panda_app
        self.preview = preview
        self.w, self.h = width, height
        self.stride = max(1, stride)
        self.colormap = colormap
        self._counter = 0

        self.depth_tex: Texture | None = None
        self.buffer = None
        self.cam: NodePath | None = None

        self._setup_offscreen_pass()

        # Task runs every frame after Panda's main render pass.
        self.app.taskMgr.add(self._tick, "DepthBridgeTick",
                             sort=90, priority=None)

    # ------------------------------------------------------------------
    # Offscreen depth capture
    # ------------------------------------------------------------------
    def _setup_offscreen_pass(self) -> None:
        fbp = FrameBufferProperties()
        fbp.set_depth_bits(32)
        fbp.set_rgb_color(False)

        win_props = WindowProperties.size(self.w, self.h)

        self.buffer = self.app.graphicsEngine.make_output(
            self.app.pipe, "depth_buffer", -100,
            fbp, win_props,
            GraphicsPipe.BFRefuseWindow,
            self.app.win.getGsg(), self.app.win,
        )
        if self.buffer is None:
            raise RuntimeError("Could not create offscreen depth buffer")

        # Attach a CPU-mirrored depth texture.
        self.depth_tex = Texture("depth_tex")
        self.depth_tex.set_format(Texture.F_depth_component)
        self.depth_tex.set_component_type(Texture.T_float)
        self.depth_tex.set_keep_ram_image(True)      # <-- the important bit

        self.buffer.add_render_texture(
            self.depth_tex,
            GraphicsOutput.RTMCopyRam,               # copy pixels back to RAM
            GraphicsOutput.RTPDepth,
        )

        # A second camera that shares the main scene / main camera's lens.
        cam_node = Camera("depth_cam")
        cam_node.set_lens(self.app.camLens)
        self.cam = self.app.render.attach_new_node(cam_node)
        self.cam.reparent_to(self.app.camera)        # follow the main camera

        dr = self.buffer.make_display_region()
        dr.set_camera(self.cam)
        dr.set_clear_depth_active(True)

    # ------------------------------------------------------------------
    # Per-frame task
    # ------------------------------------------------------------------
    def _tick(self, task):
        self._counter += 1
        if self._counter % self.stride != 0:
            return Task.cont

        tex = self.depth_tex
        if tex is None or not tex.has_ram_image():
            return Task.cont

        ram = tex.get_ram_image_as("D")              # float32 depth
        if ram is None:
            return Task.cont
        try:
            buf = memoryview(ram).tobytes()
        except Exception:
            return Task.cont
        if not buf:
            return Task.cont
        depth = np.frombuffer(buf, dtype=np.float32)
        if depth.size != self.w * self.h:
            return Task.cont

        depth = depth.reshape(self.h, self.w)

        # Normalise non-linear [0..1] depth to visible range.
        near, far = np.percentile(depth, (2, 98))
        if far - near < 1e-6:
            return Task.cont
        norm = np.clip((depth - near) / (far - near), 0.0, 1.0)

        if self.colormap == "gray":
            frame = (norm * 255.0).astype(np.uint8)
            self.preview.push_frame(frame.tobytes(), self.w, self.h, "gray8")
        else:
            rgba = self._to_heatmap(norm)            # (H, W, 4) uint8
            self.preview.push_frame(rgba.tobytes(), self.w, self.h, "rgba8")

        return Task.cont

    # ------------------------------------------------------------------
    # Simple Turbo-style LUT (no matplotlib dependency)
    # ------------------------------------------------------------------
    _LUT = None

    @classmethod
    def _build_lut(cls):
        # 5-stop gradient: black -> cyan -> mint -> lime -> white
        stops = np.array([
            [  6,   8,  12, 255],
            [  0, 140, 160, 255],
            [  0, 255, 136, 255],          # accent mint
            [210, 255, 120, 255],
            [255, 255, 255, 255],
        ], dtype=np.float32)
        xs = np.linspace(0.0, 1.0, len(stops))
        grid = np.linspace(0.0, 1.0, 256)
        lut = np.empty((256, 4), dtype=np.uint8)
        for c in range(4):
            lut[:, c] = np.interp(grid, xs, stops[:, c]).astype(np.uint8)
        cls._LUT = lut

    @classmethod
    def _to_heatmap(cls, norm: np.ndarray) -> np.ndarray:
        if cls._LUT is None:
            cls._build_lut()
        idx = (norm * 255.0).astype(np.uint8)
        return cls._LUT[idx]

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if self.buffer is not None:
            self.app.graphicsEngine.remove_window(self.buffer)
            self.buffer = None
        self.app.taskMgr.remove("DepthBridgeTick")
