from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import warp as wp
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task.Task import Task
from panda3d.core import (
    Geom,
    GeomEnums,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    Material,
    NodePath,
    OmniBoundingVolume,
    Texture,
    TransparencyAttrib,
    Vec3,
)

wp.init()

cvec = wp.vec3
cnum = wp.float32
TWO_PI = np.float32(math.tau)


@wp.func
def _rand01(index: int, salt: int, generation: int) -> cnum:
    #value = cnum(index) * 12.9898 + cnum(salt) * 78.233 + cnum(generation) * 37.719
    #return wp.frac(wp.sin(value) * 43758.5453)
    return wp.randf(wp.uint32(index + salt * 1234567))


@wp.func
def _sample_range(min_value: cnum, max_value: cnum, amount: cnum) -> cnum:
    return min_value + (max_value - min_value) * amount


@wp.func
def _random_axis(index: int, generation: int) -> cvec:
    axis = cvec(
        _sample_range(-1.0, 1.0, _rand01(index, 11, generation)),
        _sample_range(-1.0, 1.0, _rand01(index, 12, generation)),
        _sample_range(-1.0, 1.0, _rand01(index, 13, generation)),
    )

    axis_length = wp.length(axis)
    if axis_length < 1.0e-6:
        return cvec(0.0, 0.0, 1.0)

    return axis / axis_length


@wp.func
def _wrap_angle(angle: cnum) -> cnum:
    return angle - 6.283185307179586 * wp.floor(angle * 0.15915494309189535)


@wp.kernel
def _initialize_particles(
    positions: wp.array(dtype=cvec),
    speeds: wp.array(dtype=cnum),
    sizes: wp.array(dtype=cnum),
    spin_angles: wp.array(dtype=cnum),
    spin_speeds: wp.array(dtype=cnum),
    spin_axes: wp.array(dtype=cvec),
    respawn_counts: wp.array(dtype=wp.int32),
    spawn_min: cvec,
    spawn_max: cvec,
    speed_min: cnum,
    speed_max: cnum,
    size_min: cnum,
    size_max: cnum,
    spin_speed_min: cnum,
    spin_speed_max: cnum,
    random_rotation: int,
):
    i = wp.tid()
    generation = 0

    respawn_counts[i] = generation
    positions[i] = cvec(
        _sample_range(spawn_min[0], spawn_max[0], _rand01(i, 1, generation)),
        _sample_range(spawn_min[1], spawn_max[1], _rand01(i, 2, generation)),
        _sample_range(spawn_min[2], spawn_max[2], _rand01(i, 3, generation)),
    )
    speeds[i] = _sample_range(speed_min, speed_max, _rand01(i, 4, generation))
    sizes[i] = _sample_range(size_min, size_max, _rand01(i, 5, generation))
    spin_angles[i] = _sample_range(0.0, 6.283185307179586, _rand01(i, 6, generation))
    spin_axes[i] = _random_axis(i, generation)

    if random_rotation != 0:
        spin_speeds[i] = _sample_range(
            spin_speed_min,
            spin_speed_max,
            _rand01(i, 7, generation),
        )
    else:
        spin_speeds[i] = 0.0


@wp.kernel
def _update_particles(
    positions: wp.array(dtype=cvec),
    speeds: wp.array(dtype=cnum),
    sizes: wp.array(dtype=cnum),
    spin_angles: wp.array(dtype=cnum),
    spin_speeds: wp.array(dtype=cnum),
    spin_axes: wp.array(dtype=cvec),
    respawn_counts: wp.array(dtype=wp.int32),
    spawn_min: cvec,
    spawn_max: cvec,
    respawn_z: cnum,
    respawn_top_jitter: cnum,
    threshold_z: cnum,
    drift_velocity: cvec,
    dt: cnum,
    speed_min: cnum,
    speed_max: cnum,
    size_min: cnum,
    size_max: cnum,
    spin_speed_min: cnum,
    spin_speed_max: cnum,
    random_rotation: int,
):
    i = wp.tid()

    pos = positions[i]
    pos = pos + drift_velocity * dt
    pos = cvec(pos[0], pos[1], pos[2] - speeds[i] * dt)

    if random_rotation != 0:
        spin_angles[i] = _wrap_angle(spin_angles[i] + spin_speeds[i] * dt)
    else:
        spin_angles[i] = 0.0

    if pos[2] < threshold_z:
        generation = respawn_counts[i] + 1
        respawn_counts[i] = generation

        positions[i] = cvec(
            _sample_range(spawn_min[0], spawn_max[0], _rand01(i, 1, generation)),
            _sample_range(spawn_min[1], spawn_max[1], _rand01(i, 2, generation)),
            respawn_z + respawn_top_jitter * _rand01(i, 3, generation),
        )
        speeds[i] = _sample_range(speed_min, speed_max, _rand01(i, 4, generation))
        sizes[i] = _sample_range(size_min, size_max, _rand01(i, 5, generation))
        spin_angles[i] = _sample_range(0.0, 6.283185307179586, _rand01(i, 6, generation))
        spin_axes[i] = _random_axis(i, generation)

        if random_rotation != 0:
            spin_speeds[i] = _sample_range(
                spin_speed_min,
                spin_speed_max,
                _rand01(i, 7, generation),
            )
        else:
            spin_speeds[i] = 0.0
    else:
        positions[i] = pos


class WarpFallingParticles:
    BILLBOARD = "billboard"
    RANDOM_ROTATION = "random"

    def __init__(
        self,
        showbase: ShowBase,
        render_pipeline: Any,
        texture: Union[str, Texture],
        particle_count: int,
        spawn_min: Tuple[float, float, float],
        spawn_max: Tuple[float, float, float],
        respawn_threshold: float,
        rotation_mode: str = BILLBOARD,
        size_range: Tuple[float, float] = (0.15, 0.5),
        speed_range: Tuple[float, float] = (1.5, 4.0),
        spin_speed_range: Tuple[float, float] = (-3.0, 3.0),
        respawn_top_jitter: float = 0.0,
        drift_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        parent: Optional[NodePath] = None,
        camera_np: Optional[NodePath] = None,
        effect_path: Optional[str] = None,
        effect_options: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        task_name: Optional[str] = None,
        max_dt: float = 1.0 / 30.0,
        alpha_blend: bool = True,
        auto_start: bool = True,
    ) -> None:
        self.showbase = showbase
        self.render_pipeline = render_pipeline
        self.parent = parent or self.showbase.render
        self.camera_np = camera_np or self.showbase.camera
        self.rotation_mode = rotation_mode
        self.random_rotation = 1 if rotation_mode == self.RANDOM_ROTATION else 0
        self.effect_path = effect_path
        self.effect_options = dict(effect_options or {})
        self.device = self._resolve_device(device)
        self.device_info = wp.get_device(self.device)
        self.is_cuda = self.device_info.is_cuda
        self.max_dt = float(max_dt)
        self.task_name = task_name or f"warp-falling-particles-{id(self)}"
        self.alpha_blend = alpha_blend
        self.running = False
        self.destroyed = False

        self.spawn_min = self._validate_vec3("spawn_min", spawn_min)
        self.spawn_max = self._validate_vec3("spawn_max", spawn_max)
        self.drift_velocity = self._validate_vec3("drift_velocity", drift_velocity)
        self._validate_spawn_bounds()

        self.count = int(particle_count)
        if self.count <= 0:
            raise ValueError("particle_count must be greater than zero")

        self.respawn_threshold = float(respawn_threshold)
        self.respawn_z = float(self.spawn_max[2])
        self.respawn_top_jitter = float(respawn_top_jitter)

        self.size_min, self.size_max = self._validate_range("size_range", size_range, positive=True)
        self.speed_min, self.speed_max = self._validate_range("speed_range", speed_range, positive=True)
        self.spin_speed_min, self.spin_speed_max = self._validate_range("spin_speed_range", spin_speed_range)

        if rotation_mode not in (self.BILLBOARD, self.RANDOM_ROTATION):
            raise ValueError(
                f"rotation_mode must be '{self.BILLBOARD}' or '{self.RANDOM_ROTATION}'"
            )

        self.buffer_texture = Texture()
        self.buffer_texture.setup_buffer_texture(
            self.count * 4,
            Texture.T_float,
            Texture.F_rgba32,
            GeomEnums.UH_dynamic,
        )
        self.matrices = np.zeros((self.count, 4, 4), dtype=np.float32)
        self.matrices[:, 3, 3] = 1.0

        self._allocate_warp_arrays()
        self._create_quad(texture)
        self.reset()

        if auto_start:
            self.start()

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        """Pick a usable Warp device, falling back to CPU when CUDA is absent.

        A specific device string is honoured if Warp reports it as available;
        otherwise (and when device is None) we prefer the first CUDA device and
        fall back to "cpu" so the simulation also runs on machines without a GPU.
        """
        if device is not None:
            try:
                if wp.is_device_available(wp.get_device(device)):
                    return device
            except Exception:
                pass

        if wp.is_cuda_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _validate_vec3(name: str, value: Tuple[float, float, float]) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (3,):
            raise ValueError(f"{name} must be a 3-item vector")
        return vector

    @staticmethod
    def _validate_range(
        name: str,
        value: Tuple[float, float],
        *,
        positive: bool = False,
    ) -> Tuple[float, float]:
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two numbers")

        min_value = float(value[0])
        max_value = float(value[1])
        if min_value > max_value:
            raise ValueError(f"{name} minimum must be <= maximum")
        if positive and min_value <= 0.0:
            raise ValueError(f"{name} must be strictly positive")

        return min_value, max_value

    def _validate_spawn_bounds(self) -> None:
        if np.any(self.spawn_min > self.spawn_max):
            raise ValueError("spawn_min must be <= spawn_max on every axis")

    def _allocate_warp_arrays(self) -> None:
        self.positions = wp.empty(self.count, dtype=cvec, device=self.device)
        self.speeds = wp.empty(self.count, dtype=cnum, device=self.device)
        self.sizes = wp.empty(self.count, dtype=cnum, device=self.device)
        self.spin_angles = wp.empty(self.count, dtype=cnum, device=self.device)
        self.spin_speeds = wp.empty(self.count, dtype=cnum, device=self.device)
        self.spin_axes = wp.empty(self.count, dtype=cvec, device=self.device)
        self.respawn_counts = wp.zeros(self.count, dtype=wp.int32, device=self.device)

        # Pinned host memory only makes sense (and is only supported) for
        # CUDA<->host transfers. On a CPU device the staging arrays alias the
        # compute arrays, so pinning is both unnecessary and unsupported.
        pinned = self.is_cuda
        self.positions_cpu = wp.zeros(self.count, dtype=cvec, device="cpu", pinned=pinned)
        self.sizes_cpu = wp.zeros(self.count, dtype=cnum, device="cpu", pinned=pinned)
        self.spin_angles_cpu = wp.zeros(self.count, dtype=cnum, device="cpu", pinned=pinned)
        self.spin_axes_cpu = wp.zeros(self.count, dtype=cvec, device="cpu", pinned=pinned)

    def _default_effect_path(self) -> str:
        project_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ))
        config_dir = os.path.join(project_root, "config")
        if self.alpha_blend:
            return os.path.join(config_dir, "rp_instancing_transparent.yaml")
        return os.path.join(config_dir, "rp_instancing_cutout.yaml")

    def _default_effect_options(self) -> Dict[str, bool]:
        options: Dict[str, bool] = {
            "render_shadow": False,
            "render_voxelize": False,
            "render_envmap": False,
            "normal_mapping": False,
            "parallax_mapping": False,
        }

        if self.alpha_blend:
            options.update(
                {
                    "render_gbuffer": False,
                    "render_forward": True,
                    "alpha_testing": False,
                }
            )
        else:
            options.update(
                {
                    "render_gbuffer": True,
                    "render_forward": False,
                    "alpha_testing": True,
                }
            )

        return options

    def _forward_shading_enabled(self) -> bool:
        if self.render_pipeline is None or not hasattr(self.render_pipeline, "plugin_mgr"):
            return False
        plugin_mgr = self.render_pipeline.plugin_mgr
        if plugin_mgr is None or not hasattr(plugin_mgr, "is_plugin_enabled"):
            return False
        return bool(plugin_mgr.is_plugin_enabled("forward_shading"))

    def _create_quad(self, texture: Union[str, Texture]) -> None:
        vertex_data = GeomVertexData(
            "warp_particle_quad",
            GeomVertexFormat.get_v3n3t2(),
            GeomEnums.UH_static,
        )
        vertex_writer = GeomVertexWriter(vertex_data, "vertex")
        normal_writer = GeomVertexWriter(vertex_data, "normal")
        texcoord_writer = GeomVertexWriter(vertex_data, "texcoord")

        vertices = (
            (-0.5, -0.5, 0.0),
            (0.5, -0.5, 0.0),
            (0.5, 0.5, 0.0),
            (-0.5, 0.5, 0.0),
        )
        texcoords = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

        for vertex, texcoord in zip(vertices, texcoords):
            vertex_writer.add_data3f(*vertex)
            normal_writer.add_data3f(0.0, 0.0, 1.0)
            texcoord_writer.add_data2f(*texcoord)

        triangles = GeomTriangles(GeomEnums.UH_static)
        triangles.add_vertices(0, 1, 2)
        triangles.add_vertices(0, 2, 3)

        geom = Geom(vertex_data)
        geom.add_primitive(triangles)

        geom_node = GeomNode("warp_falling_particles")
        geom_node.add_geom(geom)

        self.node = self.parent.attachNewNode(geom_node)
        self.node.set_two_sided(True)
        self.node.set_shader_input("InstancingData", self.buffer_texture)
        self.node.set_instance_count(self.count)
        self.node.node().set_bounds(OmniBoundingVolume())
        self.node.node().set_final(True)
        self.node.set_color_scale(1.0, 1.0, 1.0, 1.0)

        material = Material()
        material.setBaseColor((0.7, 0.7, 0.7, 1.0))
        material.setRoughness(1.0)
        material.setMetallic(0.0)
        material.setAmbient(0.35)
        material.setRefractiveIndex(1.0)
        self.node.setMaterial(material)

        if self.alpha_blend:
            self.node.set_transparency(TransparencyAttrib.M_alpha)
            self.node.set_depth_write(False)
            self.node.set_bin("transparent", 0)

        if isinstance(texture, Texture):
            loaded_texture = texture
        else:
            loaded_texture = self.showbase.loader.load_texture(texture)
            if loaded_texture is None:
                raise FileNotFoundError(f"Could not load particle texture: {texture}")

        self.node.set_texture(loaded_texture, 1)

        if self.render_pipeline is not None:
            self.render_pipeline.prepare_scene(self.node)
            effect_path = self.effect_path or self._default_effect_path()
            effect_options = self._default_effect_options()
            effect_options.update(self.effect_options)

            if effect_options.get("render_forward", False) and not self._forward_shading_enabled():
                raise RuntimeError(
                    "alpha_blend=True requires the RenderPipeline 'forward_shading' plugin "
                    "to be enabled"
                )

            if effect_path:
                self.render_pipeline.set_effect(
                    self.node,
                    effect_path,
                    effect_options,
                )

    def _queue_state_copy(self) -> None:
        wp.copy(self.positions_cpu, self.positions)
        wp.copy(self.sizes_cpu, self.sizes)

        if self.random_rotation:
            wp.copy(self.spin_angles_cpu, self.spin_angles)
            wp.copy(self.spin_axes_cpu, self.spin_axes)

    @staticmethod
    def _axis_angle_matrices(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
        x = axes[:, 0]
        y = axes[:, 1]
        z = axes[:, 2]
        cos_angle = np.cos(angles).astype(np.float32, copy=False)
        sin_angle = np.sin(angles).astype(np.float32, copy=False)
        one_minus_cos = 1.0 - cos_angle

        matrices = np.empty((axes.shape[0], 3, 3), dtype=np.float32)
        matrices[:, 0, 0] = one_minus_cos * x * x + cos_angle
        matrices[:, 0, 1] = one_minus_cos * x * y - sin_angle * z
        matrices[:, 0, 2] = one_minus_cos * x * z + sin_angle * y
        matrices[:, 1, 0] = one_minus_cos * x * y + sin_angle * z
        matrices[:, 1, 1] = one_minus_cos * y * y + cos_angle
        matrices[:, 1, 2] = one_minus_cos * y * z - sin_angle * x
        matrices[:, 2, 0] = one_minus_cos * x * z - sin_angle * y
        matrices[:, 2, 1] = one_minus_cos * y * z + sin_angle * x
        matrices[:, 2, 2] = one_minus_cos * z * z + cos_angle
        return matrices

    def _update_billboard_matrices(self, positions: np.ndarray, sizes: np.ndarray) -> None:
        quat = self.camera_np.get_quat(self.parent)
        right = quat.xform(Vec3(1.0, 0.0, 0.0))
        up = quat.xform(Vec3(0.0, 0.0, 1.0))
        normal = -quat.xform(Vec3(0.0, 1.0, 0.0))

        right_np = np.array((right.x, right.y, right.z), dtype=np.float32)
        up_np = np.array((up.x, up.y, up.z), dtype=np.float32)
        normal_np = np.array((normal.x, normal.y, normal.z), dtype=np.float32)

        self.matrices.fill(0.0)
        self.matrices[:, 0, 0:3] = right_np[None, :] * sizes[:, None]
        self.matrices[:, 1, 0:3] = up_np[None, :] * sizes[:, None]
        self.matrices[:, 2, 0:3] = normal_np[None, :]
        self.matrices[:, 3, 0:3] = positions
        self.matrices[:, 3, 3] = 1.0

    def _update_random_rotation_matrices(
        self,
        positions: np.ndarray,
        sizes: np.ndarray,
        axes: np.ndarray,
        angles: np.ndarray,
    ) -> None:
        rotation_rows = self._axis_angle_matrices(axes, angles)

        self.matrices.fill(0.0)
        self.matrices[:, 0, 0:3] = rotation_rows[:, 0, :] * sizes[:, None]
        self.matrices[:, 1, 0:3] = rotation_rows[:, 1, :] * sizes[:, None]
        self.matrices[:, 2, 0:3] = rotation_rows[:, 2, :]
        self.matrices[:, 3, 0:3] = positions
        self.matrices[:, 3, 3] = 1.0

    def _upload_matrices(self) -> None:
        data = self.matrices.tobytes()
        ram_image = self.buffer_texture.modify_ram_image()
        ram_image.set_subdata(0, len(data), data)

    def _refresh_geometry(self) -> None:
        positions = self.positions_cpu.numpy()
        sizes = self.sizes_cpu.numpy()

        if self.random_rotation:
            axes = self.spin_axes_cpu.numpy()
            angles = self.spin_angles_cpu.numpy()
            self._update_random_rotation_matrices(positions, sizes, axes, angles)
        else:
            self._update_billboard_matrices(positions, sizes)

        self._upload_matrices()

    def reset(self) -> None:
        self._assert_not_destroyed()
        wp.launch(
            kernel=_initialize_particles,
            dim=self.count,
            inputs=[
                self.positions,
                self.speeds,
                self.sizes,
                self.spin_angles,
                self.spin_speeds,
                self.spin_axes,
                self.respawn_counts,
                tuple(float(v) for v in self.spawn_min),
                tuple(float(v) for v in self.spawn_max),
                self.size_speed_value(self.speed_min),
                self.size_speed_value(self.speed_max),
                self.size_speed_value(self.size_min),
                self.size_speed_value(self.size_max),
                self.size_speed_value(self.spin_speed_min),
                self.size_speed_value(self.spin_speed_max),
                self.random_rotation,
            ],
            device=self.device,
        )
        self._queue_state_copy()
        self._refresh_geometry()

    @staticmethod
    def size_speed_value(value: float) -> np.float32:
        return np.float32(value)

    def step(self, dt: float) -> None:
        self._assert_not_destroyed()
        clamped_dt = np.float32(max(0.0, min(float(dt), self.max_dt)))
        if clamped_dt <= 0.0:
            return

        wp.launch(
            kernel=_update_particles,
            dim=self.count,
            inputs=[
                self.positions,
                self.speeds,
                self.sizes,
                self.spin_angles,
                self.spin_speeds,
                self.spin_axes,
                self.respawn_counts,
                tuple(float(v) for v in self.spawn_min),
                tuple(float(v) for v in self.spawn_max),
                np.float32(self.respawn_z),
                np.float32(self.respawn_top_jitter),
                np.float32(self.respawn_threshold),
                tuple(float(v) for v in self.drift_velocity),
                clamped_dt,
                self.size_speed_value(self.speed_min),
                self.size_speed_value(self.speed_max),
                self.size_speed_value(self.size_min),
                self.size_speed_value(self.size_max),
                self.size_speed_value(self.spin_speed_min),
                self.size_speed_value(self.spin_speed_max),
                self.random_rotation,
            ],
            device=self.device,
        )

    def _task(self, task: Task) -> str:
        self._refresh_geometry()
        self.step(globalClock.get_dt())
        self._queue_state_copy()
        return Task.cont

    def start(self) -> None:
        self._assert_not_destroyed()
        if self.running:
            return

        self._queue_state_copy()
        self._refresh_geometry()
        self.showbase.taskMgr.add(self._task, self.task_name)
        self.running = True

    def stop(self) -> None:
        if self.destroyed or not self.running:
            return

        if self.showbase.taskMgr.hasTaskNamed(self.task_name):
            self.showbase.taskMgr.remove(self.task_name)
        self.running = False

    def destroy(self) -> None:
        if self.destroyed:
            return

        self.stop()

        if self.node is not None:
            self.node.removeNode()

        self.buffer_texture = None
        self.matrices = None
        self.positions = None
        self.speeds = None
        self.sizes = None
        self.spin_angles = None
        self.spin_speeds = None
        self.spin_axes = None
        self.respawn_counts = None
        self.positions_cpu = None
        self.sizes_cpu = None
        self.spin_angles_cpu = None
        self.spin_axes_cpu = None
        self.destroyed = True

    def _assert_not_destroyed(self) -> None:
        if self.destroyed:
            raise RuntimeError("This WarpFallingParticles instance has already been destroyed")
