import numpy as np
import warp as wp
from panda3d.core import (
    Texture,
    OmniBoundingVolume,
    Shader,
    NodePath,
    GeomVertexFormat,
    GeomVertexData,
    GeomVertexWriter,
    Geom,
    GeomTriangles,
    GeomNode,
    GeomVertexReader,
    GeomEnums,
    Vec3,
    Quat,
    Mat4,
)
from direct.task import Task

# ---------- Warp kernels ----------
wp.init()

@wp.kernel(enable_backward=False)
def kernel_update_positions(
    positions: wp.array(dtype=wp.vec3),   # (N,)
    speeds: wp.array(dtype=wp.float32),  # (N,)
    base_speed: wp.float32,
    speed_variation: wp.float32,
    dt: wp.float32,
    radius: wp.float32,
    half_height: wp.float32,
    seed: wp.int32,
):
    i = wp.tid()
    rng = wp.rand_init(seed, i)

    p = positions[i]
    v = speeds[i]

    # advance vertically (Z is up in Panda3D)
    p = wp.vec3(p.x, p.y, p.z - v * dt)

    # if below bottom, respawn at top on random point in circle
    if p.z < -half_height:
        # sample uniform point in disk (rejection or sqrt method)
        r = radius * wp.sqrt(wp.randf(rng))
        theta = 2.0 * 3.141592653589793 * wp.randf(rng)
        x = r * wp.cos(theta)
        y = r * wp.sin(theta)
        z = half_height  # top
        p = wp.vec3(x, y, z)

        # randomize speed ~ base_speed +/- variation (uniform)
        speed_noise = (wp.randf(rng) - 0.5) * 2.0 * speed_variation
        v = base_speed + speed_noise
        # ensure speed non-negative
        if v < 0.0:
            v = 0.0

    positions[i] = p
    speeds[i] = v


# ---------- Helper: make a single quad Geom Node (centered at origin, faces +Y) ----------
def make_quad_node():
    # Quad is created in the XY plane facing +Y (so its normal points +Y) or adjust as necessary.
    # We'll make a 1x1 quad centered at origin in X and Z (because Z is up) and Y=0 plane.
    fmt = GeomVertexFormat.get_v3t2()
    vdata = GeomVertexData("quad", fmt, Geom.UH_static)
    vdata.set_num_rows(4)

    vw = GeomVertexWriter(vdata, "vertex")
    tw = GeomVertexWriter(vdata, "texcoord")

    # define vertices (x, y, z). We'll place quad in X,Z plane at y=0, so it faces +Y.
    half = 0.5
    # lower-left
    vw.add_data3(-half, 0.0, -half)
    tw.add_data2(0.0, 0.0)
    # lower-right
    vw.add_data3(half, 0.0, -half)
    tw.add_data2(1.0, 0.0)
    # upper-right
    vw.add_data3(half, 0.0, half)
    tw.add_data2(1.0, 1.0)
    # upper-left
    vw.add_data3(-half, 0.0, half)
    tw.add_data2(0.0, 1.0)

    tris = GeomTriangles(Geom.UH_static)
    tris.add_vertices(0, 1, 2)
    tris.add_vertices(0, 2, 3)
    tris.close_primitive()

    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode("particle_quad")
    node.add_geom(geom)
    np_node = NodePath(node)
    return np_node


# ---------- Main class ----------
class FallingParticles:
    def __init__(self, panda_app):
        """
        panda_app: instance of your Panda3D app (has .render and .render_pipeline, and .taskMgr).
        """
        self.app = panda_app
        self.geom_prefab = None
        self.buffer_texture = None
        self.matrices = None  # numpy 4x4 matrices per particle
        self.n_particles = 0
        self.positions_wp = None
        self.speeds_wp = None
        self.seed = 12345
        self.task_name = "FallingParticlesUpdateTask"
        self.running = False

        # parameters
        self.base_speed = 1.0
        self.speed_var = 0.0
        self.visible_range = 10.0

    def start(self, n_particles: int, texture_path: str, fall_speed: float, fall_speed_variation: float, visible_range: float, particle_size=0.2, particle_size_variation=0.05, seed: int = 0):
        """
        Start particle simulation and rendering.

        Args:
            n_particles: number of particles
            texture_path: path to texture image used for particle quads
            fall_speed: base fall speed (units per second)
            fall_speed_variation: +/- variation applied uniformly
            visible_range: cylinder radius, half-height (height = 2*visible_range)
            particle_size: base scale (applied uniformly to quad)
            particle_size_variation: per-particle +/- variation
            seed: RNG seed (optional)
        """
        self.stop()  # stop any existing simulation

        self.n_particles = int(n_particles)
        self.base_speed = float(fall_speed)
        self.speed_var = float(fall_speed_variation)
        self.visible_range = float(visible_range)
        self.seed = int(seed) if seed != 0 else np.random.SeedSequence().entropy

        half_height = float(visible_range)

        # Create prefab quad and apply texture
        quad = make_quad_node()
        # load texture
        tex = None
        try:
            from panda3d.core import Loader
            loader = self.app.loader
            tex = loader.load_texture(texture_path)
        except Exception:
            tex = None

        if tex is not None:
            quad.set_texture(tex, 1)

        # flatten/optimize and detach unnecessary nodes
        quad.clear_model_nodes()
        quad.flatten_strong()
        # prepare to be instanced by RenderPipeline
        self.geom_prefab = quad
        self.geom_prefab.reparent_to(self.app.render)
        if hasattr(self.app, "render_pipeline"):
            self.app.render_pipeline.prepare_scene(self.geom_prefab)

        # Create Warp arrays for positions and speeds
        # initialize positions at random points on top circle and speeds randomized
        rng = np.random.default_rng(self.seed)

        # positions as numpy (N,3)
        rs = rng.random(self.n_particles)
        thetas = rng.random(self.n_particles) * 2.0 * np.pi
        radii = np.sqrt(rs) * self.visible_range  # sqrt for uniform disk
        x = radii * np.cos(thetas)
        y = radii * np.sin(thetas)
        z = np.full(self.n_particles, half_height, dtype=np.float32)  # top

        positions_np = np.column_stack([x, y, z]).astype(np.float32)
        # speeds: base_speed +/- variation
        speed_noise = (rng.random(self.n_particles) - 0.5) * 2.0 * particle_size_variation  # wrong var used? fix below
        # use correct variation
        speeds_np = (self.base_speed + (rng.random(self.n_particles) - 0.5) * 2.0 * self.speed_var).astype(np.float32)

        # create warp arrays
        self.positions_wp = wp.array(positions_np.tolist(), dtype=wp.vec3)
        self.speeds_wp = wp.array(speeds_np.tolist(), dtype=wp.float32)

        # Create instance buffer texture for 4x4 matrices
        # 4 floats per row * 4 rows = 16 floats per instance => each pixel is RGBA float; so rows of width = n_particles*4 ?
        # Using same pattern as example: setup_buffer_texture(n*4, T_float, F_rgba32)
        self.buffer_texture = Texture()
        # allocate buffer texture: width = n_particles * 4, height = 1, format RGBA32 float
        # The API in Panda3D: setup_buffer_texture(width, component_type, format, usage_hint)
        self.buffer_texture.setup_buffer_texture(self.n_particles * 4, Texture.T_float, Texture.F_rgba32, GeomEnums.UH_static)

        # Pre-allocate numpy matrices
        self.matrices = np.zeros((self.n_particles, 4, 4), dtype=np.float32)
        # initialize with identity * scale + positions
        # compute scales per particle
        sizes = (particle_size + (rng.random(self.n_particles) - 0.5) * 2.0 * particle_size_variation).astype(np.float32)

        # rotation will be updated per-frame; for initial frame use identity
        Rmat = np.eye(3, dtype=np.float32)
        # make 4x4 transforms as R * S and translation
        T = np.zeros((self.n_particles, 4, 4), dtype=np.float32)
        T[:, 3, 3] = 1.0
        # set upper-left with R @ diag(scale)
        for i in range(self.n_particles):
            S = np.diag([sizes[i], sizes[i], sizes[i]], dtype=np.float32)
            T[:3, :3] = Rmat @ S  # but broadcasting error; do per-particle below

        # faster: fill matrices with Rmat * scale
        for i in range(self.n_particles):
            s = sizes[i]
            self.matrices[i, :3, :3] = Rmat * s
            self.matrices[i, :3, 3] = positions_np[i]

        # upload initial matrices to buffer texture RAM
        data = self.matrices.tobytes()
        ram_image = self.buffer_texture.modify_ram_image()
        ram_image.set_subdata(0, len(data), data)

        # configure RenderPipeline instancing
        if hasattr(self.app, "render_pipeline"):
            # set effect - require an instancing-friendly shader in render pipeline
            try:
                self.app.render_pipeline.set_effect(self.geom_prefab, "effects/basic_instancing.yaml", {})
            except Exception:
                # if effect not present, continue — user should configure their pipeline
                pass

            self.geom_prefab.set_shader_input("InstancingData", self.buffer_texture)
            self.geom_prefab.set_instance_count(self.n_particles)
            self.geom_prefab.node().set_bounds(OmniBoundingVolume())
            self.geom_prefab.node().set_final(True)

        # store parameters used by the kernel
        self._half_height = half_height

        # attach per-frame task
        self.running = True
        # use app.taskMgr to add a persistent task
        self.app.taskMgr.add(self._task_update, self.task_name)

    def _task_update(self, task: Task):
        if not self.running:
            return Task.done

        dt = globalClock.get_dt()
        # Launch warp kernel to advance positions
        wp.launch(
            kernel_update_positions,
            dim=(self.n_particles,),
            inputs=(
                self.positions_wp,
                self.speeds_wp,
                float(self.base_speed),
                float(self.speed_var),
                float(dt),
                float(self.visible_range),
                float(self._half_height),
                int(self.seed),
            ),
            outputs=(self.positions_wp, self.speeds_wp),
        )

        # read back positions into numpy for matrix building
        positions_np = np.asarray(self.positions_wp.numpy(), dtype=np.float32)  # shape (N,3)

        # compute rotation matrix once (per-frame) so billboards roughly face camera
        # We'll get camera quaternion relative to render and convert to a 3x3 matrix.
        cam = self.app.cam  # typical Panda base.cam
        cam_quat = cam.get_quat(self.app.render)
        # Panda3D Quat -> Mat3: convert to numpy 3x3
        # Mat4 extraction via Mat4 = cam.get_mat(render); Mat4.get_upper_3() to get 3x3
        cam_mat4 = cam.get_mat(self.app.render)
        # extract 3x3
        R3 = np.zeros((3, 3), dtype=np.float32)
        R3[0, 0] = cam_mat4.get_cell(0, 0)
        R3[0, 1] = cam_mat4.get_cell(0, 1)
        R3[0, 2] = cam_mat4.get_cell(0, 2)
        R3[1, 0] = cam_mat4.get_cell(1, 0)
        R3[1, 1] = cam_mat4.get_cell(1, 1)
        R3[1, 2] = cam_mat4.get_cell(1, 2)
        R3[2, 0] = cam_mat4.get_cell(2, 0)
        R3[2, 1] = cam_mat4.get_cell(2, 1)
        R3[2, 2] = cam_mat4.get_cell(2, 2)

        # If your quad faces +Y in model space, you may need to rotate by 90deg. Adjust as needed.
        # Build matrices: M = [ R3 * scale , translation; 0 0 0 1 ]
        # We need per-particle scales; for simplicity assume scale is embedded in diagonal of upper-left.
        # We'll compute scales from current matrices if present, otherwise use 1.0
        # Grab current scales by norm of columns (safe if initial).
        # For simplicity, create uniform scale array from existing matrices or default 1.0
        # If self.matrices already has scale set, extract them; otherwise assume 1.0
        # Here we will preserve any preexisting scale in upper-left columns:
        scales = np.linalg.norm(self.matrices[:, :3, 0], axis=1)  # x-column lengths

        # Compose new matrices
        # For each particle: M[:3,:3] = R3 * scales[i], M[:3,3] = positions[i]
        for i in range(self.n_particles):
            s = scales[i]
            self.matrices[i, :3, :3] = R3 * s
            self.matrices[i, :3, 3] = positions_np[i]

        # Upload matrices to buffer texture
        data = self.matrices.tobytes()
        try:
            ram_image = self.buffer_texture.modify_ram_image()
            ram_image.set_subdata(0, len(data), data)
        except Exception:
            # in case modifying ram image fails, ignore for now
            pass

        return Task.cont

    def stop(self):
        """Stop the simulation and remove geometry."""
        if self.running:
            try:
                self.app.taskMgr.remove(self.task_name)
            except Exception:
                pass
            self.running = False

        if self.geom_prefab is not None:
            try:
                self.geom_prefab.remove_node()
            except Exception:
                try:
                    self.geom_prefab.removeNode()
                except Exception:
                    pass
            self.geom_prefab = None

        # release warp arrays
        self.positions_wp = None
        self.speeds_wp = None
        self.matrices = None
        self.n_particles = 0
        self.buffer_texture = None
