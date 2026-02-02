import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

@wp.kernel(enable_backward=False)
def compute_tri_areas(
    points: wp.array(dtype=wp.vec3),
    face_vertex_indices: wp.array(dtype=wp.int32),
    out_tri_areas: wp.array(dtype=wp.float32),
    out_total_area: wp.array(dtype=wp.float32),
):
    tri = wp.tid()

    # Retrieve the indices of the three vertices that form the current triangle.
    vtx_0 = face_vertex_indices[tri * 3]
    vtx_1 = face_vertex_indices[tri * 3 + 1]
    vtx_2 = face_vertex_indices[tri * 3 + 2]

    # Retrieve their 3D position.
    pt_0 = points[vtx_0]
    pt_1 = points[vtx_1]
    pt_2 = points[vtx_2]

    # Calculate the cross product of two edges of the triangle,
    # which gives a vector whose magnitude is twice the area of the triangle.
    cross = wp.cross((pt_1 - pt_0), (pt_2 - pt_0))
    len = wp.length(cross)
    area = len * 0.5
    normal = cross / len
    target = wp.vec3(0.0, 0.0, 1.0)

    if wp.dot(normal, target) < 0.000001:
        area = 0.0

    # Store the result.
    out_tri_areas[tri] = area
    wp.atomic_add(out_total_area, 0, area)


@wp.kernel(enable_backward=False)
def compute_probability_distribution(
    tri_areas: wp.array(dtype=wp.float32),
    total_area: wp.array(dtype=wp.float32),
    out_probabilities: wp.array(dtype=wp.float32),
):
    tri = wp.tid()

    # Calculate the probability of selecting this triangle,
    # which is proportional to the triangle's area relative to total mesh area.
    out_probabilities[tri] = tri_areas[tri] / total_area[0]


@wp.kernel(enable_backward=False)
def accumulate_cdf(
    tri_count: wp.int32,
    out_cdf: wp.array(dtype=wp.float32),
):
    # Transform probability values into a Cumulative Distribution Function (CDF).
    for tri in range(1, tri_count):
        out_cdf[tri] += out_cdf[tri - 1]


@wp.kernel(enable_backward=False)
def sample_mesh(
    mesh: wp.uint64,
    cdf: wp.array(dtype=wp.float32),
    seed: wp.int32,
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    rng = wp.rand_init(seed, tid)

    # Sample the triangle index using the CDF.
    sample = wp.randf(rng)
    tri = wp.lower_bound(cdf, sample)

    # Sample the location in that triangle using random barycentric coordinates.
    ru = wp.randf(rng)
    rv = wp.randf(rng)
    tri_u = 1.0 - wp.sqrt(ru)
    tri_v = wp.sqrt(ru) * (1.0 - rv)
    pos = wp.mesh_eval_position(mesh, tri, tri_u, tri_v)

    # Store the result.
    out_points[tid] = pos


class MeshDistributor:
    def __init__(self, panda_app):
        self.panda_app = panda_app

    def distribute(self, vertices, indices, distributions, seed=0):
        self.mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3),
            indices=wp.array(indices, dtype=wp.int32),
        )
        self.tri_count = len(indices) // 3

        # Compute the area of each triangle and the total area of the mesh.
        tri_areas = wp.empty(shape=(self.tri_count,), dtype=wp.float32)
        total_area = wp.zeros(shape=(1,), dtype=wp.float32)
        wp.launch(
            compute_tri_areas,
            dim=tri_areas.shape,
            inputs=(
                self.mesh.points,
                self.mesh.indices,
            ),
            outputs=(
                tri_areas,
                total_area,
            ),
        )

        # Build a Cumulative Distribution Function (CDF) where the probability
        # of sampling a given triangle is proportional to its area.
        self.cdf = wp.empty(shape=(self.tri_count,), dtype=wp.float32)
        wp.launch(
            compute_probability_distribution,
            dim=self.cdf.shape,
            inputs=(
                tri_areas,
                total_area,
            ),
            outputs=(self.cdf,),
        )
        wp.launch(
            accumulate_cdf,
            dim=(1,),
            inputs=(self.tri_count,),
            outputs=(self.cdf,),
        )

        # Array to store the sampled points.
        self.n_particles = distributions
        self.points = wp.empty(shape=(distributions,), dtype=wp.vec3)

        self.seed = seed

        wp.launch(
            sample_mesh,
            dim=self.points.shape,
            inputs=(
                self.mesh.id,
                self.cdf,
                self.seed,
            ),
            outputs=(self.points,),
        )

        return self.points
    
    def start_rendering(self, model, particle_size, particle_size_variation):
        from panda3d.core import Texture, GeomEnums
        from panda3d.core import OmniBoundingVolume
        # визуализация инстансингом
        self.buffer_texture = Texture()
        self.buffer_texture.setup_buffer_texture(self.n_particles * 4, Texture.T_float, Texture.F_rgba32, GeomEnums.UH_static)
        self.matrices = self.matrix = np.zeros((self.n_particles, 4, 4), dtype=np.float32)
        self.geom_prefab = None

        num_objects = self.matrix.shape[0]

        # Use a single NumPy generator (seed fixed)
        rng = np.random.default_rng(self.seed)

        noise = rng.uniform(-1, 1, (num_objects, 3))  # [-1, 1] like xorshift output

        # 2. Particle sizes with variation
        size_base = particle_size
        size_var = particle_size_variation
        scales = (size_base + size_var * noise)  # (N, 3)

        # 3. Fast random rotations: sample quaternions uniformly
        # Uniform random unit quaternions → uniform SO(3)
        u1, u2, u3 = rng.random((3, num_objects))
        q_w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        q_x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        q_y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        q_z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        rotations = R.from_quat(np.column_stack([q_x, q_y, q_z, q_w]))
        R_mats = rotations.as_matrix()  # (N, 3, 3)

        # 4. Build 4x4 transformation matrices efficiently
        # We want matrix[i] = [[R @ S3], [0 0 0 1]], where S3 = diag(sx,sy,sz)
        # So: first scale, then rotate → T = R @ S
        T = np.zeros((num_objects, 4, 4))
        T[:, 3, 3] = 1.0
        T[:, :3, :3] = R_mats * scales[:, None, :]  # Broadcasting: apply scale on columns (right-multiply diag)
        T[:, :3, 3] = 0  # translation zero

        # Assign all at once
        self.matrix[:] = T

        # Убираем лишние трасформации, делаем её неотрисовываемой
        model.clear_model_nodes()
        model.flatten_strong()          # оптимизируем
        model.node().set_bounds_type(0) # не будет кадрироваться
        model.detach_node()             # убираем со сцены
        
        model.reparent_to(self.panda_app.render)
        self.panda_app.render_pipeline.prepare_scene(model)

        self.geom_prefab = model

        self.panda_app.render_pipeline.set_effect(self.geom_prefab, "effects/basic_instancing.yaml", {})

        self.geom_prefab.set_shader_input("InstancingData", self.buffer_texture)
        self.geom_prefab.set_instance_count(self.n_particles)

        # We have do disable culling, so that all instances stay visible
        self.geom_prefab.node().set_bounds(OmniBoundingVolume())
        self.geom_prefab.node().set_final(True)

        self.matrix[:self.n_particles, 3, 0:3] = self.points.numpy()
        data = self.matrix.tobytes()
        ram_image = self.buffer_texture.modify_ram_image()
        ram_image.set_subdata(0, len(data), data)

    def stop_rendering(self):
        # Удаляем геометрию инстансинга, если она была создана
        if hasattr(self, 'geom_prefab') and self.geom_prefab is not None:
            self.geom_prefab.removeNode()
            self.geom_prefab = None