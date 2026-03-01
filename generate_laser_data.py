"""
FEniCSx-Based 2D Laser Heat Treatment Data Generator for Complex Geometries

Description:
- Solves the transient heat equation to simulate the heating effect of moving
  laser beams on a complex 2D metal part.
- The geometry is a complex plate with a central gear-shaped hole and two
  circular side holes, representing a realistic mechanical component.
- The simulation models multiple laser beams moving along various complex paths,
  acting as dynamic, high-intensity heat sources.
- The paths are a mix of deterministic (orbiting key features) and stochastic
  (e.g., Lissajous curves, random waypoints), designed to rigorously test a
  model's adaptability to a wide variety of unseen dynamic source terms.
- The resulting temperature field is highly dynamic and complex, with sharp,
  moving peaks and complex thermal gradients.
- All boundaries lose heat to the environment via convection.
"""

import os
import h5py
import numpy as np
import dolfinx
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import gmsh
from dolfinx.io import gmshio
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import splev, splprep

# =============================================================================
# --- 0. Simulation and Dataset Parameters ---
# =============================================================================
# --- Files and Paths ---
output_dir = "gksNet/data_laser_hardening"
h5_filename = os.path.join(output_dir, "pde_trajectories.h5")
mesh_vis_filename = os.path.join(output_dir, "laser_mesh.png")
snapshots_vis_filename = os.path.join(output_dir, "temperature_snapshots.png")
source_vis_filename = os.path.join(output_dir, "source_snapshots.png")

# --- Geometry & Mesh Parameters ---
tank_w, tank_h = 1.4, 0.9
center_x, center_y = tank_w / 2, tank_h / 2
gear_outer_r, gear_inner_r = 0.21, 0.18
hole_r = 0.096 # Radius of the two side holes
hole_offset = 0.45 # Horizontal offset of the side holes from the center

# --- Physics & Time (Properties for Steel) ---
rho = 7850.0       # Density (kg/m^3)
specific_heat = 450.0  # Specific Heat (J/kg*K)
conductivity = 50.0    # Thermal Conductivity (W/m*K)
h_conv = 25.0        # Convective heat transfer coefficient (W/m^2*K)
T_ambient = 298.15   # Ambient temperature (25°C in Kelvin)

# --- Laser Parameters ---
NUM_LASERS = 10
MAX_SPEED = 0.8        # Max speed for linear movements (m/s)
ANGULAR_VELOCITY = 1.0 # rad/s for orbital paths (will result in ~10 laps in 60s)

# --- Time Discretization ---
T_sim = 60.0   # Total simulation time (seconds)
dt = 0.5       # Time step size (seconds)

# --- Dataset Parameters ---
NUM_TRAJECTORIES = 40 # Generate 5 independent simulation trajectories

# =============================================================================
# --- 1. Mesh Generation (Integrated) ---
# =============================================================================
def create_mesh_with_tags(comm: MPI.Comm):
    # This function is a self-contained copy of our final mesh script
    gmsh.initialize()
    if comm.rank == 0: gmsh.option.setNumber("General.Terminal", 1)
    else: gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model; model.add("laser_part")
    
    num_teeth = 12
    lc_fine, lc_coarse = 0.018, 0.036  # Adjusted to increase node count to 6k-7k range
    p = [ model.occ.addPoint(0,0,0), model.occ.addPoint(tank_w,0,0), model.occ.addPoint(tank_w,tank_h,0), model.occ.addPoint(0,tank_h,0), model.occ.addPoint(tank_w*0.5,-tank_h*0.1,0), model.occ.addPoint(tank_w*1.1,tank_h*0.5,0), model.occ.addPoint(tank_w*0.5,tank_h*1.1,0), model.occ.addPoint(-tank_w*0.1,tank_h*0.5,0) ]
    c = [ model.occ.addSpline([p[0],p[4],p[1]]), model.occ.addSpline([p[1],p[5],p[2]]), model.occ.addSpline([p[2],p[6],p[3]]), model.occ.addSpline([p[3],p[7],p[0]]) ]
    tank_loop = model.occ.addCurveLoop(c)
    tank_surface_tag = model.occ.addPlaneSurface([tank_loop])
    gear_pts = [model.occ.addPoint( center_x + (gear_outer_r if i%2==0 else gear_inner_r)*np.cos(np.pi*i/num_teeth), center_y + (gear_outer_r if i%2==0 else gear_inner_r)*np.sin(np.pi*i/num_teeth), 0, lc_fine) for i in range(num_teeth*2)]
    gear_lines = [model.occ.addLine(gear_pts[i], gear_pts[(i+1)%len(gear_pts)]) for i in range(len(gear_pts))]
    gear_loop = model.occ.addCurveLoop(gear_lines); gear_surface = model.occ.addPlaneSurface([gear_loop])
    
    # Add two new circular holes on the left and right
    left_hole_center = np.array([center_x - hole_offset, center_y, 0])
    right_hole_center = np.array([center_x + hole_offset, center_y, 0])
    left_hole_curve = model.occ.addCircle(left_hole_center[0], left_hole_center[1], left_hole_center[2], hole_r)
    left_hole_loop = model.occ.addCurveLoop([left_hole_curve])
    left_hole_surface = model.occ.addPlaneSurface([left_hole_loop])
    right_hole_curve = model.occ.addCircle(right_hole_center[0], right_hole_center[1], right_hole_center[2], hole_r)
    right_hole_loop = model.occ.addCurveLoop([right_hole_curve])
    right_hole_surface = model.occ.addPlaneSurface([right_hole_loop])
    
    domain_surface, _ = model.occ.cut([(2, tank_surface_tag)], [(2, gear_surface), (2, left_hole_surface), (2, right_hole_surface)]); model.occ.synchronize()
    
    all_boundaries = model.getBoundary(domain_surface, combined=True, oriented=False)
    boundary_curves = [c[1] for c in all_boundaries]
    model.addPhysicalGroup(2, [s[1] for s in domain_surface], 1, name="domain")
    model.addPhysicalGroup(1, boundary_curves, 1, name="convection_boundary")
    
    model.mesh.field.add("Distance", 1); model.mesh.field.setNumbers(1, "CurvesList", boundary_curves)
    model.mesh.field.add("Threshold", 2); model.mesh.field.setNumber(2, "InField", 1)
    model.mesh.field.setNumber(2, "SizeMin", lc_fine); model.mesh.field.setNumber(2, "SizeMax", lc_coarse)
    model.mesh.field.setNumber(2, "DistMin", 0); model.mesh.field.setNumber(2, "DistMax", 0.15)
    model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.Algorithm", 5); model.mesh.generate(2)
    domain, _, facet_tags = gmshio.model_to_mesh(model, comm, rank=0, gdim=2)
    gmsh.finalize()
    return domain, facet_tags

# =============================================================================
# --- 2. Physics Solvers Setup ---
# =============================================================================
def setup_solvers(domain: mesh.Mesh, facet_tags: mesh.MeshTags):
    # --- Thermal Problem ---
    T_space = fem.functionspace(domain, ("Lagrange", 1))
    T_h, T_n = fem.Function(T_space, name="Temperature"), fem.Function(T_space)
    p, q = ufl.TrialFunction(T_space), ufl.TestFunction(T_space)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    Q_func = fem.Function(T_space, name="Source")
    # Convection on all boundaries (Tag 1)
    F_therm = (rho*specific_heat*(p-T_n)/dt*q*ufl.dx 
               + conductivity*ufl.inner(ufl.grad(p),ufl.grad(q))*ufl.dx 
               + h_conv*(p-T_ambient)*q*ds(1) 
               - Q_func*q*ufl.dx)
    a_therm, L_therm = fem.form(ufl.lhs(F_therm)), fem.form(ufl.rhs(F_therm))
    problem_therm = LinearProblem(a_therm, L_therm, u=T_h, bcs=[], 
                                  petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    
    return problem_therm, T_n, T_h, T_space, Q_func

# =============================================================================
# --- 3. Laser Trajectory Generation ---
# =============================================================================
class BasePath:
    def __init__(self, rng, geom_params):
        self.rng = rng
        self.geom_params = geom_params
        self.active = True
        # Power and Radius are now set in subclasses based on their role
        self.power = 0.0
        self.radius = 0.0

    def get_position(self, t):
        raise NotImplementedError

    def describe(self):
        return f"Power: {self.power:.2e}, Radius: {self.radius:.3f}"

    def _generate_safe_point(self):
        """Generates a random point that is not inside one of the holes."""
        while True:
            p = self.rng.uniform(low=[0.1, 0.1], high=[self.geom_params['tank_w']-0.1, self.geom_params['tank_h']-0.1])
            center = np.array([self.geom_params['center_x'], self.geom_params['center_y']])
            if np.linalg.norm(p - center) < self.geom_params['gear_outer_r'] + 0.05: continue
            if np.linalg.norm(p - self.geom_params['left_hole_center'][:2]) < self.geom_params['hole_r'] + 0.05: continue
            if np.linalg.norm(p - self.geom_params['right_hole_center'][:2]) < self.geom_params['hole_r'] + 0.05: continue
            return p

class OrbitPath(BasePath):
    def __init__(self, rng, geom_params, orbit_type):
        super().__init__(rng, geom_params)
        self.orbit_type = orbit_type

        # Selectively assign power based on the orbit target
        if self.orbit_type in ['left_hole', 'right_hole', 'center_gear']:
            # Reduced high-power for the three inner features
            self.power = self.rng.uniform(0.8e8, 1.1e8)
        else: # outer_boundary
            # Original high-power for the outer boundary
            self.power = self.rng.uniform(1.2e8, 1.5e8)
        
        # Radius remains the same for all specialists
        self.radius = self.rng.uniform(0.03, 0.04)
        
        self.w = ANGULAR_VELOCITY * self.rng.uniform(0.8, 1.2)
        
        if self.orbit_type == 'left_hole':
            self.center = self.geom_params['left_hole_center'][:2]
            self.radius_path = self.geom_params['hole_r'] + 0.05
        elif self.orbit_type == 'right_hole':
            self.center = self.geom_params['right_hole_center'][:2]
            self.radius_path = self.geom_params['hole_r'] + 0.05
        elif self.orbit_type == 'center_gear':
            self.center = np.array([self.geom_params['center_x'], self.geom_params['center_y']])
            self.radius_path = self.geom_params['gear_outer_r'] + 0.05
        else: # outer_boundary
            self.center = np.array([self.geom_params['center_x'], self.geom_params['center_y']])
            self.radius_path = max(self.geom_params['tank_w'], self.geom_params['tank_h']) / 2 * 0.95

    def get_position(self, t):
        x = self.center[0] + self.radius_path * np.cos(self.w * t)
        y = self.center[1] + self.radius_path * np.sin(self.w * t)
        return np.array([x, y])

    def describe(self):
        base_desc = super().describe()
        return f"OrbitPath around {self.orbit_type}. {base_desc}. Duration: Full Sim (inf)."

class WaypointPath(BasePath):
    def __init__(self, rng, geom_params):
        super().__init__(rng, geom_params)
        # Low-power, diffuse generalist for pre-heating/annealing
        self.power = self.rng.uniform(0.3e8, 0.6e8)
        self.radius = self.rng.uniform(0.08, 0.12)

        self.num_points = self.rng.integers(5, 11)
        self.waypoints = np.array([self._generate_safe_point() for _ in range(self.num_points)])
        distances = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.total_dist = np.sum(distances)
        self.travel_time = self.total_dist / MAX_SPEED
        self.segment_times = np.concatenate(([0], np.cumsum(distances / MAX_SPEED)))

    def get_position(self, t):
        if t > self.travel_time:
            self.active = False
            return None
        
        segment_idx = np.searchsorted(self.segment_times, t, side='right') - 1
        t_in_segment = t - self.segment_times[segment_idx]
        
        start_node = self.waypoints[segment_idx]
        end_node = self.waypoints[segment_idx + 1]
        direction = (end_node - start_node) / np.linalg.norm(end_node - start_node)
        
        return start_node + direction * MAX_SPEED * t_in_segment

    def describe(self):
        base_desc = super().describe()
        return f"WaypointPath traversing {self.num_points} points. {base_desc}. Duration: {self.travel_time:.2f}s."

class LissajousPath(BasePath):
    def __init__(self, rng, geom_params):
        super().__init__(rng, geom_params)
        # Low-power, diffuse generalist for pre-heating/annealing
        self.power = self.rng.uniform(0.3e8, 0.6e8)
        self.radius = self.rng.uniform(0.08, 0.12)

        self.A = self.rng.uniform(0.1, self.geom_params['tank_w'] / 2 - 0.2)
        self.B = self.rng.uniform(0.1, self.geom_params['tank_h'] / 2 - 0.2)
        self.a = self.rng.integers(1, 5)
        self.b = self.rng.integers(1, 5)
        self.delta = self.rng.uniform(0, np.pi)
        self.center = np.array([self.geom_params['center_x'], self.geom_params['center_y']])
    
    def get_position(self, t):
        x = self.center[0] + self.A * np.sin(self.a * t * 0.2 + self.delta)
        y = self.center[1] + self.B * np.sin(self.b * t * 0.2)
        return np.array([x, y])

    def describe(self):
        base_desc = super().describe()
        return f"LissajousPath (a={self.a}, b={self.b}). {base_desc}. Duration: Full Sim (inf)."

class SplinePath(BasePath):
    def __init__(self, rng, geom_params):
        super().__init__(rng, geom_params)
        # Low-power, diffuse generalist for pre-heating/annealing
        self.power = self.rng.uniform(0.3e8, 0.6e8)
        self.radius = self.rng.uniform(0.08, 0.12)

        self.num_points = self.rng.integers(5, 9)
        points = np.array([self._generate_safe_point() for _ in range(self.num_points)])
        tck, u = splprep([points[:, 0], points[:, 1]], s=0)
        self.tck = tck
        self.duration = self.rng.uniform(40, 60)

    def get_position(self, t):
        if t > self.duration:
            self.active = False
            return None
        u_interp = t / self.duration
        x, y = splev(u_interp, self.tck)
        return np.array([x, y])

    def describe(self):
        base_desc = super().describe()
        return f"SplinePath through {self.num_points} points. {base_desc}. Duration: {self.duration:.2f}s."

class RasterScanPath(BasePath):
    def __init__(self, rng, geom_params):
        super().__init__(rng, geom_params)
        # Low-power, diffuse generalist for pre-heating/annealing
        self.power = self.rng.uniform(0.3e8, 0.6e8)
        self.radius = self.rng.uniform(0.08, 0.12)

        self.w = self.rng.uniform(0.2, 0.5)
        self.h = self.rng.uniform(0.2, 0.5)
        start_point = self._generate_safe_point()
        self.x_range = [start_point[0] - self.w/2, start_point[0] + self.w/2]
        self.y_range = [start_point[1] - self.h/2, start_point[1] + self.h/2]
        self.num_lines = self.rng.integers(5, 10)
        self.y_steps = np.linspace(self.y_range[1], self.y_range[0], self.num_lines)
        self.line_duration = (self.x_range[1] - self.x_range[0]) / MAX_SPEED
        self.duration = self.num_lines * self.line_duration

    def get_position(self, t):
        if t > self.duration:
            self.active = False
            return None
        line_idx = int(t / self.line_duration)
        t_in_line = t % self.line_duration
        y = self.y_steps[line_idx]
        
        if line_idx % 2 == 0: # Move right
            x = self.x_range[0] + MAX_SPEED * t_in_line
        else: # Move left
            x = self.x_range[1] - MAX_SPEED * t_in_line
        return np.array([x, y])

    def describe(self):
        base_desc = super().describe()
        return f"RasterScanPath over a {self.w:.2f}x{self.h:.2f} area. {base_desc}. Duration: {self.duration:.2f}s."

class BilliardPath(BasePath):
    def __init__(self, rng, geom_params):
        super().__init__(rng, geom_params)
        # Low-power, diffuse generalist for pre-heating/annealing
        self.power = self.rng.uniform(0.3e8, 0.6e8)
        self.radius = self.rng.uniform(0.08, 0.12)

        self.pos = self._generate_safe_point()
        angle = self.rng.uniform(0, 2 * np.pi)
        self.vel = np.array([np.cos(angle), np.sin(angle)]) * MAX_SPEED
        self.last_update_time = 0.0
    
    def get_position(self, t):
        dt = t - self.last_update_time
        self.pos += self.vel * dt
        
        # Simplified boundary collision (rectangle) - A full implementation is much more complex
        if not (0.05 < self.pos[0] < self.geom_params['tank_w'] - 0.05):
            self.vel[0] *= -1
        if not (0.05 < self.pos[1] < self.geom_params['tank_h'] - 0.05):
            self.vel[1] *= -1
            
        self.last_update_time = t
        return self.pos

    def describe(self):
        base_desc = super().describe()
        return f"BilliardPath with rectangular reflection. {base_desc}. Duration: Full Sim (inf)."

def initialize_lasers(num_lasers, geom_params):
    rng = np.random.default_rng(seed=np.random.randint(0, 10000))
    
    # --- New Logic: Ensure base paths + add random paths ---
    lasers = []
    
    # 1. Force-assign the four core orbital paths (Specialists)
    lasers.append(OrbitPath(rng, geom_params, 'left_hole'))
    lasers.append(OrbitPath(rng, geom_params, 'right_hole'))
    lasers.append(OrbitPath(rng, geom_params, 'center_gear'))
    lasers.append(OrbitPath(rng, geom_params, 'outer_boundary'))

    # 2. Fill the rest with random paths (Generalists)
    random_path_types = [
        'waypoint', 'lissajous', 'spline', 'raster', 'billiard'
    ]
    num_random_lasers = num_lasers - len(lasers)
    
    for _ in range(num_random_lasers):
        choice = rng.choice(random_path_types)
        if choice == 'waypoint':
            lasers.append(WaypointPath(rng, geom_params))
        elif choice == 'lissajous':
            lasers.append(LissajousPath(rng, geom_params))
        elif choice == 'spline':
            lasers.append(SplinePath(rng, geom_params))
        elif choice == 'raster':
            lasers.append(RasterScanPath(rng, geom_params))
        elif choice == 'billiard':
            lasers.append(BilliardPath(rng, geom_params))
    
    print("\n--- Laser Configuration for this Run ---")
    for i, laser in enumerate(lasers):
        print(f"  Laser #{i+1}: {laser.describe()}")
    print("----------------------------------------\n")

    return lasers

# =============================================================================
# --- 4. Dataset Generation ---
# =============================================================================
def generate_dataset():
    comm = MPI.COMM_WORLD
    if comm.rank == 0 and not os.path.exists(output_dir): os.makedirs(output_dir)

    if comm.rank == 0: print("--- 1. Generating shared mesh ---")
    domain, facet_tags = create_mesh_with_tags(comm)
    
    # --- Extract static mesh info once ---
    domain.topology.create_connectivity(domain.topology.dim, 0) # cells to vertices
    domain.topology.create_connectivity(1, 0) # edges to vertices
    
    nodes_coords = domain.geometry.x[:, :2]
    num_nodes = nodes_coords.shape[0]
    
    faces_data = np.sort(domain.topology.connectivity(domain.topology.dim, 0).array.reshape(-1, 3), axis=1)
    
    edge_set = set()
    for face in faces_data:
        edge_set.add(tuple(sorted((face[0], face[1]))))
        edge_set.add(tuple(sorted((face[1], face[2]))))
        edge_set.add(tuple(sorted((face[2], face[0]))))
    edges_data = np.array(list(edge_set))

    if comm.rank == 0:
        print(f"--- Mesh generated successfully with {num_nodes} nodes, {len(edges_data)} edges, {len(faces_data)} faces. ---")
        plt.figure(figsize=(12, 8))
        plt.triplot(nodes_coords[:, 0], nodes_coords[:, 1], faces_data, color='black', linewidth=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Generated Mesh ({num_nodes} nodes)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.savefig(mesh_vis_filename)
        plt.close()
        print(f"Mesh visualization saved to: {mesh_vis_filename}")

    geom_params = {
        "tank_w": tank_w, "tank_h": tank_h, "center_x": center_x, "center_y": center_y,
        "gear_outer_r": gear_outer_r, "hole_r": hole_r,
        "left_hole_center": np.array([center_x - hole_offset, center_y, 0]),
        "right_hole_center": np.array([center_x + hole_offset, center_y, 0])
    }

    with h5py.File(h5_filename, 'w') as f:
        for i in range(NUM_TRAJECTORIES):
            if comm.rank == 0:
                print(f"\n--- Generating Trajectory {i}/{NUM_TRAJECTORIES-1} ---")

            traj_group = f.create_group(f"trajectory_{i}")
            
            time_points = np.arange(0, T_sim + dt, dt)
            num_time_steps = len(time_points)
            
            if comm.rank == 0: print("  --- 2. Setting up physics solvers ---")
            prob_therm, T_n, T_h, T_s, Q_func = setup_solvers(domain, facet_tags)
            T_n.x.array[:] = T_ambient
            
            hists = { "node_features": np.zeros((num_time_steps, num_nodes, 1)),
                      "source_terms": np.zeros((num_time_steps, num_nodes, 1)) }
            hists["node_features"][0, :, 0] = T_ambient
            
            if comm.rank == 0: print(f"  --- 3. Starting time-stepping for trajectory {i} ---")
            
            lasers = initialize_lasers(NUM_LASERS, geom_params)
            
            node_coords_func = T_s.tabulate_dof_coordinates()[:,:2]

            for t_idx in range(1, num_time_steps):
                current_time = time_points[t_idx]
                
                total_heat_values = np.zeros(num_nodes)
                for laser in lasers:
                    pos = laser.get_position(current_time)
                    if pos is not None and laser.active:
                        radii_sq = laser.radius**2
                        dist_sq = np.sum((node_coords_func - pos)**2, axis=1)
                        heat = laser.power * np.exp(-dist_sq / (2 * radii_sq))
                        total_heat_values += heat
                
                Q_func.x.array[:] = total_heat_values

                prob_therm.solve()
                hists["node_features"][t_idx, :, 0] = T_h.x.array
                hists["source_terms"][t_idx, :, 0] = Q_func.x.array
                T_n.x.array[:] = T_h.x.array
                
            if comm.rank == 0: print(f"  --- 4. Saving trajectory {i} to HDF5 ---")
            
            # Save all datasets according to the specified structure
            traj_group.create_dataset("nodes", data=nodes_coords.astype(np.float32))
            traj_group.create_dataset("edges", data=edges_data.astype(np.int32))
            traj_group.create_dataset("faces", data=faces_data.astype(np.int32))
            traj_group.create_dataset("node_features", data=hists["node_features"].astype(np.float32))
            
            # 在保存前，对源项进行归一化处理 Q -> Q / (rho * c)
            normalized_source_terms = hists["source_terms"] / (rho * specific_heat)
            traj_group.create_dataset("source_terms", data=normalized_source_terms.astype(np.float32))

            traj_group.create_dataset("initial_condition", data=hists["node_features"][0].astype(np.float32))
            traj_group.create_dataset("time_points", data=time_points.astype(np.float32))
            
            # Create boundary info, populating dirichlet with empty datasets
            boundary_group = traj_group.create_group("boundary_info")
            dirichlet_group = boundary_group.create_group("dirichlet")
            dirichlet_group.create_dataset("indices", data=np.array([], dtype=np.int32))
            dirichlet_group.create_dataset("values", data=np.array([], dtype=np.float32))
            neumann_group = boundary_group.create_group("neumann")
            neumann_group.create_dataset("source_indices", data=np.array([], dtype=np.int32))
            neumann_group.create_dataset("target_indices", data=np.array([], dtype=np.int32))

    # The main visualization function might need adjustment if it's meant to run after
    # generating multiple trajectories. For now, we assume it visualizes the last one
    # if the HDF5 file is re-read, or we can disable it if it causes issues.
    # The current implementation will visualize trajectory_0 by default.
    # A small change to visualize the last trajectory could be:
    # f[f"trajectory_{NUM_TRAJECTORIES-1}/nodes"][:]
    # But for now, we leave it as is.

# =============================================================================
# --- 5. Visualization ---
# =============================================================================
def visualize_results():
    if MPI.COMM_WORLD.rank != 0: return
    print("\n--- 5. Starting visualization ---")
    with h5py.File(h5_filename, 'r') as f:
        nodes=f["trajectory_0/nodes"][:]; faces=f["trajectory_0/faces"][:]
        temp=f["trajectory_0/node_features"][:]; 
        src=f["trajectory_0/source_terms"][:]; time=f["trajectory_0/time_points"][:]
    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], faces)

    # For temperature, use percentile to avoid extreme peaks washing out the colormap
    temp_above_ambient = temp[temp > T_ambient + 1e-4] # Look at heated areas
    if temp_above_ambient.size > 0:
        vmax_temp = np.percentile(temp_above_ambient, 99.9)
    else:
        vmax_temp = T_ambient + 1.0 # Default if no significant heating
    plot_snapshot(triangulation, time, temp, "Temperature (K)", "hot", snapshots_vis_filename, vmin=T_ambient, vmax=vmax_temp)
    
    vmax_src = np.percentile(src[src > 1e-9], 99.9) if (src > 1e-9).any() else 1e-9
    plot_snapshot(triangulation, time, src, "Heat Source (W/m³)", "hot", source_vis_filename, vmin=0, vmax=vmax_src)

def plot_snapshot(tri, time, data, label, cmap, filename, vmin, vmax):
    if MPI.COMM_WORLD.rank != 0: return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle(f"Laser Heat Treatment {label.split(' ')[0]} Snapshots", fontsize=16)
    indices = np.linspace(0, len(time) - 1, 4, dtype=int)
    for i, t_idx in enumerate(indices):
        ax = axes.flatten()[i]
        mappable = ax.tripcolor(tri, data[t_idx, :, 0], cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        ax.set_title(f"Time: {time[t_idx]:.2f} s")
        ax.set_aspect('equal', adjustable='box')
        if i >= 2: ax.set_xlabel("x (m)"); 
        if i % 2 == 0: ax.set_ylabel("y (m)")
    fig.subplots_adjust(bottom=0.15, top=0.9)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal', label=label)
    plt.savefig(filename)
    plt.close()
    print(f"{label.split(' ')[0]} snapshots saved to: {filename}")

# =============================================================================
# --- Main Execution ---
# =============================================================================
if __name__ == "__main__":
    generate_dataset()
    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        visualize_results()
