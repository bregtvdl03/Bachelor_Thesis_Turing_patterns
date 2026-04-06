import matplotlib as mpl
import pyvista
import ufl
import numpy as np
import itertools

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, assign
import dolfinx.io.gmsh as gmshio
import gmsh

OUT_FILE = "out_cells/3d_spheres_refined.gif"
FPS = 10

#region ========== MODEL PARAMETERS ==========

m = 2
n = 1

Du      = 1.0       # Diffusion coef for u
Dv      = 40.0      # Diffusion coef for v
Pu      = 0.125     # Production coef for u
Pv      = 0.420     # Production coef for v
gamma   = 64.0      # Reaction scaling

uniform_steady_state_u = Pu + Pv
uniform_steady_state_v = (Pv / (Pu + Pv)**m) ** (1/n)

perturbation_strength = 0.1

def initial_condition_u(x):
    return uniform_steady_state_u + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return [uniform_steady_state_u] * x.shape[1]

def initial_condition_v(x):
    return uniform_steady_state_v + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return [uniform_steady_state_v] * x.shape[1]

t = 0.0
T = 50.0 / gamma
num_steps = 2048
dt = T / num_steps

#endregion

#region ========== DEFINING MESH AND FUNCTIONSPACE ==========

dim = 3
L = 32
half_L = L / 2
cell_size = 8
half_cell = cell_size / 2

gmsh.initialize()
gmsh.open("meshes/3d_spheres.msh")

model_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=dim)

gmsh.finalize()

msh = model_data.mesh

centers = list(itertools.product([-L/4, L/4], repeat=dim))

def on_holes(x):
    flags = np.zeros(x.shape[1], dtype=bool)
    for (cx, cy, cz) in centers:
        c = np.reshape([cx, cy, cz], (3, 1))
        dist = np.linalg.norm(x - c, ord=2, axis=0)
        flags |= np.isclose(dist, half_cell)
    return flags

facet_indices = mesh.locate_entities(msh, dim - 1, on_holes)
facet_markers = mesh.meshtags(msh, dim - 1, facet_indices, 10)

msh.topology.create_entities(dim - 1)
boundary_msh, boundary_msh_emap = mesh.create_submesh(msh, dim - 1, facet_indices)[:2]

ds      = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers)
dx      = ufl.Measure("dx", domain=msh)
ds_b    = ufl.Measure("dx", domain=boundary_msh)

V   = fem.functionspace(msh         , ("Lagrange", 1))
Vb  = fem.functionspace(boundary_msh, ("Lagrange", 1))

W = ufl.MixedFunctionSpace(V, Vb)

#endregion

#region ========== DEFINING FUNCTIONS ==========

# u_{n}
u_n = fem.Function(Vb)
u_n.name = "u_n"
u_n.interpolate(initial_condition_u)

# u_{n+1}
uh = fem.Function(Vb)
uh.name = "uh"
uh.interpolate(initial_condition_u)

# v_{n}
v_n = fem.Function(V)
v_n.name = "v_n"
v_n.interpolate(initial_condition_v)

# v_{n+1}
vh = fem.Function(V)
vh.name = "vh"
vh.interpolate(initial_condition_v)

v, u = ufl.TrialFunctions(W)
psi, phi = ufl.TestFunctions(W)

#endregion

#region ========== VARIATIONAL FORM ==========

# TODO: try fractional theta method: https://www.sciencedirect.com/science/article/pii/S0168874X15001377

a = u * phi * ds(10) \
    + dt * Du * ufl.dot(ufl.grad(u), ufl.grad(phi)) * ds(10) \
    + v * psi * dx \
    + dt * Dv * ufl.dot(ufl.grad(v), ufl.grad(psi)) * dx \
    + dt * gamma * (u) * phi * ds(10) \

L = u_n * phi * ds(10) \
    + v_n * psi * dx \
    + dt * gamma * (Pu) * phi * ds(10) \
    + dt * gamma * (Pv) * psi * ds(10) \
    + dt * gamma * (u_n * u_n * v_n) * phi * ds(10) \
    - dt * gamma * (u_n * u_n * v_n) * psi * ds(10) \

#endregion

#region ========== DEFINING SOLVERS ==========

bilinear_form = fem.form(ufl.extract_blocks(a), entity_maps=[boundary_msh_emap])
linear_form = fem.form(ufl.extract_blocks(L), entity_maps=[boundary_msh_emap])

A = assemble_matrix(bilinear_form)
A.assemble()
b = create_vector(fem.extract_function_spaces(linear_form))

solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

#endregion

#region ========== PLOTTING SETUP ==========

# Uncomment this for offscreen rendering
# pyvista.start_xvfb()

warpfactor = 0

grid_b = pyvista.UnstructuredGrid(*plot.vtk_mesh(Vb))
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

grid_b.point_data["uh"] = uh.x.array
grid.point_data["vh"] = vh.x.array
u_graph = grid_b.warp_by_scalar("uh", factor=warpfactor)
v_graph = grid.warp_by_scalar("vh", factor=warpfactor)

plotter = pyvista.Plotter()
plotter.open_gif(OUT_FILE, fps=FPS)
plotter.show_grid()
plotter.enable_parallel_projection()
# plotter.isometric_view()
# plotter.view_xy()
plotter.show_grid(
    font_size = 15,
    font_family = "times",
    xtitle = "x",
    ytitle = "y",
    ztitle = "z"
)

blues = mpl.colormaps.get_cmap("jet").resampled(64)
ylorrd = mpl.colormaps.get_cmap("YlOrRd").resampled(32)

plotter.add_mesh(
    u_graph,
    show_edges=False,
    lighting=False,
    cmap=blues,
    clim=[-0.1, 3],
    scalar_bar_args={
        "font_family": "times",
        "position_x": 0.2,
        "position_y": 0.9
    }
)

plotter.add_mesh(
    v_graph,
    opacity=0.1,
    show_edges=False,
    lighting=False,
    cmap=ylorrd,
    clim=[uniform_steady_state_v - perturbation_strength, uniform_steady_state_v + perturbation_strength],
    scalar_bar_args={
        "font_family": "times",
        "position_x": 0.2,
        "position_y": 0.82
    }
)

time_text = plotter.add_text(
    f"{str(int(t/T * 100))}\t/100 %",
    font_size=10,
    font="times"
)

plotter.add_text(
    "d = 40, a = 0.125, b = 0.420",
    font_size=10,
    font="times",
    position=(5, 5)
)

plotter.write_frame()

#endregion

#region ========== SOLVING ITERATIVELY ==========

for n in range(num_steps):
    t += dt
    progress = int(n/num_steps * 100)
    time_text.SetText(2, f"{str(progress)}\t/100 %")
    
    try:
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)
        x = create_vector([V, Vb])
        solver.solve(b, x)
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    except PETSc.Error as e:
        if e.ierr == 92:
            print("The required PETSc solver/preconditioner is not available. Exiting.")
            print(e)
            exit(0)
        else:
            raise e
    
    assign(x, [vh, uh])
    x.destroy()

    u_n.x.array[:] = uh.x.array
    v_n.x.array[:] = vh.x.array
    
    if n % 16 == 0:
        new_warped = grid_b.warp_by_scalar("uh", factor=warpfactor)
        u_graph.points[:, :] = new_warped.points
        u_graph.point_data["uh"][:] = uh.x.array

        new_warped = grid.warp_by_scalar("vh", factor=warpfactor)
        v_graph.points[:, :] = new_warped.points
        v_graph.point_data["vh"][:] = vh.x.array
        
        # minimum = np.min(uh.x.array)
        # maximum = np.max(uh.x.array)
        # print(F"MIN: {minimum}, MAX: {maximum}")

        plotter.write_frame()

plotter.close()

#endregion