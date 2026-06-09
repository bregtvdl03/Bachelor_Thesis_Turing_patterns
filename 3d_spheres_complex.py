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

from steady_states import get_steady_states
from datetime import timedelta
import time
import os


OUT_FILE = "out_cells/3d_spheres_complex.gif"
FPS = 10
WRITE_EVERY = 500

DOMAIN_TAG      = 100
MEMBRANE_TAG    = 10

#region ========== MODEL PARAMETERS ==========

m = 2
n = 1

Du = 1.0        # Diffusion coef for u
Dv = 50.0       # Diffusion coef for v
Pu = 0.1        # Production coef for u
Pv = 0.9        # Production coef for v
gamma = 64   # Reaction scaling
kappa = 50      # Additional scaling for w

Dw      = 1.0   # Diffusion coef for w
delta_u = 1.0
delta_w = 1.0
delta_v = 0.1
k_on    = 20.0
k_off   = k_on * delta_w - delta_w
alpha   = (m + n) * delta_w

steady_states = get_steady_states(Pu, Pv, delta_u, delta_v, delta_w, k_on, k_off, alpha)

uss_u = steady_states[0]
uss_v = steady_states[1]
uss_w = steady_states[2]

print(uss_u)
print(uss_w)
print(uss_v)

perturbation_strength = 1.0

def initial_condition_u(x):
    return uss_u + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return [uss_u] * x.shape[1]

def initial_condition_v(x):
    return uss_v + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return [uss_v] * x.shape[1]

t = 0.0
T = 100.0 / gamma
num_steps = 100000
dt = T / num_steps

#endregion

#region ========== DEFINING MESH AND FUNCTIONSPACE ==========

gmsh.initialize()

dim = 3
L = 32
half_L = L / 2
cell_size = 8
half_cell = cell_size / 2

centers = list(itertools.product([-L/4, L/4], repeat=dim))

gmsh.open("meshes/3d_spheres_coarse.msh")

model_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=dim)

gmsh.finalize()

msh = model_data.mesh

def on_holes(x):
    flags = np.zeros(x.shape[1], dtype=bool)
    for (cx, cy, cz) in centers:
        c = np.reshape([cx, cy, cz], (3, 1))
        dist = np.linalg.norm(x - c, ord=2, axis=0)
        flags |= np.isclose(dist, half_cell)
    return flags

facet_indices = mesh.locate_entities(msh, dim - 1, on_holes)
facet_markers = mesh.meshtags(msh, dim - 1, facet_indices, MEMBRANE_TAG)

msh.topology.create_entities(dim - 1)
boundary_msh, boundary_msh_emap = mesh.create_submesh(msh, dim - 1, facet_indices)[:2]

ds      = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers)
dx      = ufl.Measure("dx", domain=msh)
ds_b    = ufl.Measure("dx", domain=boundary_msh)

V   = fem.functionspace(msh         , ("Lagrange", 1))
Vb  = fem.functionspace(boundary_msh, ("Lagrange", 1))

W = ufl.MixedFunctionSpace(V, Vb, Vb)

#endregion

#region ========== DEFINING FUNCTIONS ==========

# u_{n}
u_n = fem.Function(Vb)
u_n.name = "u_n"
u_n.interpolate(initial_condition_v)

# u_{n+1}
uh = fem.Function(Vb)
uh.name = "uh"
uh.interpolate(initial_condition_v)

# w_{n}
w_n = fem.Function(Vb)
w_n.name = "w_n"
w_n.interpolate(initial_condition_v)

# w_{n+1}
wh = fem.Function(Vb)
wh.name = "wh"
wh.interpolate(initial_condition_v)

# v_{n}
v_n = fem.Function(V)
v_n.name = "v_n"
v_n.interpolate(initial_condition_v)

# v_{n+1}
vh = fem.Function(V)
vh.name = "vh"
vh.interpolate(initial_condition_v)

v, u, w = ufl.TrialFunctions(W)
psi, phi, chi = ufl.TestFunctions(W)

#endregion

#region ========== VARIATIONAL FORM ==========

# TODO: try fractional theta method: https://www.sciencedirect.com/science/article/pii/S0168874X15001377

a = u * phi * ds(MEMBRANE_TAG) \
    + w * chi * ds(MEMBRANE_TAG) \
    + v * psi * dx \
    + dt * Du * ufl.dot(ufl.grad(u), ufl.grad(phi)) * ds(MEMBRANE_TAG) \
    + dt * Dw * ufl.dot(ufl.grad(w), ufl.grad(chi)) * ds(MEMBRANE_TAG) \
    + dt * Dv * ufl.dot(ufl.grad(v), ufl.grad(psi)) * dx \
    + dt * gamma * (delta_u * u - (alpha + 2 * k_off) * w) * phi * ds(MEMBRANE_TAG) \
    + dt * kappa * gamma * (delta_w + k_off) * w * chi * ds(MEMBRANE_TAG) \
    - dt * gamma * k_off * w * psi * ds(MEMBRANE_TAG) \
    # + dt * gamma * u * phi * ds(MEMBRANE_TAG) \
    # + dt * kappa * gamma * delta_w * w * chi * ds(MEMBRANE_TAG) \
    # + dt * gamma * delta_v * v * psi * ufl.dx \

L = u_n * phi * ds(MEMBRANE_TAG) \
    + w_n * chi * ds(MEMBRANE_TAG) \
    + v_n * psi * dx \
    + dt * gamma * (Pu - 2 * k_on * u_n * u_n * v_n) * phi * ds(MEMBRANE_TAG) \
    + dt * gamma * (Pv - k_on * u_n * u_n * v_n) * psi * ds(MEMBRANE_TAG) \
    + dt * kappa * gamma * (k_on * u_n * u_n * v_n) * chi * ds(MEMBRANE_TAG) \
    # + dt * gamma * (Pu + alpha * w_n - 2 * k_on * u_n * u_n * v_n + 2 * k_off * w_n) * phi * ds(MEMBRANE_TAG) \
    # + dt * gamma * (Pv - k_on * u_n * u_n * v_n + k_off * w_n) * psi * ds(MEMBRANE_TAG) \
    # + dt * kappa * gamma * (k_on * u_n * u_n * v_n - k_off * w_n) * chi * ds(MEMBRANE_TAG) \

#endregion

#region ========== DEFINING SOLVERS ==========

bilinear_form   = fem.form(ufl.extract_blocks(a), entity_maps=[boundary_msh_emap])
linear_form     = fem.form(ufl.extract_blocks(L), entity_maps=[boundary_msh_emap])

A = assemble_matrix(bilinear_form)
A.assemble()
b = create_vector(fem.extract_function_spaces(linear_form))

# solver = PETSc.KSP().create(msh.comm)
# solver.setOperators(A)
# solver.setType(PETSc.KSP.Type.PREONLY)
# solver.getPC().setType(PETSc.PC.Type.LU)

solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.GMRES)

pc = solver.getPC()
pc.setType(PETSc.PC.Type.HYPRE)
pc.setHYPREType("boomeramg")

solver.setTolerances(
    rtol=1e-8,
    atol=1e-12,
    max_it=500,
)

#endregion

#region ========== PLOTTING SETUP ==========

# Uncomment this for offscreen rendering
# pyvista.start_xvfb()

warpfactor = 0

grid_u = pyvista.UnstructuredGrid(*plot.vtk_mesh(Vb))
grid_w = pyvista.UnstructuredGrid(*plot.vtk_mesh(Vb))
grid_v = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

grid_u.point_data["uh"] = uh.x.array
grid_w.point_data["wh"] = wh.x.array
grid_v.point_data["vh"] = vh.x.array
u_graph = grid_u.warp_by_scalar("uh", factor=warpfactor)
w_graph = grid_w.warp_by_scalar("wh", factor=warpfactor)
v_graph = grid_v.warp_by_scalar("vh", factor=warpfactor)

plotter = pyvista.Plotter()
plotter.open_gif(OUT_FILE, fps=FPS)
plotter.enable_parallel_projection()
plotter.isometric_view()
# plotter.view_xy()
plotter.show_grid(
    font_size = 15,
    font_family = "times",
    xtitle = "x",
    ytitle = "y",
    ztitle = "z"
)

plotter.add_mesh(
    u_graph,
    show_edges=False,
    lighting=False,
    cmap="Blues",
    clim=[uss_u - perturbation_strength, uss_u + perturbation_strength],
    scalar_bar_args={
        "font_family": "times",
        "position_x": 0.2,
        "position_y": 0.9
    }
)

plotter.add_mesh(
    w_graph,
    show_edges=False,
    lighting=False,
    cmap="Reds",
    clim=[uss_u - perturbation_strength, uss_u + perturbation_strength],
    scalar_bar_args={
        "font_family": "times",
        "position_x": 0.2,
        "position_y": 0.9
    }
)

plotter.add_mesh(
    v_graph,
    opacity=0.3,
    show_edges=False,
    lighting=False,
    cmap="Greens",
    clim=[uss_v - perturbation_strength, uss_v + perturbation_strength],
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
        x = create_vector([V, Vb, Vb])
        solver.solve(b, x)
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    except PETSc.Error as e:
        if e.ierr == 92:
            print("The required PETSc solver/preconditioner is not available. Exiting.")
            print(e)
            exit(0)
        else:
            raise e
    
    assign(x, [vh, uh, wh])
    x.destroy()
    
    u_n.x.array[:] = uh.x.array
    w_n.x.array[:] = wh.x.array
    v_n.x.array[:] = vh.x.array
    
    if n % WRITE_EVERY == 0:
        # new_warped = grid_u.warp_by_scalar("uh", factor=warpfactor)
        # u_graph.points[:, :] = new_warped.points
        u_graph.point_data["uh"][:] = uh.x.array

        # new_warped = grid_w.warp_by_scalar("wh", factor=warpfactor)
        # w_graph.points[:, :] = new_warped.points
        w_graph.point_data["wh"][:] = wh.x.array

        # new_warped = grid_v.warp_by_scalar("vh", factor=warpfactor)
        # v_graph.points[:, :] = new_warped.points
        v_graph.point_data["vh"][:] = vh.x.array

        plotter.write_frame()
        
    if n == WRITE_EVERY - 1:
        start = time.monotonic()
    if n >= WRITE_EVERY and n % WRITE_EVERY == 0:
        os.system("clear")
        eta = timedelta(seconds=round((time.monotonic() - start) / (n - WRITE_EVERY + 1) * num_steps - time.monotonic() + start))
        print(f"ETA: {eta}")
        
        print(f"u_n: {min(u_n.x.array)} - {max(u_n.x.array)}")
        print(f"w_n: {min(w_n.x.array)} - {max(w_n.x.array)}")
        print(f"v_n: {min(v_n.x.array)} - {max(v_n.x.array)}")
          
plotter.close()

#endregion