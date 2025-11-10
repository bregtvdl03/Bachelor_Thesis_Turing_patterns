import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

import basix.ufl
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector

OUT_FILE = "out_schnakenberg/schakenberg.gif"
OUT_SCREENSHOT = "out_schnakenberg/schnakenberg_profile.jpg"
FPS = 10

#region ========== PARAMETERS ==========

m = 2
n = 1

Du = 1.0    # Diffusion coef for u
Dv = 40.0   # Diffusion coef for v
Pu = 0.125    # Production coef for u
Pv = 0.420    # Production coef for v
gamma = 128.0**2 # Reaction scaling

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
num_steps = 1024
dt = T / num_steps

nx, ny = 128, 128

domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=[[-1.0, -1.0], [1.0, 1.0]],
    n=[nx, ny],
    cell_type=mesh.CellType.triangle
)

# domain = mesh.create_unit_square(
#     comm=MPI.COMM_WORLD,
#     nx=nx,
#     ny=ny,
#     cell_type=mesh.CellType.triangle
# )

el_u = basix.ufl.element("Lagrange", basix.CellType.triangle, 1)
el_v = basix.ufl.element("Lagrange", basix.CellType.triangle, 1)
el_mixed = basix.ufl.mixed_element([el_u, el_v])

V = fem.functionspace(domain, el_mixed)

#endregion

#region ========== DEFINING FUNCTIONS ==========

# u_{n}
u_n, v_n = fem.Function(V).split()
u_n.name = "u_n"
u_n.interpolate(initial_condition_u)

# v_{n}
v_n.name = "v_n"
v_n.interpolate(initial_condition_v)

# u_{n + 1} and v_{n + 1}
uv_sol = fem.Function(V)
u_sol, v_sol = uv_sol.split()

(u, v) = ufl.TrialFunctions(V)
(phi, psi) = ufl.TestFunctions(V)

#endregion

#region ========== VARIATIONAL FORM ==========

# IMEX
a = u * phi * ufl.dx \
    + dt * Du * ufl.dot(ufl.grad(u), ufl.grad(phi)) * ufl.dx \
    + v * psi * ufl.dx \
    + dt * Dv * ufl.dot(ufl.grad(v), ufl.grad(psi)) * ufl.dx

L = (u_n + dt * gamma * (Pu - u_n + u_n * u_n * v_n)) * phi * ufl.dx \
    + (v_n + dt * gamma * (Pv - u_n * u_n * v_n)) * psi * ufl.dx

#endregion

#region ========== SETUP PROBLEM AND SOLVERS ==========

bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(bilinear_form)
A.assemble()
b = create_vector(linear_form)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

#endregion

#region ========== SETUP PLOTTING ==========

# Uncomment for offscreen rendering
# pyvista.start_xvfb()

V0, mapu = V.sub(0).collapse()
V1, mapv = V.sub(1).collapse()

u_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V0))
v_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V1))

plotter = pyvista.Plotter()
plotter.open_gif(OUT_FILE, fps=FPS)
plotter.enable_parallel_projection()
plotter.isometric_view()
plotter.view_xy()
plotter.camera.zoom(0.2)
plotter.show_grid(
    font_size = 20,
    font_family = "times",
    xtitle = "x",
    ytitle = "y",
    ztitle = "z"
)

warp_factor = 0.1
u_grid.point_data["uh"] = u_n.x.array[mapu]
v_grid.point_data["vh"] = v_n.x.array[mapv]
u_graph = u_grid.warp_by_scalar("uh", factor=warp_factor)
v_graph = v_grid.warp_by_scalar("vh", factor=warp_factor)

blues = mpl.colormaps.get_cmap("Blues").resampled(64)
ylorrd = mpl.colormaps.get_cmap("YlOrRd").resampled(64)
colorwidth = 0.1

plotter.add_mesh(
    u_graph,
    show_edges=False,
    lighting=False,
    opacity=0.9,
    cmap=blues,
    clim=[0, 5 * uniform_steady_state_u],
    scalar_bar_args={
        "font_family": "times",
        "position_x": 0.2,
        "position_y": 0.9
    }
)

plotter.add_mesh(
    v_graph,
    show_edges=False,
    lighting=False,
    cmap=ylorrd,
    clim=[0.5 * uniform_steady_state_v, 2 * uniform_steady_state_v],
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

#endregion

#region ========== SOLVE PROBLEM AND PLOT ==========

for n in range(num_steps):
    t += dt
    
    progress = int(t/T * 100)
    time_text.SetText(2, f"{str(progress)}\t/100 %")
    print(progress)
    
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    
    solver.solve(b, uv_sol.x.petsc_vec)
    uv_sol.x.scatter_forward()
    
    u_h, v_h = uv_sol.split()
    
    u_n.x.array[mapu] = u_h.x.array[mapu]
    v_n.x.array[mapv] = v_h.x.array[mapv]
    
    if n % 16 == 0:
        u_graph_new = u_grid.warp_by_scalar("uh", factor=warp_factor)
        v_graph_new = v_grid.warp_by_scalar("vh", factor=warp_factor)
        u_graph.points[:, :] = u_graph_new.points
        v_graph.points[:, :] = v_graph_new.points
        u_graph.point_data["uh"][:] = u_h.x.array[mapu]
        v_graph.point_data["vh"][:] = v_h.x.array[mapv]
        plotter.write_frame()

plotter.view_xz()
plotter.screenshot(OUT_SCREENSHOT, scale = 3)

plotter.close()

#endregion