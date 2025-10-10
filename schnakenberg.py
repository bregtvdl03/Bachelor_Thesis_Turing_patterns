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
FPS = 40

Du = 1.0    # Diffusion coef for u
Dv = 30.0   # Diffusion coef for v
Pu = 0.2    # Production coef for u
Pv = 0.8    # Production coef for v
gamma = 1.0 # Reaction scaling

uniform_steady_state_u = Pu + Pv
uniform_steady_state_v = Pv / (Pu + Pv)**2

perturbation_strength = 0.2

def initial_condition_u(x):
    return uniform_steady_state_u + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return [uniform_steady_state_u] * x.shape[1]

def initial_condition_v(x):
    return uniform_steady_state_v + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return [uniform_steady_state_v] * x.shape[1]

t = 0.0
T = 50.0
num_steps = 128
dt = T / num_steps

nx, ny = 128, 128

domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=[[-128.0, -128.0], [128.0, 128.0]],
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

# BACKWARDS EULER
a = u * phi * ufl.dx \
    + dt * Du * ufl.dot(ufl.grad(u), ufl.grad(phi)) * ufl.dx \
    + v * psi * ufl.dx \
    + dt * Dv * ufl.dot(ufl.grad(v), ufl.grad(psi)) * ufl.dx

L = (u_n + dt * gamma * (Pu - u_n + u_n * u_n * v_n)) * phi * ufl.dx \
    + (v_n + dt * gamma * (Pv - u_n * u_n * v_n)) * psi * ufl.dx


# SEMI-IMPLICIT REACTION
# a = u * phi * ufl.dx \
#     + dt * Du * ufl.dot(ufl.grad(u), ufl.grad(phi)) * ufl.dx \
#     + v * psi * ufl.dx \
#     + dt * Dv * ufl.dot(ufl.grad(v), ufl.grad(psi)) * ufl.dx \
#     - dt * gamma * u * u_n * v_n * phi * ufl.dx \
#     + dt * gamma * u * u_n * v_n * psi * ufl.dx

# L = (u_n + dt * gamma * (Pu - u_n)) * phi * ufl.dx \
#     + (v_n + dt * gamma * Pv) * psi * ufl.dx    


# CRANK NICOLSON
# a = u * phi * ufl.dx \
#     - dt * gamma / 2 * (Pu - u + u * u * v) * phi * ufl.dx \
#     + dt * Du / 2 * ufl.inner(ufl.grad(u), ufl.grad(phi)) * ufl.dx \
#     + v * psi * ufl.dx \
#     - dt * gamma / 2 * (Pv - u * u * v) * psi * ufl.dx \
#     + dt * Dv / 2 * ufl.inner(ufl.grad(v), ufl.grad(psi)) * ufl.dx

# L = u_n * phi * ufl.dx \
#     + dt * gamma / 2 * (Pu - u_n + u_n * u_n * v_n) * phi * ufl.dx \
#     - dt * Du / 2 * ufl.inner(ufl.grad(u_n), ufl.grad(phi)) * ufl.dx \
#     + v_n * psi * ufl.dx \
#     + dt * gamma / 2 * (Pv - u_n * u_n * v_n) * psi * ufl.dx \
#     - dt * Dv / 2 * ufl.inner(ufl.grad(v_n), ufl.grad(psi)) * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(bilinear_form)
A.assemble()
b = create_vector(linear_form)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# pyvista.start_xvfb()

V0, mapu = V.sub(0).collapse()
V1, mapv = V.sub(1).collapse()

u_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V0))
v_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V1))

plotter = pyvista.Plotter()
plotter.open_gif(OUT_FILE, fps=FPS)
plotter.enable_parallel_projection()
plotter.isometric_view()
plotter.show_grid(
    font_size = 15,
    font_family = "times",
    xtitle = "x",
    ytitle = "y",
    ztitle = "z"
)

u_grid.point_data["uh"] = u_n.x.array[mapu]
v_grid.point_data["vh"] = v_n.x.array[mapv]
u_graph = u_grid.warp_by_scalar("uh", factor=1)
v_graph = v_grid.warp_by_scalar("vh", factor=1)

blues = mpl.colormaps.get_cmap("Blues").resampled(32)
ylorrd = mpl.colormaps.get_cmap("YlOrRd").resampled(32)
colorwidth = 0.005

plotter.add_mesh(
    u_graph,
    show_edges=False,
    lighting=False,
    opacity=0.8,
    cmap=blues,
    clim=[uniform_steady_state_u - colorwidth, uniform_steady_state_u + colorwidth],
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
    clim=[uniform_steady_state_v - colorwidth, uniform_steady_state_v + colorwidth],
    scalar_bar_args={
        "font_family": "times",
        "position_x": 0.2,
        "position_y": 0.82
    }
)

time_text = plotter.add_text(
    "t = 0.00",
    font_size=10,
    font="times"
)

for n in range(num_steps):
    t += dt
    time_text.SetText(2, f"t = {t:.3f}")
    print(t)
    
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    
    solver.solve(b, uv_sol.x.petsc_vec)
    uv_sol.x.scatter_forward()
    
    u_h, v_h = uv_sol.split()
    
    u_n.x.array[mapu] = u_h.x.array[mapu]
    v_n.x.array[mapv] = v_h.x.array[mapv]
    
    u_graph_new = u_grid.warp_by_scalar("uh", factor=1)
    v_graph_new = v_grid.warp_by_scalar("vh", factor=1)
    u_graph.points[:, :] = u_graph_new.points
    v_graph.points[:, :] = v_graph_new.points
    u_graph.point_data["uh"][:] = u_h.x.array[mapu]
    v_graph.point_data["vh"][:] = v_h.x.array[mapv]
    plotter.write_frame()

plotter.close()