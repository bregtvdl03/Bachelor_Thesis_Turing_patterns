import pyvista
import ufl
import numpy as np

import time
import os
from datetime import timedelta
from steady_states import get_steady_states

from petsc4py import PETSc
from mpi4py import MPI

import basix.ufl
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector

OUT_FILE = "out_schnakenberg/schakenberg_complex_2.gif"
OUT_SCREENSHOT = None   # "out_schnakenberg/schnakenberg_profile.jpg"
FPS = 10

#region ========== PARAMETERS ==========

m = 2
n = 1

Du = 1.0        # Diffusion coef for u
Dv = 10.0       # Diffusion coef for v
Pu = 0.1        # Production coef for u
Pv = 0.9        # Production coef for v
gamma = 32.0**2 # Reaction scaling
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

perturbation_strength = 0.1

def initial_condition_u(x):
    return uss_u + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return (np.exp(-400 * (x[0]**2 + x[1]**2)) + np.exp(-400 * ((x[0] - 0.5)**2 + (x[1] + 0.3)**2)) + np.exp(-400 * ((x[0] + 0.2)**2 + (x[1] - 0.7)**2)) + np.exp(-400 * ((x[0] + 0.3)**2 + (x[1] + 0.8)**2))) + uss_u
    # return [uss_u] * x.shape[1]

def initial_condition_w(x):
    return uss_w + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return [uss_w] * x.shape[1]

def initial_condition_v(x):
    return uss_v + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return (np.exp(-400 * (x[0]**2 + x[1]**2)) + np.exp(-400 * ((x[0] - 0.5)**2 + (x[1] + 0.3)**2)) + np.exp(-400 * ((x[0] + 0.2)**2 + (x[1] - 0.7)**2)) + np.exp(-400 * ((x[0] + 0.3)**2 + (x[1] + 0.8)**2))) + uss_v
    # return [uss_v] * x.shape[1]

t = 0.0
T = 100.0 / gamma
num_steps = 64000
dt = T / num_steps

WRITE_EVERY = 512

nx, ny = 128, 128

domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=[[-1.0, -1.0], [1.0, 1.0]],
    n=[nx, ny],
    cell_type=mesh.CellType.triangle
)

el_u = basix.ufl.element("Lagrange", basix.CellType.triangle, 1)
el_w = basix.ufl.element("Lagrange", basix.CellType.triangle, 1)
el_v = basix.ufl.element("Lagrange", basix.CellType.triangle, 1)
el_mixed = basix.ufl.mixed_element([el_u, el_w, el_v])

V = fem.functionspace(domain, el_mixed)

#endregion

#region ========== DEFINING FUNCTIONS ==========

# u_{n}
u_n, w_n, v_n = fem.Function(V).split()
u_n.name = "u_n"
u_n.interpolate(initial_condition_u)

# w_{n}
w_n.name = "w_n"
w_n.interpolate(initial_condition_w)

# v_{n}
v_n.name = "v_n"
v_n.interpolate(initial_condition_v)

# u_{n + 1} and v_{n + 1}
uwv_sol = fem.Function(V)
u_sol, w_sol, v_sol = uwv_sol.split()

(u, w, v) = ufl.TrialFunctions(V)
(phi, chi, psi) = ufl.TestFunctions(V)

#endregion

#region ========== VARIATIONAL FORM ==========

# IMEX
a = u * phi * ufl.dx \
    + w * chi * ufl.dx \
    + v * psi * ufl.dx \
    + dt * Du * ufl.dot(ufl.grad(u), ufl.grad(phi)) * ufl.dx \
    + dt * Dw * ufl.dot(ufl.grad(w), ufl.grad(chi)) * ufl.dx \
    + dt * Dv * ufl.dot(ufl.grad(v), ufl.grad(psi)) * ufl.dx \
    + dt * gamma * u * phi * ufl.dx \
    + dt * kappa * gamma * delta_w * w * chi * ufl.dx \
    + dt * gamma * delta_v * v * psi * ufl.dx \

L = (u_n + dt * gamma * (Pu + alpha * w_n - 2 * k_on * u_n * u_n * v_n + 2 * k_off * w_n)) * phi * ufl.dx \
    + (w_n + dt * kappa * gamma * (k_on * u_n * u_n * v_n - k_off * w_n)) * chi * ufl.dx \
    + (v_n + dt * gamma * (Pv - k_on * u_n * u_n * v_n + k_off * w_n)) * psi * ufl.dx

#endregion

#region ========== SETUP PROBLEM AND SOLVERS ==========

bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(bilinear_form)
A.assemble()
b = create_vector(fem.extract_function_spaces(linear_form))

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

#endregion

#region ========== SETUP PLOTTING ==========

# Uncomment for offscreen rendering
# pyvista.start_xvfb()

V0, mapu = V.sub(0).collapse()
V1, mapw = V.sub(1).collapse()
V2, mapv = V.sub(2).collapse()

u_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V0))
w_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V1))
v_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V2))

plotter = pyvista.Plotter()
plotter.open_gif(OUT_FILE, fps=FPS)
plotter.enable_parallel_projection()
plotter.isometric_view()
# plotter.view_xy()
# plotter.camera.zoom(0.2)
plotter.show_grid(
    font_size = 20,
    font_family = "times",
    xtitle = "x",
    ytitle = "y",
    ztitle = "z"
)

warp_factor = 0.1

u_grid.point_data["uh"] = u_n.x.array[mapu]
w_grid.point_data["wh"] = u_n.x.array[mapw]
v_grid.point_data["vh"] = v_n.x.array[mapv]
u_graph = u_grid.warp_by_scalar("uh", factor=warp_factor)
w_graph = w_grid.warp_by_scalar("wh", factor=warp_factor)
v_graph = v_grid.warp_by_scalar("vh", factor=warp_factor)

colorwidth = 0.3

plotter.add_mesh(
    u_graph,
    show_edges=False,
    lighting=False,
    opacity=1.0,
    cmap="Blues",
    clim=[uss_u - colorwidth, uss_u + colorwidth],
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
    clim=[uss_w - colorwidth, uss_w + colorwidth],
    scalar_bar_args={
        "font_family": "times",
        "position_x": 0.2,
        "position_y": 0.82
    }
)

plotter.add_mesh(
    v_graph,
    show_edges=False,
    lighting=False,
    cmap="Greens",
    clim=[uss_v - colorwidth, uss_v + colorwidth],
    scalar_bar_args={
        "font_family": "times",
        "position_x": 0.2,
        "position_y": 0.74
    }
)

time_text = plotter.add_text(
    f"{str(int(t/T * 100))}\t/100 %",
    font_size=14,
    font="times",
)

#endregion

#region ========== SOLVE PROBLEM AND PLOT ==========

start = time.monotonic()

for n in range(num_steps):
    t += dt
    
    progress = int(t/T * 100)
    time_text.SetText(2, f"{str(progress)}\t/100 %")
    
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    
    solver.solve(b, uwv_sol.x.petsc_vec)
    uwv_sol.x.scatter_forward()
    
    u_h, w_h, v_h = uwv_sol.split()
    
    u_n.x.array[mapu] = u_h.x.array[mapu]
    w_n.x.array[mapw] = w_h.x.array[mapw]
    v_n.x.array[mapv] = v_h.x.array[mapv]
    
    if n % WRITE_EVERY == 0:
        u_graph_new = u_grid.warp_by_scalar("uh", factor=warp_factor)
        w_graph_new = w_grid.warp_by_scalar("wh", factor=warp_factor)
        v_graph_new = v_grid.warp_by_scalar("vh", factor=warp_factor)
        u_graph.points[:, :] = u_graph_new.points
        w_graph.points[:, :] = w_graph_new.points
        v_graph.points[:, :] = v_graph_new.points
        u_graph.point_data["uh"][:] = u_h.x.array[mapu]
        w_graph.point_data["wh"][:] = w_h.x.array[mapw]
        v_graph.point_data["vh"][:] = v_h.x.array[mapv]
        plotter.write_frame()
        
        os.system("clear")
        eta = timedelta(seconds=round((time.monotonic() - start) / (n + 1) * num_steps - time.monotonic() + start))
        print(f"ETA: {eta}")
        
        print(f"u_n: {min(u_n.x.array[mapu])} - {max(u_n.x.array[mapu])}")
        print(f"w_n: {min(w_n.x.array[mapw])} - {max(w_n.x.array[mapw])}")
        print(f"v_n: {min(v_n.x.array[mapv])} - {max(v_n.x.array[mapv])}")

if OUT_SCREENSHOT:
    plotter.view_xz()
    plotter.screenshot(OUT_SCREENSHOT, scale = 3)

plotter.close()

#endregion