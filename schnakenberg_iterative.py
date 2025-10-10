import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector

OUT_FILE = "out_schnakenberg/schakenberg_iterative.gif"
FPS = 40

Du = 1.0    # Diffusion coef for u
Dv = 30.0   # Diffusion coef for v
Pu = 0.2    # Production coef for u
Pv = 0.8    # Production coef for v
gamma = 1000.0 # Reaction scaling

uniform_steady_state_u = Pu + Pv
uniform_steady_state_v = Pv / (Pu + Pv)**2

perturbation_strength = 0.2

def initial_condition_u(x):
    return uniform_steady_state_u + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return [uniform_steady_state_u] * x.shape[1]

def initial_condition_v(x):
    return uniform_steady_state_v + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return [uniform_steady_state_v] * x.shape[1]

t = 0
T = 50.0
num_steps = 999999
dt = T / num_steps

nx, ny = 128, 128

# domain = mesh.create_rectangle(
#     comm=MPI.COMM_WORLD,
#     points=[[-128.0, -128.0], [128.0, 128.0]],
#     n=[nx, ny],
#     cell_type=mesh.CellType.triangle
# )

domain = mesh.create_unit_square(
    comm=MPI.COMM_WORLD,
    nx=nx,
    ny=ny,
    cell_type=mesh.CellType.triangle
)

V = fem.functionspace(domain, ("Lagrange", 1))

#region Defining functions

# u_{n}
u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition_u)

# u_{n+1}
uh = fem.Function(V)
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

u = ufl.TrialFunction(V)
phi = ufl.TestFunction(V)

v = ufl.TrialFunction(V)
psi = ufl.TestFunction(V)

#endregion

#region Variational form

a_u = u * phi * ufl.dx \
    + dt * Du * ufl.dot(ufl.grad(u), ufl.grad(phi)) * ufl.dx

L_u = (u_n + dt * gamma * (Pu - u_n + u_n * u_n * v_n)) * phi * ufl.dx

a_v = v * psi * ufl.dx \
    + dt * Dv * ufl.dot(ufl.grad(v), ufl.grad(psi)) * ufl.dx

L_v = (v_n + dt * gamma * (Pv - uh * uh * v_n)) * psi * ufl.dx

#endregion

#region Defining solvers

bilinear_form_u = fem.form(a_u)
linear_form_u = fem.form(L_u)

A_u = assemble_matrix(bilinear_form_u)
A_u.assemble()
b_u = create_vector(linear_form_u)

solver_u = PETSc.KSP().create(domain.comm)
solver_u.setOperators(A_u)
solver_u.setType(PETSc.KSP.Type.PREONLY)
solver_u.getPC().setType(PETSc.PC.Type.LU)

bilinear_form_v = fem.form(a_v)
linear_form_v = fem.form(L_v)

A_v = assemble_matrix(bilinear_form_v)
A_v.assemble()
b_v = create_vector(linear_form_v)

solver_v = PETSc.KSP().create(domain.comm)
solver_v.setOperators(A_v)
solver_v.setType(PETSc.KSP.Type.PREONLY)
solver_v.getPC().setType(PETSc.PC.Type.LU)

#endregion

# pyvista.start_xvfb()

grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

grid.point_data["uh"] = uh.x.array
grid.point_data["vh"] = vh.x.array
u_graph = grid.warp_by_scalar("uh", factor=1)
v_graph = grid.warp_by_scalar("vh", factor=1)

#region Plotting setup

plotter = pyvista.Plotter()
plotter.open_gif(OUT_FILE, fps=FPS)
plotter.show_grid()
plotter.enable_parallel_projection()
plotter.isometric_view()
plotter.show_grid(
    font_size = 15,
    font_family = "times",
    xtitle = "x",
    ytitle = "y",
    ztitle = "z"
)

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

#endregion

#region Solving iteratively

for n in range(num_steps):
    t += dt
    time_text.SetText(2, f"t = {t:.3f}")
    print(t)
    
    # Update and solve u
    with b_u.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_u, linear_form_u)
    
    solver_u.solve(b_u, uh.x.petsc_vec)
    uh.x.scatter_forward()
    
    # Update and solve v
    with b_v.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_v, linear_form_v)
    
    solver_v.solve(b_v, vh.x.petsc_vec)
    vh.x.scatter_forward()

    # Updating and plotting
    u_n.x.array[:] = uh.x.array
    v_n.x.array[:] = vh.x.array
    
    new_warped = grid.warp_by_scalar("uh", factor=1)
    u_graph.points[:, :] = new_warped.points
    u_graph.point_data["uh"][:] = uh.x.array
    
    new_warped = grid.warp_by_scalar("vh", factor=1)
    v_graph.points[:, :] = new_warped.points
    v_graph.point_data["vh"][:] = vh.x.array
    
    plotter.write_frame()

plotter.close()

#endregion