import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

import basix.ufl
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector

t = 0
T = 1.0
num_steps = 128
dt = T / num_steps

Du = 1.0 # Diffusion coef for u
Dv = 30.0 # Diffusion coef for v
Pu = 0.2 # Production coef for u
Pv = 0.66 # Production coef for v

nx, ny = 128, 128

domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=[[-2.0, -2.0], [2.0, 2.0]],
    n=[nx, ny],
    cell_type=mesh.CellType.triangle
)

el_u = basix.ufl.element("Lagrange", basix.CellType.triangle, 1)
el_v = basix.ufl.element("Lagrange", basix.CellType.triangle, 1)
el_mixed = basix.ufl.mixed_element([el_u, el_v])

V = fem.functionspace(domain, el_mixed)

def initial_condition_u(x):
    return Pu + Pv + 0.2 * (np.random.rand(x.shape[1]) - 0.5)
    # return [Pu + Pv] * x.shape[1]

def initial_condition_v(x):
    return Pv / (Pu + Pv)**2 + 0.2 * (np.random.rand(x.shape[1]) - 0.5)
    # return [Pv / (Pu + Pv)**2] * x.shape[1]

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

a = u * phi * ufl.dx \
    + dt * Du * ufl.dot(ufl.grad(u), ufl.grad(phi)) * ufl.dx \
    + v * psi * ufl.dx \
    + dt * Dv * ufl.dot(ufl.grad(v), ufl.grad(psi)) * ufl.dx

L = (u_n + dt * (Pu - u_n + u_n * u_n * v_n)) * phi * ufl.dx \
    + (v_n + dt * (Pv - u_n * u_n * v_n)) * psi * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(bilinear_form)
A.assemble()
b = create_vector(linear_form)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

pyvista.start_xvfb()

V0, mapu = V.sub(0).collapse()
V1, mapv = V.sub(1).collapse()

u_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V0))
v_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V1))

plotter = pyvista.Plotter()
plotter.open_gif("out_schnakenberg/schakenberg.gif", fps=10)
plotter.show_grid()

u_grid.point_data["uh"] = u_n.x.array[mapu]
v_grid.point_data["vh"] = v_n.x.array[mapv]
u_graph = u_grid.warp_by_scalar("uh", factor=1)
v_graph = v_grid.warp_by_scalar("vh", factor=1)

blues = mpl.colormaps.get_cmap("Blues").resampled(20)
ylorrd = mpl.colormaps.get_cmap("YlOrRd").resampled(20)

plotter.add_mesh(
    u_graph,
    show_edges=False,
    lighting=False,
    cmap=blues,
    clim=[Pu + Pv - 0.2, Pu + Pv + 0.2]
)

plotter.add_mesh(
    v_graph,
    show_edges=False,
    lighting=False,
    cmap=ylorrd,
    clim=[Pv / (Pu + Pv)**2 - 0.2, Pv / (Pu + Pv)**2 + 0.2],
    opacity=0.8
)

for n in range(num_steps):
    t += dt
    
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