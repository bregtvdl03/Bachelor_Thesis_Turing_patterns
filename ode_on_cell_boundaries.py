import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

import basix.ufl
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector
from dolfinx.io import gmshio
import gmsh

OUT_FILE = "out_cells/test.gif"
FPS = 1

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

perturbation_strength = 0.01

def initial_condition_u(x):
    # return uniform_steady_state_u + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    return [uniform_steady_state_u] * x.shape[1]

def initial_condition_v(x):
    # return uniform_steady_state_v + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    return [uniform_steady_state_v] * x.shape[1]

t = 0.0
T = 1.0 / gamma
num_steps = 1024
dt = T / num_steps

#endregion

#region ========== DEFINING MESH AND FUNCTIONSPACE ==========

gmsh.initialize()

L = 2.0
half_L = L / 2
cell_size = 0.5
half_cell = cell_size / 2

gmsh.model.add("square_with_holes")

outer = gmsh.model.occ.addRectangle(-half_L, -half_L, 0, L, L)

holes = []
centers = [(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]
for cx, cy in centers:
    holes.append(gmsh.model.occ.addRectangle(cx - half_cell, cy - half_cell, 0, cell_size, cell_size))

main_domain, _ = gmsh.model.occ.cut([(2, outer)], [(2, h) for h in holes])

gmsh.model.occ.synchronize()

boundaries = gmsh.model.getBoundary(main_domain, oriented=False, recursive=False)

outer_boundary_lines = []
hole_boundaries = []

for (loop_dim, loop_tag) in boundaries:
    bb = gmsh.model.getBoundingBox(loop_dim, loop_tag)
    xmin, ymin, zmin, xmax, ymax, zmax = bb
    area = (xmax - xmin) * (ymax - ymin)
    if area > 4e-7:
        outer_boundary_lines.append(loop_tag)
    else:
        hole_boundaries.append(loop_tag)

gmsh.model.addPhysicalGroup(2, [main_domain[0][1]], tag=100)
gmsh.model.setPhysicalName(2, 100, "main_domain")

gmsh.model.addPhysicalGroup(1, outer_boundary_lines, tag=1)
gmsh.model.setPhysicalName(1, 1, "outer_boundary")

gmsh.model.addPhysicalGroup(1, hole_boundaries, tag=10)
gmsh.model.setPhysicalName(1, 10, "holes")

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.01)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01)
gmsh.model.mesh.generate(2)

msh, cell_markers, facet_markers = gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=2
)

gmsh.finalize()

ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers)

V = fem.functionspace(msh, ("Lagrange", 1))

#endregion

#region ========== DEFINING FUNCTIONS ==========

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

#region ========== VARIATIONAL FORM ==========

a_u = u * phi * ufl.dx \
    + dt * Du * ufl.dot(ufl.grad(u), ufl.grad(phi)) * ufl.dx

# L_u = u_n * phi * ufl.dx + (dt * gamma * (Pu - u_n + u_n * u_n * v_n)) * phi * ds(10)
L_u = u_n * phi * ufl.dx + (dt * gamma * Pu) * phi * ds(10)

a_v = v * psi * ufl.dx \
    + dt * Dv * ufl.dot(ufl.grad(v), ufl.grad(psi)) * ufl.dx

# L_v = v_n * psi * ufl.dx + (dt * gamma * (Pv - u_n * u_n * v_n)) * psi * ds(10)
L_v = v_n * psi * ufl.dx

#endregion

#region ========== DEFINING SOLVERS ==========

bilinear_form_u = fem.form(a_u)
linear_form_u = fem.form(L_u)

A_u = assemble_matrix(bilinear_form_u)
A_u.assemble()
b_u = create_vector(linear_form_u)

solver_u = PETSc.KSP().create(msh.comm)
solver_u.setOperators(A_u)
solver_u.setType(PETSc.KSP.Type.PREONLY)
solver_u.getPC().setType(PETSc.PC.Type.LU)

bilinear_form_v = fem.form(a_v)
linear_form_v = fem.form(L_v)

A_v = assemble_matrix(bilinear_form_v)
A_v.assemble()
b_v = create_vector(linear_form_v)

solver_v = PETSc.KSP().create(msh.comm)
solver_v.setOperators(A_v)
solver_v.setType(PETSc.KSP.Type.PREONLY)
solver_v.getPC().setType(PETSc.PC.Type.LU)

#endregion

# Uncomment this for offscreen rendering
# pyvista.start_xvfb()

grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

grid.point_data["uh"] = uh.x.array
grid.point_data["vh"] = vh.x.array
u_graph = grid.warp_by_scalar("uh", factor=1)
v_graph = grid.warp_by_scalar("vh", factor=1)

#region ========== PLOTTING SETUP ==========

plotter = pyvista.Plotter()
plotter.open_gif(OUT_FILE, fps=FPS)
plotter.show_grid()
plotter.enable_parallel_projection()
plotter.isometric_view()
# plotter.view_xy()
plotter.view_xz()
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
    clim=[0, 1],
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
    clim=[0, 1],
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

#region ========== SOLVING ITERATIVELY ==========

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
    
    if n % 1 == 0:
        new_warped = grid.warp_by_scalar("uh", factor=1)
        u_graph.points[:, :] = new_warped.points
        u_graph.point_data["uh"][:] = uh.x.array

        new_warped = grid.warp_by_scalar("vh", factor=1)
        v_graph.points[:, :] = new_warped.points
        v_graph.point_data["vh"][:] = vh.x.array

        plotter.write_frame()

plotter.close()

#endregion