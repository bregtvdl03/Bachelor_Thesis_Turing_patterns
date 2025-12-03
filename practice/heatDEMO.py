# SOURCE: https://jsdokken.com/dolfinx-tutorial/chapter2/diffusion_code.html

import matplotlib as mpl
import pyvista
import ufl
import numpy as np
import gmsh
from dolfinx.io import gmshio

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

# Define temporal parameters
t = 0
T = 2 * np.pi
num_steps = 128
dt = T / num_steps

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
    if area > 1.0:
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

domain, cell_markers, facet_markers = gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=2
)

gmsh.finalize()

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_markers)

# Define function space
V = fem.functionspace(domain, ("Lagrange", 1))

# Create initial condition
def initial_condition(x, a=5):
    return 2 * np.exp(-a * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)) + np.exp(-2*a * ((x[0] + 1)**2 + (x[1] + 0.5)**2))

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    msh=domain,
    dim=fdim,
    marker=lambda x: np.full(x.shape[1], True, dtype=bool)
)
bc = fem.dirichletbc(
    value=PETSc.ScalarType(0),
    dofs=fem.locate_dofs_topological(V, fdim, boundary_facets),
    V=V
)

# u_{n}
u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# u_{n+1}
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

# Variational problem in UFL
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)
time = fem.Constant(domain, PETSc.ScalarType(0))
f = 5 * ufl.exp(-5*((x[0] + ufl.cos(5 * time))**2 + (x[1] + ufl.sin(5 * time))**2)) + \
    10 * ufl.exp(-5*((x[0] + ufl.cos(-3 * time))**2 + (x[1] + ufl.sin(-3 * time))**2))
# f = fem.Constant(domain, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx + dt * 100 * v * ds(10)

# Convert UFL variational form to DolfinX
bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(bilinear_form, bcs = [bc])
A.assemble()
b = create_vector(linear_form)

# Create solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# pyvista.start_xvfb()

grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

plotter = pyvista.Plotter()
plotter.open_gif("out_heat/diffusion.gif", fps=10)
plotter.show_grid()

grid.point_data["uh"] = uh.x.array
warped = grid.warp_by_scalar("uh", factor=1)

viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
sargs = dict(
    title_font_size=25,
    label_font_size=20,
    fmt="%.2e",
    color="black",
    position_x=0.1,
    position_y=0.8,
    width=0.5,
    height=0.05
)

renderer = plotter.add_mesh(
    warped,
    show_edges=False,
    lighting=False,
    cmap=viridis,
    scalar_bar_args=sargs,
    clim=[-1, 1]
)

for i in range(num_steps):
    t += dt
    time.value = t

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array
    
    # Update plot
    new_warped = grid.warp_by_scalar("uh", factor=1)
    warped.points[:, :] = new_warped.points
    warped.point_data["uh"][:] = uh.x.array
    plotter.write_frame()

plotter.close()