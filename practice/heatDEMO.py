# SOURCE: https://jsdokken.com/dolfinx-tutorial/chapter2/diffusion_code.html

import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

# Define temporal parameters
t = 0
T = 2 * np.pi
num_steps = 128
dt = T / num_steps

# Define mesh
nx, ny = 64, 64
domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD, 
    points=[np.array([-2, -2]), np.array([2, 2])],
    n=[nx, ny],
    cell_type=mesh.CellType.triangle
)

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

xdmf = io.XDMFFile(domain.comm, "out_heat/diffusion.xdmf", "w")
xdmf.write_mesh(domain)

# u_{n}
u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# u_{n+1}
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

xdmf.write_function(uh, t)

# Variational problem in UFL
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)
time = fem.Constant(domain, PETSc.ScalarType(0))
f = 5 * ufl.exp(-5*((x[0] + ufl.cos(5 * time))**2 + (x[1] + ufl.sin(5 * time))**2)) + \
    10 * ufl.exp(-5*((x[0] + ufl.cos(-3 * time))**2 + (x[1] + ufl.sin(-3 * time))**2))
# f = fem.Constant(domain, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx

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

pyvista.start_xvfb()

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
    show_edges=True,
    lighting=False,
    cmap=viridis,
    scalar_bar_args=sargs,
    clim=[0, max(uh.x.array)]
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

    # Write solution to file
    xdmf.write_function(uh, t)
    
    # Update plot
    new_warped = grid.warp_by_scalar("uh", factor=1)
    warped.points[:, :] = new_warped.points
    warped.point_data["uh"][:] = uh.x.array
    plotter.write_frame()

plotter.close()
xdmf.close()