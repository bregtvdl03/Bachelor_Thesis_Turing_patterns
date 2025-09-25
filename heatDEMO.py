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
t = 0  # Start time
T = 1.0  # Final time
num_steps = 64
dt = T / num_steps  # time step size

# Define mesh
nx, ny = 64, 64
domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD, 
    points=[np.array([-2, -2]), np.array([2, 2])],
    n=[nx, ny],
    cell_type=mesh.CellType.triangle)

# Define function space
V = fem.functionspace(domain, ("Lagrange", 1))

# Create initial condition
def initial_condition(x, a=5):
    return np.exp(-a * (x[0]**2 + x[1]**2))

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

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

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, t)

# Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx