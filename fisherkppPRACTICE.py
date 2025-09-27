import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

t = 0
T = 2 * np.pi
num_steps = 128
dt = T / num_steps

nx, ny = 64, 64
domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD, 
    points=[np.array([-2, -2]), np.array([2, 2])],
    n=[nx, ny],
    cell_type=mesh.CellType.triangle
)

V = fem.functionspace(domain, ("Lagrange", 1))

def initial_condition(x, a=5):
    return np.exp(-a * (x[0]**2 + x[1]**2))

# fdim = domain.topology.dim - 1
# boundary_facets = mesh.locate_entities_boundary(
#     msh=domain,
#     dim=fdim,
#     marker=lambda x: np.full(x.shape[1], True, dtype=bool)
# )
# bc = fem.dirichletbc(
#     value=PETSc.ScalarType(0),
#     dofs=fem.locate_dofs_topological(V, fdim, boundary_facets),
#     V=V
# )

# u_{n}
u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# u_{n+1}
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ((dt + 1) * u_n - dt * u_n * u_n) * v * ufl.dx

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

grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

plotter = pyvista.Plotter()
plotter.open_gif("out_fisherkpp/fisherkpp.gif", fps=10)
plotter.show_grid()

grid.point_data["uh"] = uh.x.array
warped = grid.warp_by_scalar("uh", factor=1)

plotter.add_mesh(
    warped,
    show_edges=True
)

for n in range(num_steps):
    t += dt
    
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    
    # apply_lifting(b, [bilinear_form], [[bc]])
    # b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    # set_bc(b, [bc])
    
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    u_n.x.array[:] = uh.x.array
    
    new_warped = grid.warp_by_scalar("uh", factor=1)
    warped.points[:, :] = new_warped.points
    warped.point_data["uh"][:] = uh.x.array
    plotter.write_frame()

plotter.close()