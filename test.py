from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner
from petsc4py.PETSc import ScalarType
import pyvista

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (4.0, 1.0)),
    n=(64, 16),
    cell_type=mesh.CellType.triangle
)

V = fem.functionspace(msh, ("Lagrange", 1))

facets = mesh.locate_entities_boundary(
    msh,
    dim=msh.topology.dim - 1,
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 4.0)
)

dofs = fem.locate_dofs_topological(
    V=V,
    entity_dim=1,
    entities=facets
)

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-((x[0] - 0.5)**2 + (x[1] - 0.5)**2) / 0.02)
g = ufl.sin(5 * x[0])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)

cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")
plotter = pyvista.Plotter()
plotter.show_grid()
#plotter.add_mesh(grid, show_edges=True)
warped = grid.warp_by_scalar()
plotter.add_mesh(warped, show_edges=False)
if pyvista.OFF_SCREEN:
    pyvista.start_xvfb(wait=0.1)
    plotter.screenshot("uh_poisson.png")
else:
    plotter.show()