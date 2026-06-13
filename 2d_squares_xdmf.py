import ufl
import numpy as np
import itertools

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, assign
import dolfinx.io.gmsh as gmshio
import gmsh

OUT_FILE_BULK = "out_cells/XDMFs/2d_squares_bulk.xdmf"
OUT_FILE_SURF = "out_cells/XDMFs/2d_squares_surf.xdmf"
FPS = 10

DOMAIN_TAG      = 100
MEMBRANE_TAG    = 10

#region ========== MODEL PARAMETERS ==========

m = 2
n = 1

Du      = 1.0       # Diffusion coef for u
Dv      = 50.0      # Diffusion coef for v
Pu      = 0.1     # Production coef for u
Pv      = 0.9     # Production coef for v
gamma   = 64.0      # Reaction scaling

uniform_steady_state_u = Pu + Pv
uniform_steady_state_v = (Pv / (Pu + Pv)**m) ** (1/n)

perturbation_strength = 0.1

def initial_condition_u(x):
    return uniform_steady_state_u + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return [uniform_steady_state_u] * x.shape[1]

def initial_condition_v(x):
    return uniform_steady_state_v + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return [uniform_steady_state_v] * x.shape[1]

t = 0.0
T = 100.0 / gamma
num_steps = 1024
dt = T / num_steps

#endregion

#region ========== DEFINING MESH AND FUNCTIONSPACE ==========

dim = 2
L = 32
half_L = L / 2
cell_size = 8
half_cell = cell_size / 2

centers = list(itertools.product([-L/4, L/4], repeat=dim))

mesh_data = gmshio.read_from_msh("meshes/2d_squares.msh", MPI.COMM_WORLD, rank=0, gdim=dim)
msh = mesh_data.mesh

def on_holes(x):
    flags = np.zeros(x.shape[1], dtype=bool)
    for (cx, cy) in centers:
        c = np.reshape([cx, cy, 0], (3, 1))
        dist = np.linalg.norm(x - c, ord=np.inf, axis=0)
        flags |= np.isclose(dist, half_cell)
    return flags

facet_indices = mesh.locate_entities(msh, dim - 1, on_holes)
facet_markers = mesh.meshtags(msh, dim - 1, facet_indices, MEMBRANE_TAG)

msh.topology.create_entities(dim - 1)
boundary_msh, boundary_msh_emap = mesh.create_submesh(msh, dim - 1, facet_indices)[:2]

ds      = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers)
dx      = ufl.Measure("dx", domain=msh)
ds_b    = ufl.Measure("dx", domain=boundary_msh)

V   = fem.functionspace(msh         , ("Lagrange", 1))
Vb  = fem.functionspace(boundary_msh, ("Lagrange", 1))

W = ufl.MixedFunctionSpace(V, Vb)

#endregion

#region ========== DEFINING FUNCTIONS ==========

# u_{n}
u_n = fem.Function(Vb)
u_n.name = "u_n"
u_n.interpolate(initial_condition_u)

# u_{n+1}
uh = fem.Function(Vb)
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

v, u = ufl.TrialFunctions(W)
psi, phi = ufl.TestFunctions(W)

#endregion

#region ========== VARIATIONAL FORM ==========

# TODO: try fractional theta method: https://www.sciencedirect.com/science/article/pii/S0168874X15001377

a = u * phi * ds(MEMBRANE_TAG) \
    + dt * Du * ufl.dot(ufl.grad(u), ufl.grad(phi)) * ds(MEMBRANE_TAG) \
    + v * psi * dx \
    + dt * Dv * ufl.dot(ufl.grad(v), ufl.grad(psi)) * dx \
    + dt * gamma * (u) * phi * ds(MEMBRANE_TAG) \

L = u_n * phi * ds(MEMBRANE_TAG) \
    + v_n * psi * dx \
    + dt * gamma * (Pu) * phi * ds(MEMBRANE_TAG) \
    + dt * gamma * (Pv) * psi * ds(MEMBRANE_TAG) \
    + dt * gamma * (u_n * u_n * v_n) * phi * ds(MEMBRANE_TAG) \
    - dt * gamma * (u_n * u_n * v_n) * psi * ds(MEMBRANE_TAG) \

#endregion

#region ========== DEFINING SOLVERS ==========

bilinear_form   = fem.form(ufl.extract_blocks(a), entity_maps=[boundary_msh_emap])
linear_form     = fem.form(ufl.extract_blocks(L), entity_maps=[boundary_msh_emap])

A = assemble_matrix(bilinear_form)
A.assemble()
b = create_vector(fem.extract_function_spaces(linear_form))

solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

#endregion

#region ========== PLOTTING SETUP ==========

xdmf_surf = io.XDMFFile(msh.comm, OUT_FILE_SURF, "w")
xdmf_surf.write_mesh(boundary_msh)
xdmf_surf.write_function(u_n, t)

xdmf_bulk = io.XDMFFile(msh.comm, OUT_FILE_BULK, "w")
xdmf_bulk.write_mesh(msh)
xdmf_bulk.write_function(v_n, t)

#endregion

#region ========== SOLVING ITERATIVELY ==========

for n in range(num_steps):
    t += dt
    progress = int(n/num_steps * 100)
    
    try:
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        
        x = create_vector([V, Vb])
        solver.solve(b, x)
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    except PETSc.Error as e:
        if e.ierr == 92:
            print("The required PETSc solver/preconditioner is not available. Exiting.")
            print(e)
            exit(0)
        else:
            raise e
    
    assign(x, [vh, uh])
    x.destroy()
    
    u_n.x.array[:] = uh.x.array
    v_n.x.array[:] = vh.x.array
    
    if n % 16 == 0:
        xdmf_surf.write_function(u_n, t)
        xdmf_bulk.write_function(v_n, t)

xdmf_surf.close()
xdmf_bulk.close()

#endregion