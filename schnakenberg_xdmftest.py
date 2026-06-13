import pyvista
import ufl
import numpy as np

import time
import os
from datetime import timedelta

from petsc4py import PETSc
from mpi4py import MPI

import basix.ufl
from dolfinx import fem, mesh, plot, io
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector

OUT_FILE = "out_schnakenberg/kleurtestje.gif"
OUT_SCREENSHOT = None   # "out_schnakenberg/schnakenberg_profile.jpg"
FPS = 10

#region ========== PARAMETERS ==========

m = 2
n = 1

Du = 1.0    # 1.0 Diffusion coef for u
Dv = 10.0   # 10.0 Diffusion coef for v
Pu = 0.1    # Production coef for u
Pv = 0.9    # Production coef for v
gamma = 32**2 # Reaction scaling

uniform_steady_state_u = Pu + Pv
uniform_steady_state_v = (Pv / (Pu + Pv)**m) ** (1/n)

perturbation_strength = 0.1

def initial_condition_u(x):
    return uniform_steady_state_u + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return (np.exp(-400 * (x[0]**2 + x[1]**2)) + np.exp(-400 * ((x[0] - 0.5)**2 + (x[1] + 0.3)**2)) + np.exp(-400 * ((x[0] + 0.2)**2 + (x[1] - 0.7)**2)) + np.exp(-400 * ((x[0] + 0.3)**2 + (x[1] + 0.8)**2))) + uniform_steady_state_u
    # return [uniform_steady_state_u] * x.shape[1]

def initial_condition_v(x):
    return uniform_steady_state_v + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return (np.exp(-400 * (x[0]**2 + x[1]**2)) + np.exp(-400 * ((x[0] - 0.5)**2 + (x[1] + 0.3)**2)) + np.exp(-400 * ((x[0] + 0.2)**2 + (x[1] - 0.7)**2)) + np.exp(-400 * ((x[0] + 0.3)**2 + (x[1] + 0.8)**2))) + uniform_steady_state_v
    # return [uniform_steady_state_v] * x.shape[1]

t = 0.0
T = 100.0 / gamma
num_steps = 4096
dt = T / num_steps

WRITE_EVERY = 32

nx, ny = 128, 128

domain = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=[[-1.0, -1.0], [1.0, 1.0]],
    n=[nx, ny],
    cell_type=mesh.CellType.triangle
)

el_u = basix.ufl.element("Lagrange", basix.CellType.triangle, 1)
el_v = basix.ufl.element("Lagrange", basix.CellType.triangle, 1)
el_mixed = basix.ufl.mixed_element([el_u, el_v])

V = fem.functionspace(domain, el_mixed)

#endregion

#region ========== DEFINING FUNCTIONS ==========

uv_n = fem.Function(V)
uv_n.name = "uv_n"

# u_{n}
u_n = uv_n.sub(0).collapse()
u_n.interpolate(initial_condition_u)

# v_{n}
v_n = uv_n.sub(1).collapse()
v_n.interpolate(initial_condition_v)

# u_{n + 1} and v_{n + 1}
uv_sol = fem.Function(V)

#endregion

#region ========== VARIATIONAL FORM ==========

(u, v) = ufl.TrialFunctions(V)
(phi, psi) = ufl.TestFunctions(V)

# IMEX
a = u * phi * ufl.dx \
    + v * psi * ufl.dx \
    + dt * Du * ufl.dot(ufl.grad(u), ufl.grad(phi)) * ufl.dx \
    + dt * Dv * ufl.dot(ufl.grad(v), ufl.grad(psi)) * ufl.dx

L = (u_n + dt * gamma * (Pu - u_n + u_n * u_n * v_n)) * phi * ufl.dx \
    + (v_n + dt * gamma * (Pv - 0.1 * v_n - u_n * u_n * v_n)) * psi * ufl.dx

#endregion

#region ========== SETUP PROBLEM AND SOLVERS ==========

bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(bilinear_form)
A.assemble()
b = create_vector(fem.extract_function_spaces(linear_form))

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.GMRES)
solver.setTolerances(rtol=1e-8, max_it=50)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.HYPRE)
pc.setHYPREType("boomeramg")

#endregion

#region ========== SETUP PLOTTING ==========

# Uncomment for offscreen rendering
# pyvista.start_xvfb()

xdmf = io.XDMFFile(domain.comm, "out_cells/XDMFs/test.xdmf", "w")
xdmf.write_mesh(domain)
xdmf.write_function(u_n, t)

V0, mapu = V.sub(0).collapse()
V1, mapv = V.sub(1).collapse()

#endregion

#region ========== SOLVE PROBLEM AND PLOT ==========

start = time.monotonic()

for n in range(num_steps):
    t += dt
    progress = int(t/T * 100)
    
    with b.localForm() as loc:
        loc.set(0)
    assemble_vector(b, linear_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    
    solver.solve(b, uv_sol.x.petsc_vec)
    uv_sol.x.scatter_forward()
    
    u_n.x.array[:] = uv_sol.sub(0).x.array[mapu]
    v_n.x.array[:] = uv_sol.sub(1).x.array[mapv]
    
    if n % WRITE_EVERY == 0:
        xdmf.write_function(u_n, t)
        
        if MPI.COMM_WORLD.rank == 0:
            os.system("clear")
            eta = timedelta(seconds=round((time.monotonic() - start) / (n + 1) * num_steps - time.monotonic() + start))
            print(f"ETA: {eta}")

xdmf.close()

#endregion