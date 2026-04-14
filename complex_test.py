import matplotlib as mpl
import pyvista
import ufl
import numpy as np
import cmath

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, assign

OUT_FILE = "out_cells/complex_test.gif"
FPS = 10

DOMAIN_TAG      = 100
MEMBRANE_TAG    = 10

#region ========== MODEL PARAMETERS ==========

m = 2
n = 1

Du      = 1.0           # Diffusion coef for u
Dw      = 1.0           # Diffusion coef for w
Dv      = 40.0          # Diffusion coef for v
Pu      = 0.125         # Production coef for u
Pv      = 0.420         # Production coef for v
k1      = 1.0           # k_on
k2      = 1.0           # k_off
du      = 1.0           # Decay coef for u
dw      = 1.0           # Decay coef for w
dv      = 0.1           # Decay coef for v
rho     = (m + n) * dw  # Upregulation coef
gamma   = 128.0 ** 2          # Reaction scaling

# g = k1 / (k2 + dw)
# a = k2 * g - k1
# b = (dv * rho * g / (n * a) + m * dv / n) ** m
# c = - (Pu + rho * g * Pv / (n * a) + m * Pv / n) ** m - du ** m * dv / (n * a)
# d = du ** m * Pv / (n * a)

# p = [b, 0, c, d]
# roots = np.roots(p)
# real_roots = []
# for r in roots:
#     if np.isreal(r):
#         real_roots.append(r.real)
#         break

# uniform_steady_state_v = min(real_roots)
# uniform_steady_state_u = ((Pv - dv * uniform_steady_state_v) / (n * a * uniform_steady_state_v ** n)) ** (1/m)
# uniform_steady_state_w = g * uniform_steady_state_u ** m * uniform_steady_state_v ** n

uniform_steady_state_u = Pu + Pv
uniform_steady_state_w = Pu + Pv
uniform_steady_state_v = (Pv / (Pu + Pv)**m) ** (1/n)

perturbation_strength = 0.1

print(f"uni_st_st_u: {uniform_steady_state_u}")
print(f"uni_st_st_w: {uniform_steady_state_w}")
print(f"uni_st_st_v: {uniform_steady_state_v}")

def initial_condition_u(x):
    return uniform_steady_state_u + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return [uniform_steady_state_u] * x.shape[1]
    
def initial_condition_w(x):
    return uniform_steady_state_w + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
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
L = 2.0
half_L = L / 2

nx, ny = 128, 128

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=[[-half_L, -half_L], [half_L, half_L]],
    n=[nx, ny],
    cell_type=mesh.CellType.triangle
)

V   = fem.functionspace(msh, ("Lagrange", 1))

W = ufl.MixedFunctionSpace(V, V, V)

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

# w_{n}
w_n = fem.Function(V)
w_n.name = "w_n"
w_n.interpolate(initial_condition_w)

# w_{n+1}
wh = fem.Function(V)
wh.name = "wh"
wh.interpolate(initial_condition_w)

# v_{n}
v_n = fem.Function(V)
v_n.name = "v_n"
v_n.interpolate(initial_condition_v)

# v_{n+1}
vh = fem.Function(V)
vh.name = "vh"
vh.interpolate(initial_condition_v)

v, u, w = ufl.TrialFunctions(W)
psi, phi, chi = ufl.TestFunctions(W)

#endregion

#region ========== VARIATIONAL FORM ==========

a = u * phi * ufl.dx \
    + w * chi * ufl.dx \
    + v * psi * ufl.dx \
    + dt * Du * ufl.dot(ufl.grad(u), ufl.grad(phi)) * ufl.dx \
    + dt * Dw * ufl.dot(ufl.grad(w), ufl.grad(chi)) * ufl.dx \
    + dt * Dv * ufl.dot(ufl.grad(v), ufl.grad(psi)) * ufl.dx \
    + dt * gamma * du * u * phi * ufl.dx \
    + dt * gamma * dw * w * chi * ufl.dx \
    + dt * gamma * dv * v * psi * ufl.dx \
    - dt * gamma * m * k2 * w * phi * ufl.dx \
    + dt * gamma *     k2 * w * chi * ufl.dx \
    - dt * gamma * n * k2 * w * psi * ufl.dx \

L = u_n * phi * ufl.dx \
    + w_n * chi * ufl.dx \
    + v_n * psi * ufl.dx \
    + dt * gamma * (Pu + rho * w_n) * phi * ufl.dx \
    + dt * gamma * Pv * psi * ufl.dx \
    - dt * gamma * m * k1 * (u_n * u_n * v_n) * phi * ufl.dx \
    + dt * gamma *     k1 * (u_n * u_n * v_n) * chi * ufl.dx \
    - dt * gamma * n * k1 * (u_n * u_n * v_n) * psi * ufl.dx \

#endregion

#region ========== DEFINING SOLVERS ==========

bilinear_form = fem.form(ufl.extract_blocks(a))
linear_form   = fem.form(ufl.extract_blocks(L))

A = assemble_matrix(bilinear_form)
A.assemble()
b = create_vector(fem.extract_function_spaces(linear_form))

solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

#endregion

#region ========== PLOTTING SETUP ==========

# Uncomment this for offscreen rendering
# pyvista.start_xvfb()

warpfactor = 1.0

grid_u = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
grid_w = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
grid_v = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

grid_u.point_data["uh"] = uh.x.array
grid_w.point_data["wh"] = wh.x.array
grid_v.point_data["vh"] = vh.x.array
u_graph = grid_u.warp_by_scalar("uh", factor=warpfactor)
w_graph = grid_w.warp_by_scalar("wh", factor=warpfactor)
v_graph = grid_v.warp_by_scalar("vh", factor=warpfactor)

plotter = pyvista.Plotter()
plotter.open_gif(OUT_FILE, fps=FPS)
plotter.show_grid()
plotter.enable_parallel_projection()
# plotter.isometric_view()
# plotter.view_xy()
plotter.show_grid(
    font_size = 15,
    font_family = "times",
    xtitle = "x",
    ytitle = "y",
    ztitle = "z"
)

blues  = mpl.colormaps.get_cmap("Blues").resampled(32)
jet    = mpl.colormaps.get_cmap("jet").resampled(32)
ylorrd = mpl.colormaps.get_cmap("YlOrRd").resampled(32)

plotter.add_mesh(
    u_graph,
    show_edges=False,
    lighting=False,
    cmap=blues,
    clim=[uniform_steady_state_u - perturbation_strength, uniform_steady_state_u + perturbation_strength],
    scalar_bar_args={
        "font_family": "times",
        "position_x": 0.2,
        "position_y": 0.9
    }
)

plotter.add_mesh(
    w_graph,
    show_edges=False,
    lighting=False,
    cmap=jet,
    clim=[uniform_steady_state_w - perturbation_strength, uniform_steady_state_w + perturbation_strength],
    scalar_bar_args={
        "font_family": "times",
        "position_x": 0.2,
        "position_y": 0.82
    }
)

plotter.add_mesh(
    v_graph,
    opacity=0.8,
    show_edges=False,
    lighting=False,
    cmap=ylorrd,
    clim=[uniform_steady_state_v - perturbation_strength, uniform_steady_state_v + perturbation_strength],
    scalar_bar_args={
        "font_family": "times",
        "position_x": 0.2,
        "position_y": 0.74
    }
)

time_text = plotter.add_text(
    f"{str(int(t/T * 100))}\t/100 %",
    font_size=10,
    font="times"
)

plotter.add_text(
    "d = 40, a = 0.125, b = 0.420",
    font_size=10,
    font="times",
    position=(5, 5)
)

plotter.write_frame()

#endregion

#region ========== SOLVING ITERATIVELY ==========

for n in range(num_steps):
    t += dt
    progress = int(n/num_steps * 100)
    time_text.SetText(2, f"{str(progress)}\t/100 %")
    print(progress)
    
    try:
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)
        x = create_vector([V, V, V])
        solver.solve(b, x)
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    except PETSc.Error as e:
        if e.ierr == 92:
            print("The required PETSc solver/preconditioner is not available. Exiting.")
            print(e)
            exit(0)
        else:
            raise e
    
    assign(x, [vh, uh, wh])
    x.destroy()
    
    u_n.x.array[:] = uh.x.array
    w_n.x.array[:] = wh.x.array
    v_n.x.array[:] = vh.x.array
    
    if n % 16 == 0:
        new_warped = grid_u.warp_by_scalar("uh", factor=warpfactor)
        u_graph.points[:, :] = new_warped.points
        u_graph.point_data["uh"][:] = uh.x.array

        new_warped = grid_w.warp_by_scalar("wh", factor=warpfactor)
        w_graph.points[:, :] = new_warped.points
        w_graph.point_data["wh"][:] = wh.x.array

        new_warped = grid_v.warp_by_scalar("vh", factor=warpfactor)
        v_graph.points[:, :] = new_warped.points
        v_graph.point_data["vh"][:] = vh.x.array
        
        print(f"u: {np.min(uh.x.array)} \t to \t {np.max(uh.x.array)}")
        print(f"w: {np.min(wh.x.array)} \t to \t {np.max(wh.x.array)}")
        print(f"v: {np.min(vh.x.array)} \t to \t {np.max(vh.x.array)}")

        plotter.write_frame()

plotter.close()

#endregion