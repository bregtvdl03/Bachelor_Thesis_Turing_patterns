import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector
import dolfinx.io.gmsh as gmshio
import gmsh

OUT_FILE = "out_cells/test.gif"
FPS = 10

#region ========== MODEL PARAMETERS ==========

m = 2
n = 1

Du = 1.0    # Diffusion coef for u
Dv = 40.0   # Diffusion coef for v
Pu = 0.125    # Production coef for u
Pv = 0.420    # Production coef for v
gamma = 64.0**2 # Reaction scaling

uniform_steady_state_u = Pu + Pv
uniform_steady_state_v = (Pv / (Pu + Pv)**m) ** (1/n)

perturbation_strength = 0.01

def initial_condition_u(x):
    return uniform_steady_state_u + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return [uniform_steady_state_u] * x.shape[1]

def initial_condition_v(x):
    return uniform_steady_state_v + perturbation_strength * (np.random.rand(x.shape[1]) - 0.5)
    # return [uniform_steady_state_v] * x.shape[1]

t = 0.0
T = 50.0 / gamma
num_steps = 1024
dt = T / num_steps

#endregion

#region ========== DEFINING MESH AND FUNCTIONSPACE ==========

gmsh.initialize()

dim = 2
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

main_domain, _ = gmsh.model.occ.cut([(dim, outer)], [(dim, h) for h in holes])

gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(dim, [main_domain[0][1]], tag=100)
gmsh.model.setPhysicalName(dim, 100, "main_domain")

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.01)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01)
gmsh.model.mesh.generate(dim)

model_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=dim)
msh = model_data.mesh

gmsh.finalize()

def on_holes(x):
    flags = np.zeros(x.shape[1], dtype=bool)
    for (cx, cy) in centers:
        top    = np.isclose(x[1], cy + half_cell) & (cx - half_cell <= x[0]) & (x[0] <= cx + half_cell)
        right  = np.isclose(x[0], cx + half_cell) & (cy - half_cell <= x[1]) & (x[1] <= cy + half_cell)
        bottom = np.isclose(x[1], cy - half_cell) & (cx - half_cell <= x[0]) & (x[0] <= cx + half_cell)
        left   = np.isclose(x[0], cx - half_cell) & (cy - half_cell <= x[1]) & (x[1] <= cy + half_cell)
        flags |= top | right | bottom | left
    return flags

facet_indices = mesh.locate_entities(msh, dim - 1, on_holes)
facet_markers = mesh.meshtags(msh, dim - 1, facet_indices, 10)

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
    # + dt * gamma * (Pu - u + u * u * v_n) * phi * ds(10)

L_u = u_n * phi * ufl.dx + dt * gamma * (Pu - u_n + u_n * u_n * v_n) * phi * ufl.dx
# L_u = u_n * phi * ufl.dx + (dt * gamma * Pu) * phi * ds(10)

a_v = v * psi * ufl.dx \
    + dt * Dv * ufl.dot(ufl.grad(v), ufl.grad(psi)) * ufl.dx
    # + dt * gamma * (Pv - u_n * u_n * v) * psi * ds(10)

L_v = v_n * psi * ufl.dx + dt * gamma * (Pv - u_n * u_n * v_n) * psi * ufl.dx
# L_v = v_n * psi * ufl.dx

#endregion

#region ========== DEFINING SOLVERS ==========

bilinear_form_u = fem.form(a_u)
linear_form_u = fem.form(L_u)

A_u = assemble_matrix(bilinear_form_u)
A_u.assemble()
b_u = create_vector(fem.extract_function_spaces(linear_form_u))

solver_u = PETSc.KSP().create(msh.comm)
solver_u.setOperators(A_u)
solver_u.setType(PETSc.KSP.Type.PREONLY)
solver_u.getPC().setType(PETSc.PC.Type.LU)

bilinear_form_v = fem.form(a_v)
linear_form_v = fem.form(L_v)

A_v = assemble_matrix(bilinear_form_v)
A_v.assemble()
b_v = create_vector(fem.extract_function_spaces(linear_form_v))

solver_v = PETSc.KSP().create(msh.comm)
solver_v.setOperators(A_v)
solver_v.setType(PETSc.KSP.Type.PREONLY)
solver_v.getPC().setType(PETSc.PC.Type.LU)

#endregion

# Uncomment this for offscreen rendering
# pyvista.start_xvfb()

grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

warpfactor = 0.1

grid.point_data["uh"] = uh.x.array
grid.point_data["vh"] = vh.x.array
u_graph = grid.warp_by_scalar("uh", factor=warpfactor)
v_graph = grid.warp_by_scalar("vh", factor=warpfactor)

#region ========== PLOTTING SETUP ==========

plotter = pyvista.Plotter()
plotter.open_gif(OUT_FILE, fps=FPS)
plotter.show_grid()
plotter.enable_parallel_projection()
plotter.isometric_view()
plotter.view_xy()
plotter.show_grid(
    font_size = 15,
    font_family = "times",
    xtitle = "x",
    ytitle = "y",
    ztitle = "z"
)

blues = mpl.colormaps.get_cmap("Blues").resampled(32)
ylorrd = mpl.colormaps.get_cmap("YlOrRd").resampled(32)

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

#endregion

#region ========== SOLVING ITERATIVELY ==========

for n in range(num_steps):
    t += dt
    progress = int(n/num_steps * 100)
    time_text.SetText(2, f"{str(progress)}\t/100 %")
    print(progress)
    
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
    
    if n % 16 == 0:
        new_warped = grid.warp_by_scalar("uh", factor=warpfactor)
        u_graph.points[:, :] = new_warped.points
        u_graph.point_data["uh"][:] = uh.x.array

        new_warped = grid.warp_by_scalar("vh", factor=warpfactor)
        v_graph.points[:, :] = new_warped.points
        v_graph.point_data["vh"][:] = vh.x.array

        plotter.write_frame()

plotter.close()

#endregion