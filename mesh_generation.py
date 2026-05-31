import itertools
import gmsh

# gmsh.initialize()

# dim = 3
# L = 32
# half_L = L / 2
# cell_size = 8
# half_cell = cell_size / 2

# gmsh.model.add("3d_spheres")

# outer = gmsh.model.occ.addBox(-half_L, -half_L, -half_L, L, L, L)

# holes = []
# centers = list(itertools.product([-L/4, L/4], repeat=dim))
# for (cx, cy, cz) in centers:
#     holes.append(gmsh.model.occ.addSphere(cx, cy, cz, half_cell))
    
# gmsh.model.occ.synchronize()

# sphere_surfaces = [s[1] for s in gmsh.model.getBoundary([(dim, h) for h in holes])]

# main_domain, _ = gmsh.model.occ.cut([(dim, outer)], [(dim, h) for h in holes])

# gmsh.model.occ.synchronize()

# DISTANCE_FIELD_TAG  = 1
# SAMPLING            = 999
# THRESHOLD_FIELD_TAG = 2
# SIZE_MIN            = 0.1
# SIZE_MAX            = 2
# DIST_MIN            = 0
# DIST_MAX            = 2

# gmsh.model.mesh.field.add("Distance", DISTANCE_FIELD_TAG)
# gmsh.model.mesh.field.setNumbers(DISTANCE_FIELD_TAG, "SurfacesList", sphere_surfaces)
# gmsh.model.mesh.field.setNumber(DISTANCE_FIELD_TAG, "Sampling", SAMPLING)

# gmsh.model.mesh.field.add("Threshold", THRESHOLD_FIELD_TAG)
# gmsh.model.mesh.field.setNumber(THRESHOLD_FIELD_TAG, "InField", DISTANCE_FIELD_TAG)
# gmsh.model.mesh.field.setNumber(THRESHOLD_FIELD_TAG, "DistMin", DIST_MIN)
# gmsh.model.mesh.field.setNumber(THRESHOLD_FIELD_TAG, "DistMax", DIST_MAX)
# gmsh.model.mesh.field.setNumber(THRESHOLD_FIELD_TAG, "SizeMin", SIZE_MIN)
# gmsh.model.mesh.field.setNumber(THRESHOLD_FIELD_TAG, "SizeMax", SIZE_MAX)

# gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
# gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
# gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

# gmsh.model.mesh.field.setAsBackgroundMesh(THRESHOLD_FIELD_TAG)

# gmsh.option.setNumber("Mesh.Algorithm", 5)

# gmsh.model.addPhysicalGroup(dim, [main_domain[0][1]], tag=100)
# gmsh.model.setPhysicalName(dim, 100, "main_domain")

# gmsh.model.mesh.generate(dim)

# gmsh.write("meshes/3d_spheres.msh")

# gmsh.finalize()



gmsh.initialize()

DOMAIN_TAG      = 100
MEMBRANE_TAG    = 10

dim = 2
L = 32
half_L = L / 2
cell_size = 8
half_cell = cell_size / 2

gmsh.model.add("square_with_holes")

outer = gmsh.model.occ.addRectangle(-half_L, -half_L, 0, L, L)

holes = []
centers = list(itertools.product([-L/4, L/4], repeat=dim))
for (cx, cy) in centers:
    holes.append(gmsh.model.occ.addRectangle(cx - half_cell, cy - half_cell, 0, cell_size, cell_size))

main_domain, _ = gmsh.model.occ.cut([(dim, outer)], [(dim, h) for h in holes])

gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(dim, [main_domain[0][1]], tag=DOMAIN_TAG)
gmsh.model.setPhysicalName(dim, DOMAIN_TAG, "main_domain")

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.12)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.04)
gmsh.model.mesh.generate(dim)

gmsh.write("meshes/2d_squares.msh")

gmsh.finalize()