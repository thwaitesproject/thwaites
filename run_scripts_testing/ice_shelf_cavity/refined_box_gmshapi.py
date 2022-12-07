# base mesh for crevasse flume geometry structured gmsh
import gmsh
import sys

gmsh.initialize(sys.argv)

gmsh.model.add("2d_crevasse_flume")

# Let's create a simple rectangular geometry:
lc = 250
L = 5000
gmsh.model.geo.addPoint(0.0, 0.0, 0, lc, 1)
gmsh.model.geo.addPoint(L, 0.0, 0, lc, 2)
gmsh.model.geo.addPoint(L, L, 0, lc, 3)
gmsh.model.geo.addPoint(0, L, 0, lc, 4)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 5)
gmsh.model.geo.addPlaneSurface([5], 6)

gmsh.model.geo.synchronize()

# We could also use a `Box' field to impose a step change in element sizes
# inside a box
gmsh.model.mesh.field.add("Box", 1)
gmsh.model.mesh.field.setNumber(1, "VIn", lc / 50)
gmsh.model.mesh.field.setNumber(1, "VOut", lc)
gmsh.model.mesh.field.setNumber(1, "XMin", 2400)
gmsh.model.mesh.field.setNumber(1, "XMax", 2600)
gmsh.model.mesh.field.setNumber(1, "YMin", 0)
gmsh.model.mesh.field.setNumber(1, "YMax", 5000)
gmsh.model.mesh.field.setNumber(1, "Thickness", 1000)
# Let's use the minimum of all the fields as the background mesh field:
#gmsh.model.mesh.field.add("Min", 2)
#gmsh.model.mesh.field.setNumbers(2, "FieldsList", [1])

gmsh.model.mesh.field.setAsBackgroundMesh(1)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.Algorithm", 5)

gmsh.model.mesh.generate(2)
gmsh.write("2d_crevasse_flume_test.msh")
gmsh.finalize()
