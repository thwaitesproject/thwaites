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


# We could also use a `Box' field to impose a step change in element sizes
# inside a box

# with horizontal resolution under crevasse of 5m got memory issues on cx1
# (limited by 1 node singularity solution...) 
# try 20m horizontal but keep 5m vertical.
gmsh.model.mesh.field.add("Box", 1)
gmsh.model.mesh.field.setNumber(1, "VIn", lc / 12.5)
gmsh.model.mesh.field.setNumber(1, "VOut", lc)
gmsh.model.mesh.field.setNumber(1, "XMin", 2400)
gmsh.model.mesh.field.setNumber(1, "XMax", 2600)
gmsh.model.mesh.field.setNumber(1, "YMin", 0)
gmsh.model.mesh.field.setNumber(1, "YMax", 5000)
gmsh.model.mesh.field.setNumber(1, "Thickness", 1000)
# Let's use the minimum of all the fields as the background mesh field:
gmsh.model.mesh.field.add("Min", 2)
gmsh.model.mesh.field.setNumbers(2, "FieldsList", [1])

gmsh.model.mesh.field.setAsBackgroundMesh(1)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.Algorithm", 5)

gmsh.model.mesh.generate(2)


h = 100
ov = gmsh.model.geo.extrude([(2, 6)], 0, 0, h, [20])
print(ov)


gmsh.model.addPhysicalGroup(2, [ov[0][1]], 1) # top
gmsh.model.addPhysicalGroup(2, [ov[2][1]], 2) # y = 0
gmsh.model.addPhysicalGroup(2, [ov[3][1]], 3) # x = 5000
gmsh.model.addPhysicalGroup(2, [ov[4][1]], 4) # y = 5000
gmsh.model.addPhysicalGroup(2, [ov[5][1]], 5) # x = 0 

gmsh.model.addPhysicalGroup(2, [6], 6) # bottom

gmsh.model.addPhysicalGroup(3, [1], 101)
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write("3d_crevasse_flume_test.msh")
gmsh.finalize()
