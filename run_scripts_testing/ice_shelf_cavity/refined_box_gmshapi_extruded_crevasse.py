# base mesh for crevasse flume geometry structured gmsh
import gmsh
import sys

gmsh.initialize(sys.argv)

gmsh.model.add("2d_crevasse_flume")

# Let's create a simple rectangular geometry:
lc = 500
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
gmsh.model.mesh.field.setNumber(1, "VIn", lc / 5)
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


h = -1000
ov = gmsh.model.geo.extrude([(2, 6)], 0, 0, h, [20])
print(ov)


gmsh.model.addPhysicalGroup(2, [ov[0][1]], 1) # top
gmsh.model.addPhysicalGroup(2, [ov[2][1]], 2) # y = 0
gmsh.model.addPhysicalGroup(2, [ov[3][1]], 3) # x = 5000
gmsh.model.addPhysicalGroup(2, [ov[4][1]], 4) # y = 5000
gmsh.model.addPhysicalGroup(2, [ov[5][1]], 5) # x = 0 

gmsh.model.addPhysicalGroup(2, [6], 6) # bottom

gmsh.model.addPhysicalGroup(3, [1], 101)

#add crevasse
gmsh.model.geo.addPoint(0, 0.0, 500, lc, 15)
gmsh.model.geo.addPoint(L, 0.0, 500, lc, 16)
gmsh.model.geo.addPoint(L, L, 500, lc, 17)
gmsh.model.geo.addPoint(0, L, 500, lc, 18)

gmsh.model.geo.addLine(15, 16, 55)
gmsh.model.geo.addLine(16, 17, 56)
gmsh.model.geo.addLine(17, 18, 57)
gmsh.model.geo.addLine(18, 15, 58)


gmsh.model.geo.addLine(1, 15, 59)
gmsh.model.geo.addLine(2, 16, 60)
gmsh.model.geo.addLine(3, 17, 61)
gmsh.model.geo.addLine(4, 18, 62)


gmsh.model.geo.addCurveLoop([55, 56,57,58], 6)  # top
gmsh.model.geo.addCurveLoop([1, 60,-55, -59], 7)# y =0
gmsh.model.geo.addCurveLoop([2, 61,-56, -60], 8)# x =L
gmsh.model.geo.addCurveLoop([3, 62,-57, -61], 9)# y = L`
gmsh.model.geo.addCurveLoop([4, 59,-58, -62], 10)# y = L`

gmsh.model.geo.addPlaneSurface([6], 7)
gmsh.model.geo.addPlaneSurface([7], 8)
gmsh.model.geo.addPlaneSurface([8], 9)
gmsh.model.geo.addPlaneSurface([9], 10)
gmsh.model.geo.addPlaneSurface([10], 11)

gmsh.model.geo.addSurfaceLoop([6,7,8,9,10,11], 2)

gmsh.model.geo.addVolume([2], 2)

gmsh.model.addPhysicalGroup(3, [2], 102)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write("3d_crevasse_flume_withcrevassefull.msh")
gmsh.finalize()
