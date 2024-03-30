# base mesh for crevasse flume geometry structured gmsh
import gmsh
import sys

gmsh.initialize(sys.argv)

gmsh.model.add("2d_crevasse_flume")

# Let's create a simple rectangular geometry:
lc = 250
L = 5000

negative=False
#add crevasse
if negative:
    # negative slant of 59 deg. i.e tan (5km/3km) = 59deg
    gmsh.model.geo.addPoint(3800, 0.0, 0, 5, 21)
    gmsh.model.geo.addPoint(3875, 0.0, 50, 5, 22)
    gmsh.model.geo.addPoint(3925, 0.0, 50, 5, 23)
    gmsh.model.geo.addPoint(4000, 0.0, 0, 5, 24)


    gmsh.model.geo.addPoint(1000, L, 0, 5, 25)
    gmsh.model.geo.addPoint(1075, L, 50, 5, 26)
    gmsh.model.geo.addPoint(1125, L, 50, 5, 27)
    gmsh.model.geo.addPoint(1200, L, 0, 5, 28)
else:
    # positive slant of 59 deg. i.e tan (5km/3km) = 59deg
    gmsh.model.geo.addPoint(1000, 0.0, 0, 5, 21)
    gmsh.model.geo.addPoint(1075, 0.0, 50, 5, 22)
    gmsh.model.geo.addPoint(1125, 0.0, 50, 5, 23)
    gmsh.model.geo.addPoint(1200, 0.0, 0, 5, 24)


    gmsh.model.geo.addPoint(3800, L, 0, 5, 25)
    gmsh.model.geo.addPoint(3875, L, 50, 5, 26)
    gmsh.model.geo.addPoint(3925, L, 50, 5, 27)
    gmsh.model.geo.addPoint(4000, L, 0, 5, 28)



gmsh.model.geo.addLine(21, 22, 21)
gmsh.model.geo.addLine(22, 23, 22)
gmsh.model.geo.addLine(23, 24, 23)
gmsh.model.geo.addLine(24, 21, 24)


gmsh.model.geo.addLine(25, 26, 25)
gmsh.model.geo.addLine(26, 27, 26)
gmsh.model.geo.addLine(27, 28, 27)
gmsh.model.geo.addLine(28, 25, 28)

# bottom crevasse connecting lines
gmsh.model.geo.addLine(21, 25, 29)
gmsh.model.geo.addLine(24, 28, 30)

# top crevasse connecting lines
gmsh.model.geo.addLine(22, 26, 31)
gmsh.model.geo.addLine(23, 27, 32)



gmsh.model.geo.addCurveLoop([31, 26, -32, -22], 21)  # top
gmsh.model.geo.addCurveLoop([21, 22, 23, 24], 22)# y =0
gmsh.model.geo.addCurveLoop([30, -27,-32, 23], 23)# x =L
gmsh.model.geo.addCurveLoop([28, 25,26, 27], 24)# y = L`
gmsh.model.geo.addCurveLoop([-29, 21,31, -25], 25)# x = 0
gmsh.model.geo.addCurveLoop([29, -28,-30, 24], 26)# bottom

gmsh.model.geo.addPlaneSurface([21], 51)
gmsh.model.geo.addPlaneSurface([22], 52)
gmsh.model.geo.addPlaneSurface([23], 53)
gmsh.model.geo.addPlaneSurface([24], 54)
gmsh.model.geo.addPlaneSurface([25], 55)
gmsh.model.geo.addPlaneSurface([26], 56)

gmsh.model.geo.addSurfaceLoop([51, 52, 53, 54, 55, 56], 2)

gmsh.model.geo.addVolume([2], 2)

gmsh.model.addPhysicalGroup(3, [2], 102)

gmsh.model.geo.addPoint(0.0, 0.0, 0, lc, 1)
gmsh.model.geo.addPoint(L, 0.0, 0, lc, 2)
gmsh.model.geo.addPoint(L, L, 0, lc, 3)
gmsh.model.geo.addPoint(0, L, 0, lc, 4)


# linees around the square
gmsh.model.geo.addLine(1, 21, 1)
gmsh.model.geo.addLine(24, 2, 2)
gmsh.model.geo.addLine(2, 3, 3)
gmsh.model.geo.addLine(3, 28, 4)
gmsh.model.geo.addLine(25, 4, 5)
gmsh.model.geo.addLine(4, 1, 6)

gmsh.model.geo.addCurveLoop([1, 29, 5, 6], 1) 
gmsh.model.geo.addCurveLoop([2, 3, 4, -30], 2) 
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.addPlaneSurface([2], 2)

# add physical tag for melt top boundary outside of crevasse
#gmsh.model.geo.addCurveLoop([1, 29, 5, 6], 31)  # x < 2400
#gmsh.model.geo.addCurveLoop([2, 3, 4, -30], 32) # x > 2600
#gmsh.model.geo.addPlaneSurface([31], 31)
#gmsh.model.geo.addPlaneSurface([32], 32)

# We could also use a `Box' field to impose a step change in element sizes
# inside a box

# with horizontal resolution under crevasse of 5m got memory issues on cx1
# (limited by 1 node singularity solution...) 
# try 20m horizontal but keep 5m vertical.
# Let's use the minimum of all the fields as the background mesh field:
#below box is for 90deg alignment!!!
#gmsh.model.mesh.field.add("Box", 1)
#gmsh.model.mesh.field.setNumber(1, "VIn", lc / 12.5)
#gmsh.model.mesh.field.setNumber(1, "VOut", lc)
#gmsh.model.mesh.field.setNumber(1, "XMin", 2400)
#gmsh.model.mesh.field.setNumber(1, "XMax", 2600)
#gmsh.model.mesh.field.setNumber(1, "YMin", 0)
#gmsh.model.mesh.field.setNumber(1, "YMax", 5000)
#gmsh.model.mesh.field.setNumber(1, "Thickness", 1000)


gmsh.model.geo.synchronize()
#  higher res inside crevasse - use lines 29 and 30 

gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", [29, 30])
gmsh.model.mesh.field.setNumber(1, "Sampling", 1000)
#gmsh.model.mesh.field.add("Threshold", 2)
#gmsh.model.mesh.field.setNumber(2, "DistMax", 1000)
#gmsh.model.mesh.field.setNumber(2, "DistMin", 100)
#gmsh.model.mesh.field.setNumber(2, "InField", 1)
#gmsh.model.mesh.field.setNumber(2, "SizeMin", lc / 12.5)
#gmsh.model.mesh.field.setNumber(2, "SizeMax", lc)
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", lc / 12.5)
gmsh.model.mesh.field.setNumber(2, "SizeMax", lc)
gmsh.model.mesh.field.setNumber(2, "DistMin", 100)
gmsh.model.mesh.field.setNumber(2, "DistMax", 500)



gmsh.model.mesh.field.add("Box", 3)
gmsh.model.mesh.field.setNumber(3, "VIn", 5)
gmsh.model.mesh.field.setNumber(3, "VOut", lc)
gmsh.model.mesh.field.setNumber(3, "XMin", 0)
gmsh.model.mesh.field.setNumber(3, "XMax", 5000)
gmsh.model.mesh.field.setNumber(3, "YMin", 0)
gmsh.model.mesh.field.setNumber(3, "YMax", 5000)
gmsh.model.mesh.field.setNumber(3, "ZMin", 5)
gmsh.model.mesh.field.setNumber(3, "ZMax", 50)
gmsh.model.mesh.field.setNumber(3, "Thickness", 10)
gmsh.model.mesh.field.add("Min", 4)
gmsh.model.mesh.field.setNumbers(4, "FieldsList", [2,3])

gmsh.model.mesh.field.setAsBackgroundMesh(4)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.Algorithm", 5)

gmsh.model.mesh.generate(2)

h = -100
ov = gmsh.model.geo.extrude([(2, 1), (2,56), (2,2)], 0, 0, h, [20])
print(ov)


gmsh.model.addPhysicalGroup(2, [ov[0][1], ov[6][1], ov[12][1]], 1) # bottom  # 3 volumes go up in 6: for 4 sides + bottom + vol 
gmsh.model.addPhysicalGroup(2, [ov[2][1], ov[11][1],ov[14][1],52], 2) # y = 0  # middle volume is defined differently because the surface loop is clocwise cf other two anticlockwise! 
gmsh.model.addPhysicalGroup(2, [ov[15][1]], 5) # x = L 
gmsh.model.addPhysicalGroup(2, [ov[4][1], ov[9][1], ov[16][1], 54], 3) # y = L  # middle volume is defined differently because the surface loop is clocwise cf other two anticlockwise! 

gmsh.model.addPhysicalGroup(2, [ov[5][1]], 6) # x = 0 

# top melt surface to 4
gmsh.model.addPhysicalGroup(2, [1,2, 51, 53 , 55], 4) # 32 31

gmsh.model.addPhysicalGroup(3, [ov[1][1],ov[7][1],ov[13][1]], 101)

gmsh.model.geo.synchronize()
#gmsh.write("3d_crevasse_flume_test_extrapoints.geo_unrolled")
gmsh.model.mesh.generate(3)
if negative:
    angle = "negative"
else:
    angle = "positive"
gmsh.write(f"3d_crevasse_flume_dx250mto20m_dz5m_crevdxz5m_{angle}.msh")
gmsh.finalize()
