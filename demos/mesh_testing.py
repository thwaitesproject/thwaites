from thwaites import *
from math import pi
#mesh = Mesh("ice_shelf_mesh3.msh")

mesh = Mesh("./meshes/square_mesh_test2.msh")

V = VectorFunctionSpace(mesh, "DG", 1)  # It is a vector function space!!!!!!

v = Function(V)

v_init = Constant((0.0,0.0))
v.interpolate(v_init)

v_file = File("mesh_test.pvd")
v_file.write(v)

# test 2 !!1

mesh = Mesh("./meshes/square_mesh_test2.msh")

V = FunctionSpace(mesh, "DG", 1)  # It is a vector function space!!!!!!

v = Function(V)

v_init = Constant(0.0)
v.interpolate(v_init)

v_file = File("mesh_test2.pvd")
v_file.write(v)