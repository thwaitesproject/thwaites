# Buoyancy driven overturning circulation
# beneath ice shelf. Wedge geometry. 5km
# Outside temp forcing 3.0degC above freezing point for walter (S = 34.5) at 1000m depth.
# viscosity = temp diffusivity = sal diffusivity: varies linearly over the domain.
from thwaites import *
from thwaites.utility import ice_thickness, cavity_thickness, get_top_boundary
##########

folder = "./"  # output folder.
L = 4.75
H1 = 2
H2 = 100
dx = 50
nx = round(L/dx)
nz = 10
dz = H1/nz
mesh = Mesh("./initial_mesh.msh")



Q = FunctionSpace(mesh, "DG", 1)  # melt function space
m = Function(Q)
##########

m.assign(Constant(0))

##########

# Output files for velocity, pressure, temperature and salinity
initial_mesh_file = File(folder+"initial_mesh.pvd")
initial_mesh_file.write(m)



##########

Lnew = 10000.
lhs_stretching = 10.0
rhs_stretching = Lnew/L
x = mesh.coordinates.dat.data[:,0]
y = mesh.coordinates.dat.data[:,1]
mesh.coordinates.dat.data[:,0] = ((rhs_stretching - lhs_stretching) * x / L + lhs_stretching) * x


Q2 = FunctionSpace(mesh, "DG", 1)  # melt function space
q2 = Function(Q)
##########

q2.assign(Constant(0))

##########

# Output files for velocity, pressure, temperature and salinity
stretched_yx_file = File(folder+"stretched_x_mesh.pvd")
stretched_yx_file.write(q2)



x = mesh.coordinates.dat.data[:,0]
y = mesh.coordinates.dat.data[:,1]
mesh.coordinates.dat.data[:,1] = ((x/Lnew)*(H2-H1) + H1)*y


##########

# Set up function spaces

Q = FunctionSpace(mesh, "DG", 1)  # melt function space
q = Function(Q)
##########

q.assign(Constant(0))



# Output files for velocity, pressure, temperature and salinity
stretched_y_file = File(folder+"stretched_xy_mesh.pvd")
stretched_y_file.write(q)
