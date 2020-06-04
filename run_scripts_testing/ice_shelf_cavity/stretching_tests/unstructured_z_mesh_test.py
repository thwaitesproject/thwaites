# Buoyancy driven overturning circulation
# beneath ice shelf. Wedge geometry. 5km
# Outside temp forcing 3.0degC above freezing point for walter (S = 34.5) at 1000m depth.
# viscosity = temp diffusivity = sal diffusivity: varies linearly over the domain.
from thwaites import *
from thwaites.utility import ice_thickness, cavity_thickness, get_top_boundary
##########

folder = "./"  # output folder.
L = 5*1e3
H1 = 1
H2 = 100
dx = 50
nx = round(L/dx)
nz = 10
dz = H1/nz
mesh = Mesh("./test_unstructured_rectangle.msh")
x = mesh.coordinates.dat.data[:,0]
y = mesh.coordinates.dat.data[:,1]
mesh.coordinates.dat.data[:,1] = ((x/L)*(H2-H1) + H1)*y


x, z = SpatialCoordinate(mesh)


##########

# Set up function spaces

Q = FunctionSpace(mesh, "DG", 1)  # melt function space
q = Function(mesh,"DG",1)
##########

q.assign(Constant(0))

##########

# Output files for velocity, pressure, temperature and salinity
u_file = File(folder+"mesh.pvd")
u_file.write(q)

