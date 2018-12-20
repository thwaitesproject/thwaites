# Buoyancy driven overturning circulation
# implement a bc which is equivalent to ocean - ice boundary when ice acts as an insulator.
# i.e no advection/no diffusion of heat into ice so Q^T_I = 0
# Two equation formulation so assume instataneous diffusion of salinity. -> working towards three equation
# T_b = aS_b + b + cP  # salinity/pressure dependent melting point of water.
# w_b = T_m - (aS_b+b-cP).C_p_m . gamma_T / L_f  # melt rate of water see end of notebook 1. (Holland and Jenkins 1999)
#
from thwaites import *
from math import pi
import numpy as np
mesh = UnitSquareMesh(40, 40) #Mesh("ice_shelf_mesh3.msh")#

#mesh = Mesh("./meshes/square_mesh_test3.msh")
# We set up a function space of discontinous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
W = FunctionSpace(mesh, "CG", 2)  # pressure space
Z = MixedFunctionSpace([V,W])

Q = FunctionSpace(mesh, "DG", 1)  # density space
K = FunctionSpace(mesh,"DG",1)    # temperature space
S = FunctionSpace(mesh,"DG",1)    # salinity space

M = FunctionSpace(mesh,"DG",1)      # melt space - dont really need but might be easier for plotting in paraview


z = Function(Z)
u_, p_ = z.split()
rho = Function(Q)
temp = Function(K)
sal = Function(S)
melt = Function(M)


# We set up the initial velocity field using a simple analytic expression. ::

x,y = SpatialCoordinate(mesh)
#print(x.dat.data[:,:])

#x1 = mesh.coordinates.dat.data


top_boundary_coordinates =[]
for i in range(40):
    top_boundary_coordinates.append([i*0.025, 0.98])

print(top_boundary_coordinates)


u_init = Constant((0.0, 0.0))
u_.assign(u_init)

# the diffusivity and viscosity
kappa_temp = Constant(1e-3)
kappa_sal = Constant(1e-3)
#mu = Constant(1e-3)
mu = Constant(20e-3)



#bell_r0 = 40*0.15; bell_x0 = 40*0.25; bell_y0 = 40*0.5
#cone_r0 = 40*0.15; cone_x0 = 40*0.5; cone_y0 = 40*0.25
#cyl_r0 = 40*0.15; cyl_x0 = 40*0.5; cyl_y0 = 40*0.75
#slot_left = 40*0.475; slot_right = 40*0.525; slot_top = 40*0.85


bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
cone_r0 = 0.15; cone_x0 = 0.5; cone_y0 = 0.25
cyl_r0 = 0.15; cyl_x0 = 0.5; cyl_y0 = 0.75
slot_left = 0.475; slot_right = 0.525; slot_top = 0.85



bell = 0.25*(1+cos(pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
             conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
               0.0, 1.0), 0.0)

# the tracer function and its initial condition
# change from rho to temp...
#rho_init = Constant(0.0)+slot_cyl
#rho.interpolate(rho_init)

temp_init = Constant(0.0)+slot_cyl   # deg C from ben's thesis page 54
temp.interpolate(temp_init)


sal_init = Constant(34.4) # PSU  from ben's thesis page 54
sal.interpolate(sal_init)

melt_init = Constant(0)
melt.interpolate(melt_init)

folder = "./05.11.18.two_eq_ice_insulator_test2/"
# We declare the output filenames, and write out the initial conditions. ::
u_file = File(folder+"velocity.pvd")
u_file.write(u_)
p_file = File(folder+"pressure.pvd")
p_file.write(p_)
d_file = File(folder+"density.pvd")
d_file.write(rho)
t_file = File(folder+"temperature.pvd")
t_file.write(temp)
s_file = File(folder+"salinity.pvd")
s_file.write(sal)
m_file = File(folder+"melt.pvd")
m_file.write(melt)

# time period and time step
T = 10.
dt = T/1000.

#T = 5.
#dt = T/5000.




u_test, p_test = TestFunctions(Z)

#mom_eq = MomentumEquation(u_test, Z.sub(0))
# i think stephan changed this from test to testspace in the last git pull...
mom_eq = MomentumEquation(Z.sub(0), Z.sub(0))

#cty_eq = ContinuityEquation(p_test, Z.sub(1))
cty_eq = ContinuityEquation(Z.sub(1), Z.sub(1))


#rho_test = TestFunction(Q)
#rho_eq = ScalarAdvectionDiffusionEquation(temp_test, Q)

temp_test = TestFunction(K)
#temp_eq = ScalarAdvectionDiffusionEquation(temp_test, K)
temp_eq = ScalarAdvectionDiffusionEquation(K, K)


sal_test = TestFunction(S)
#sal_eq = ScalarAdvectionDiffusionEquation(sal_test, S)
sal_eq = ScalarAdvectionDiffusionEquation(S, S)


u, p = split(z)

#source = as_vector((0, -1.0))*rho  # momentum source: the buoyancy term

#From Ben's thesis page 31
#rho_0 = 1.0
Tref= 0.0
Sref=34.8 #PSU
beta_temp = 3.87*10E-5 #5.0E-3
beta_sal=7.86*10E-4
g = 9.81
mom_source = as_vector((0, -g))*(-beta_temp*(temp-Tref)+beta_sal*(sal-Sref))  # momentum source: the buoyancy term boussinesq approx


rho0=1020  # used later to recalculate the full pressure for meltrate without hydrostatic term



# sourcefor scalar equations at open boundary. linearly relaxes to T/Srestore
absorption_factor =2.0E-4

#Trestore = 0.37  # degC  n.b delta T = 3degC  (ben's thesis page
Trestore = -0.5

Srestore = 34.5 #PSU

interp_source_temp = Function(K)
interp_source_sal = Function(S)

source_temp = conditional(x>0.75,absorption_factor*((x-0.75)/0.25)*Trestore, 0.0)
source_sal = conditional(x>0.75,absorption_factor*((x-0.75)/0.25)*Trestore, 0.0)

interp_source_temp.interpolate(source_temp)
interp_source_sal.interpolate(source_sal)
#x_vector, y_vector = interpolate(x[0], Function(DG_2d)).dat.data, interpolate(x[1], Function(DG_2d)).dat.data

#absorption term
interp_absorp_temp = Function(K)
interp_absorp_sal = Function(S)

absorp_temp = conditional(x>0.75,(absorption_factor*(x-0.75)/0.25), 0.0)
absorp_sal = conditional(x>0.75,(absorption_factor*(x-0.75)/0.25), 0.0)

interp_absorp_temp.interpolate(absorp_temp)
interp_absorp_sal.interpolate(absorp_sal)


up_fields = {'viscosity': mu, 'source': mom_source}
#rho_fields = {'diffusivity': kappa, 'velocity': u}
temp_fields = {'diffusivity': kappa_temp, 'velocity': u, 'source': interp_source_temp, 'absorption coefficient': interp_absorp_temp}

sal_fields = {'diffusivity': kappa_sal, 'velocity': u, 'source': interp_source_sal, 'absorption coefficent': interp_absorp_sal}

mumps_solver_parameters = {
    'snes_monitor': True,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij'
}
# weakly applied dirichlet bcs on top and bottom for density
#rho_top = 1.0
#rho_bottom = 0.0
#rho_bcs = {3: {'q': rho_bottom}, 4: {'q': rho_top}}
#rho_solver_parameters = mumps_solver_parameters

# weakly applied dirichlet bcs on top and bottom for temp


def meltrate(S, T, Pin):
    dy = 1.0/40.0

    a = -5.73E-2  # salinity coefficient of freezing eqation
    b = 9.39E-2   # constant coeff of freezing equation
    c = -7.53E-8    # pressure coeff of freezing equation

    c_p_m = 3974 # specific heat capacity of mixed layer
    gammaT = 1E-4 # roughly thermal exchange velocity
    Lf = 3.34E5  # latent heat of fusion

    #Sb = np.array(S.at(coordinates))
    #Tm = np.array(T.at(coordinates))
    #P = np.array(Pin.at(coordinates))

    rho_ice = 920.0
    h_ice = 1. # in m
    P_h = rho_ice*g*h_ice  # hydrostatic pressure just from ice
    Pfull= rho0*Pin+P_h

    Tb = a*S + b + c*Pfull

    return conditional(y > 1-dy, (T-Tb)*c_p_m*gammaT/Lf, 0.0)  # only calculate for top layer...

class melt_boundary(Expression):
    def eval(self, value, X):
        value[:] = numpy.dot(X, X)

#a = melt_boundary()
#print("a")
#print(a)



temp_bcs = {}#4: {'flux': rho0*melt*3.34*10**5}}   # heat flux = rho0*melt*Lf
temp_solver_parameters = mumps_solver_parameters



#sal_right = 40.0
sal_bcs ={} # {1: {'q': sal_ice}}
sal_solver_parameters = mumps_solver_parameters

no_normal_flow = {'un': 0.}
up_bcs = {1: no_normal_flow, 2: {}, 3: no_normal_flow, 4: no_normal_flow}
up_solver_parameters = mumps_solver_parameters

up_coupling = [{'pressure': 1}, {'velocity': 0}]

up_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], z, up_fields, up_coupling, dt, up_bcs, solver_parameters=up_solver_parameters)
temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=temp_solver_parameters)
sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)


t = 0.0
step = 0
while t < T - 0.5*dt:

    up_timestepper.advance(t)
    #rho_timestepper.advance(t)
    temp_timestepper.advance(t)
    sal_timestepper.advance(t)
    # find melt rate...


    wb = meltrate(sal,temp,p_)
    print(wb)
    melt.interpolate(wb)

    step += 1
    t += dt

    if step % 1 == 0:
        u_file.write(u_)
        p_file.write(p_)
        #d_file.write(rho)
        t_file.write(temp)
        s_file.write(sal)
        m_file.write(melt)
        print("t=", t)
