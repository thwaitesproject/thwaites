# Buoyancy driven overturning circulation now with idealised ice-shelf geometry
#
from thwaites import *
from math import pi
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



z = Function(Z)
u_, p_ = z.split()
rho = Function(Q)
temp = Function(K)
sal = Function(S)


# We set up the initial velocity field using a simple analytic expression. ::

x,y = SpatialCoordinate(mesh)

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




folder = "./25.08.10.sq_box.source_test1/"
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
mom_source = as_vector((0, -9.81))*(-beta_temp*(temp-Tref)+beta_sal*(sal-Sref))  # momentum source: the buoyancy term boussinesq approx



#####T= #DG_2d = FunctionSpace(mesh2d, 'DG', 1)
#x = SpatialCoordinate(mesh)
#print(len(x))
print(x)

#print(len(y))
print(y)



# sourcefor scalar equations at open boundary. linearly relaxes to T/Srestore
absorption_factor =2.0E-4

#Trestore = 0.37  # degC  n.b delta T = 3degC  (ben's thesis page
Trestore = -0.5

Srestore = 34.5 #PSU

interp_source_temp = Function(K)
interp_source_sal = Function(S)

source_temp = conditional(y>0.75,absorption_factor*Trestore/((y-0.75)/0.25), 0.0)
source_sal = conditional(y>0.75,absorption_factor*Trestore/((y-0.75)/0.25), 0.0)

interp_source_temp.interpolate(source_temp)
interp_source_sal.interpolate(source_sal)
#x_vector, y_vector = interpolate(x[0], Function(DG_2d)).dat.data, interpolate(x[1], Function(DG_2d)).dat.data

#absorption term
interp_absorp_temp = Function(K)
interp_absorp_sal = Function(S)

absorp_temp = conditional(y>0.75,(absorption_factor*(y-0.75)/0.25), 0.0)
absorp_sal = conditional(y>0.75,(absorption_factor*(y-0.75)/0.25), 0.0)

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
temp_ice = -25.0
#temp_openocean = 1
temp_bottom = 3.0
temp_bcs = {3: {'q': temp_bottom}} #{3: {'q': temp_ice}}#, 3: {'q': temp_bottom}, 4: {'q': temp_top}}
temp_solver_parameters = mumps_solver_parameters

# weakly applied dirichlet bcs on top and bottom for sal
sal_ice = 0.0
sal_bottom=45.0
#sal_right = 40.0
sal_bcs ={}# {1: {'q': sal_ice}}
sal_solver_parameters = mumps_solver_parameters

no_normal_flow = {'un': 0.}
up_bcs = {2: no_normal_flow, 3: no_normal_flow, 4: no_normal_flow}
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

    step += 1
    t += dt

    if step % 1 == 0:
        u_file.write(u_)
        p_file.write(p_)
        #d_file.write(rho)
        t_file.write(temp)
        s_file.write(sal)
        print("t=", t)
