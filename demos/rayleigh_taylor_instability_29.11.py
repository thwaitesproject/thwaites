# Rayleigh-Taylor instability. see fluidity document for incomplete description.
# notebook has set up with derivation of non dimensional form of momentum equation.
#
from thwaites import *
from math import pi

d=1
mesh = RectangleMesh(50, 400,0.5,4) #Mesh("ice_shelf_mesh3.msh")#

import mpi4py
from mpi4py import MPI
print("You have Comm WORLD size = ",mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)



#mesh = Mesh("./meshes/square_mesh_test3.msh")
# We set up a function space of discontinous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
W = FunctionSpace(mesh, "CG", 2)  # pressure space
Z = MixedFunctionSpace([V,W])

Q = FunctionSpace(mesh, "DG", 1)  # density space


#K = FunctionSpace(mesh,"DG",1)    # temperature space
#S = FunctionSpace(mesh,"DG",1)    # salinity space



z = Function(Z)
u_, p_ = z.split()
rho = Function(Q)
#temp = Function(K)
#sal = Function(S)


# We set up the initial velocity field using a simple analytic expression. ::

x,y = SpatialCoordinate(mesh)


# convert y: 0->4 to -2->2
y = y-2

u_init = Constant((0.0, 0.0))
u_.assign(u_init)

# the diffusivity and viscosity
kappa = Constant(0)  # no diffusion of density /temp


#kappa_temp = Constant(1e-3)
#kappa_sal = Constant(1e-3)
#mu = Constant(1e-3)
 # need to change this for reynolds number.... Set to 1.0 so should not matter!!!!

Reynolds_number = Constant(1000.0)
mu = Constant(1/Reynolds_number)
#bell_r0 = 40*0.15; bell_x0 = 40*0.25; bell_y0 = 40*0.5


#cone_r0 = 40*0.15; cone_x0 = 40*0.5; cone_y0 = 40*0.25
#cyl_r0 = 40*0.15; cyl_x0 = 40*0.5; cyl_y0 = 40*0.75
#slot_left = 40*0.475; slot_right = 40*0.525; slot_top = 40*0.85


def eta(x,d):
    return -0.1*d*cos((2*pi*x)/d)


# the tracer function and its initial condition
# change from rho to temp...
# these numbers come from At = (rho_max - rho_min)/(rho_max+rho_min) = 0.75
# and dimensionalising N.S (see notebook - monday 28th october)
# rho_min is used to non dimensionalise rho. so rho_min = 1 and rho_max = 7

At = 0.5  # must be less than 1.0

rho_min=1.0 # dimensionless rho_min
rho_max = rho_min*(1.0+At)/(1.0-At)


rho_init = 0.5*(rho_max+rho_min) + 0.5*(rho_max-rho_min)*tanh((y-eta(x,d))/(0.01*d))
rho.interpolate(rho_init)

#temp_init = Constant(0.0)+slot_cyl   # deg C from ben's thesis page 54
#temp.interpolate(temp_init)


#sal_init = Constant(34.4) # PSU  from ben's thesis page 54
#sal.interpolate(sal_init)




folder = "/data/thwaites/29.11.18.rayleigh_taylor_inst_open1/"
# We declare the output filenames, and write out the initial conditions. ::
u_file = File(folder+"velocity.pvd")
u_file.write(u_)
p_file = File(folder+"pressure.pvd")
p_file.write(p_)
d_file = File(folder+"density.pvd")
d_file.write(rho)
#t_file = File(folder+"temperature.pvd")
#t_file.write(temp)
#s_file = File(folder+"salinity.pvd")
#s_file.write(sal)


# time period and time step
T = 5.
dt = 0.01#T/800.

#T = 5.
#dt = T/5000.




u_test, p_test = TestFunctions(Z)

#mom_eq = MomentumEquation(u_test, Z.sub(0))
# i think stephan changed this from test to testspace in the last git pull...
mom_eq = MomentumEquation(Z.sub(0), Z.sub(0))

#cty_eq = ContinuityEquation(p_test, Z.sub(1))
cty_eq = ContinuityEquation(Z.sub(1), Z.sub(1))


rho_test = TestFunction(Q)
rho_eq = ScalarAdvectionDiffusionEquation(Q, Q)

#temp_test = TestFunction(K)
#temp_eq = ScalarAdvectionDiffusionEquation(temp_test, K)
#temp_eq = ScalarAdvectionDiffusionEquation(K, K)


#sal_test = TestFunction(S)
#sal_eq = ScalarAdvectionDiffusionEquation(sal_test, S)
#sal_eq = ScalarAdvectionDiffusionEquation(S, S)


u, p = split(z)

#source = as_vector((0, -1.0))*rho  # momentum source: the buoyancy term

#From Ben's thesis page 31
#rho_0 = 1.0
#Tref= 0.0
#Sref=34.8 #PSU
#beta_temp = 3.87*10E-5 #5.0E-3
#beta_sal=7.86*10E-4


# g = -1.

rho_mean = 0.5*(rho_min+rho_max)
mom_source = as_vector((0, -1.0))*(rho-rho_min) # momentum source: the buoyancy term boussinesq approx







up_fields = {'viscosity': mu, 'source': mom_source}
rho_fields = {'diffusivity': kappa, 'velocity': u}
#temp_fields = {'diffusivity': kappa_temp, 'velocity': u, 'source': interp_source_temp, 'absorption coefficient': interp_absorp_temp}

#sal_fields = {'diffusivity': kappa_sal, 'velocity': u, 'source': interp_source_sal, 'absorption coefficent': interp_absorp_sal}

mumps_solver_parameters = {
    'snes_monitor': True,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij'
}
# weakly applied dirichlet bcs on top and bottom for density

rho_bcs = {4: {'q': rho_max}}
#rho_bcs = {}
rho_solver_parameters = mumps_solver_parameters


no_normal_flow = {'un': 0.}
no_normal_no_slip_flow = {'u': as_vector((0,0))}

up_bcs = {1: no_normal_flow, 2: no_normal_flow, 3: no_normal_no_slip_flow, 4: {}}#, 4: no_normal_no_slip_flow}
up_solver_parameters = mumps_solver_parameters

up_coupling = [{'pressure': 1}, {'velocity': 0}]

up_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], z, up_fields, up_coupling, dt, up_bcs, solver_parameters=up_solver_parameters)
rho_timestepper = DIRK33(rho_eq, rho, rho_fields, dt, rho_bcs, solver_parameters=rho_solver_parameters)

#temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=temp_solver_parameters)
#sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)


t = 0.0
step = 0

output_dt = 0.5
output_step = output_dt/dt


while t < T - 0.5*dt:

    up_timestepper.advance(t)
    rho_timestepper.advance(t)
    #temp_timestepper.advance(t)
    #sal_timestepper.advance(t)

    step += 1
    t += dt

    if step % output_step == 0:
        u_file.write(u_)
        p_file.write(p_)
        d_file.write(rho)
        #t_file.write(temp)
        #s_file.write(sal)
        print("t=", t)
