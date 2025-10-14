# Rayleigh-Taylor instability. see fluidity document for incomplete description.
# notebook has set up with derivation of non dimensional form of momentum equation.

from thwaites import *
from math import pi
import numpy as np
d=1
mesh = PeriodicRectangleMesh(100, 20,10,2,direction="x") #Mesh("ice_shelf_mesh3.msh")#

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
u_, p_ = z.subfunctions
rho = Function(Q)
#temp = Function(K)
#sal = Function(S)


# We set up the initial velocity field using a simple analytic expression. ::

x,y = SpatialCoordinate(mesh)





u_init = Constant((0.4, 0.0))
u_.assign(u_init)

# the diffusivity and viscosity
kappa = Constant(2)  # no diffusion of density /temp


#kappa_temp = Constant(1e-3)
#kappa_sal = Constant(1e-3)
#mu = Constant(1e-3)
 # need to change this for reynolds number.... Set to 1.0 so should not matter!!!!

#Reynolds_number = Constant(200.0)
mu = Constant(1)


rho_init = Constant(1.0)
rho.interpolate(rho_init)


u_comp = Function(Q)
u_comp.interpolate(u_[0])

folder = "/data/thwaites/5.12.18.bottom_friction_drag_test_periodic/"
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
T = 8000.
dt = 0.25#T/800.

#T = 5.
#dt = T/5000.




u_test, p_test = TestFunctions(Z)

mom_eq = MomentumEquation(Z.sub(0), Z.sub(0))

cty_eq = ContinuityEquation(Z.sub(1), Z.sub(1))


rho_test = TestFunction(Q)
rho_eq = ScalarAdvectionDiffusionEquation(Q, Q)



u, p = split(z)






up_fields = {'viscosity': mu}
rho_fields = {'diffusivity': kappa, 'velocity': u}


mumps_solver_parameters = {
    'snes_monitor': True,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij',
    'snes_max_it': 20,
    'snes_atol': 2.0e-7
}
# weakly applied dirichlet bcs on top and bottom for density

# 1: lhs, 2: rhs, 3: bottom, 4: top



rho_solver_parameters = mumps_solver_parameters

#u_in = {'u': as_vector((1,0))}
no_normal_flow = {'un': 0.}
no_normal_no_slip_flow = {'u': as_vector((0,0))}

wind_stress = {'stress': as_vector((6.25E-4,0))}


drag =  {'drag': 0.0025,}

up_bcs = {1: {'drag': 0.0025, 'un': 0.}, 2:  {'stress': as_vector((6.25E-4,0)),'un': 0.}}
up_solver_parameters = mumps_solver_parameters


up_coupling = [{'pressure': 1}, {'velocity': 0}]

up_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], z, up_fields, up_coupling, dt, up_bcs, solver_parameters=up_solver_parameters)
#rho_timestepper = DIRK33(rho_eq, rho, rho_fields, dt, rho_bcs, solver_parameters=rho_solver_parameters)

#temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=temp_solver_parameters)
#sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)


n = 20
dy = 0.1
y1 = np.array([i * dy for i in range(n)])

#y = ice_thickness(x1, 0.0, 999.0, 5000.0, 900.0)
vert_profile = []
for i in range(n):
    x_i = 7.5
    y_i = y1[i]

    vert_profile.append([x_i, y_i])

#shelf_boundary_points = np.array(shelf_boundary_points)
import pandas as pd


df = pd.DataFrame()


df['u_dt_0'] = u_comp.at(vert_profile)


if mesh.comm.rank ==0:

    print(df['u_dt_0'])
    df.to_csv(folder+"u_vel_profile.csv")

t = 0.0
step = 0

output_dt = 500
output_step = output_dt/dt


while t < T - 0.5*dt:

    up_timestepper.advance(t)
    #rho_timestepper.advance(t)
    #temp_timestepper.advance(t)
    #sal_timestepper.advance(t)

    step += 1
    t += dt

    if step % output_step == 0:
        u_file.write(u_)
        p_file.write(p_)
        #d_file.write(rho)
        u_comp.interpolate(u_[0])
        t_str = str(t)
        df['u_dt_' + t_str] = u_comp.at(vert_profile)
        if mesh.comm.rank ==0:

            df.to_csv(folder + "u_vel_profile.csv")
        print("t=", t)



#import numpy as np

n = 20
dy = 0.1
y1 = np.array([i * dy for i in range(n)])

#import matplotlib.pyplot as plt
#import pandas as pd

#df = pd.read_csv("/data/thwaites/5.12.18.bottom_friction_drag_test_periodic/u_vel_profile.csv")

if mesh.comm.rank ==0:
    for i in range(1,11):
        time_str = str(i*500)
        plt.plot(df['u_dt_'+time_str+'.0'],y1,label="time = "+time_str+"s")
    plt.xlabel("horizontal velocity / m/s")
    plt.ylabel("Height from bed")
    plt.legend()
    plt.title("Bottom friction drag test, CD = 0.0025, Wind Stress = 6.25E-4")
    plt.show()

