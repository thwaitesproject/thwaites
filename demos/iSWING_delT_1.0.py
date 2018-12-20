# Buoyancy driven overturning circulation
# beneath ice shelf. Wedge geometry.
# Outside temp forcing 1.0degC above freezing point for walter (S = 34.5) at 1000m depth.
# viscosity = temp diffusivity = sal diffusivity: varies linearly over the domain.
from thwaites import *
from math import pi
import numpy as np
from firedrake.petsc import PETSc
from firedrake import FacetNormal


L = 5000 
nx = 100
nz = 100

dz = 5#float(L)/float(nz)
mesh = Mesh("/data2/wis15/meshes/mesh.msh")#ice_shelf_mesh5.msh") #SquareMesh(nx, nz, L) #Mesh("ice_shelf_mesh3.msh")#


import mpi4py
from mpi4py import MPI
print("You have Comm WORLD size = ",mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)



#mesh = Mesh("./meshes/square_mesh_test3.msh")
# We set up a function space of discontinous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
W = FunctionSpace(mesh, "CG", 2)  # pressure space
M = MixedFunctionSpace([V,W])

Q = FunctionSpace(mesh, "DG", 1)  # density space
K = FunctionSpace(mesh,"DG",1)    # temperature space
S = FunctionSpace(mesh,"DG",1)    # salinity space

#M = FunctionSpace(mesh,"DG",1)      # melt space - dont really need but might be easier for plotting in paraview


m = Function(M)
u_, p_ = m.split()


temp = Function(K)
sal = Function(S)

melt = Function(Q)
Q_mixed = Function(Q)
Q_ice = Function(Q)
Q_latent = Function(Q)
Tb = Function(Q)

full_pressure = Function(M.sub(1))




# ice thickness shouldn't be used!!! unless the water depth is not relative to 1km!!!
def ice_thickness(x,x0,y0,x1,y1):
    m = (y1-y0)/(x1-x0)
    return y0 + m*x

def cavity_thickness(x,x0,y0,x1,y1):
    m = (y1-y0)/(x1-x0)
    return y0 + m*x

x,z = SpatialCoordinate(mesh)

h_ice = ice_thickness(x,0.0,999.0,5000.0,900.0)
h_cav = cavity_thickness(x,0.0,1.0,5000.0,100.0)

water_depth = 1000.0

cavity_depth = h_cav - water_depth

z = z -water_depth

u_init = Constant((0.0, 0.0))
u_.assign(u_init)

# the diffusivity and viscosity
#kappa_temp = Constant(2)
#kappa_sal = Constant(2)
#mu = Constant(1e-3)
#mu = Constant(2)  # kinematic

# linearly vary viscosity/diffusivities over domain.
kappa_temp = 3.96E-4*x + 0.02
kappa_sal = 3.96E-4*x + 0.02
mu = 3.96E-4*x + 0.02

rho0=1025  # Holland and jenkins 1997) used later to recalculate the full pressure for meltrate with hydrostatic term

Freezing_temp_at_GL = MeltRateParam(34.5, 0, 0,0,-1000.0,1).freezing_point()
delT = 1.0

Trestore = Freezing_temp_at_GL+delT


temp_init = Constant(Trestore)   # Tinit =Tres = -1.63 deg C from ben's thesis page 54
temp.interpolate(temp_init)


sal_init = Constant(34.4) # stationay flow S = Srest#set this to slightly less than Sres. horizontal pressure gradient increases as depth increases. pressure greater outside domain so drives flow in!
sal.interpolate(sal_init)



folder = "/data2/wis15/outputs/9.12.18.iSWING_3_eq_param_dt_864_t120_delT_1.0/"
# We declare the output filenames, and write out the initial conditions. ::
u_file = File(folder+"velocity.pvd")
u_file.write(u_)
p_file = File(folder+"pressure.pvd")
p_file.write(p_)

t_file = File(folder+"temperature.pvd")
t_file.write(temp)
s_file = File(folder+"salinity.pvd")
s_file.write(sal)



# time period and time step
T = 3600*24*120
dt = 864.



u_test, p_test = TestFunctions(M)


# i think stephan changed this from test to testspace in the last git pull...
mom_eq = MomentumEquation(M.sub(0), M.sub(0))


cty_eq = ContinuityEquation(M.sub(1), M.sub(1))


#rho_test = TestFunction(Q)
#rho_eq = ScalarAdvectionDiffusionEquation(temp_test, Q)

temp_test = TestFunction(K)
#temp_eq = ScalarAdvectionDiffusionEquation(temp_test, K)
temp_eq = ScalarAdvectionDiffusionEquation(K, K)


sal_test = TestFunction(S)
#sal_eq = ScalarAdvectionDiffusionEquation(sal_test, S)
sal_eq = ScalarAdvectionDiffusionEquation(S, S)


u, p = split(m)


#From Ben's thesis page 31

Tref= 0.0
Sref=34.8 #PSU
beta_temp = 3.87*10E-5 #5.0E-3
beta_sal=7.86*10E-4
g = 9.81

# momentum source: the buoyancy term boussinesq approx
mom_source = as_vector((0, -g))*(-beta_temp*(temp-Tref)+beta_sal*(sal-Sref))






# sourcefor scalar equations at open boundary. linearly relaxes to T/Srestore
absorption_factor =2.0E-4


# see above: Trestore = Freezing_temp_at_GL+delT  # delT = Trestore - Tf, where Tf = -2.63  # degC  n.b delta T = 0.1degC  (ben's thesis page

Srestore = 34.5 #PSUi

interp_source_temp = Function(K)
interp_source_sal = Function(S)

source_temp = conditional(x>L*0.96,absorption_factor*((x-L*0.96)/(L*0.04))*Trestore, 0.0)
source_sal = conditional(x>L*0.96,absorption_factor*((x-L*0.96)/(L*0.04))*Srestore, 0.0)

interp_source_temp.interpolate(source_temp)
#source_temp_file = File(folder+"source_temp.pvd")
#source_temp_file.write(interp_source_temp)


interp_source_sal.interpolate(source_sal)



#absorption term
interp_absorp_temp = Function(K)
interp_absorp_sal = Function(S)

absorp_temp = conditional(x>L*0.96,absorption_factor*((x-L*0.96)/(L*0.04)), 0.0)
absorp_sal = conditional(x>L*0.96,absorption_factor*((x-L*0.96)/(L*0.04)), 0.0)

interp_absorp_temp.interpolate(absorp_temp)

#absorb_temp_file = File(folder+"absorb_temp.pvd")
#absorb_temp_file.write(interp_absorp_temp)

interp_absorp_sal.interpolate(absorp_sal)


Q_s = Function(K)


up_fields = {'viscosity': mu, 'source': mom_source}
#rho_fields = {'diffusivity': kappa, 'velocity': u}
temp_fields = {'diffusivity': kappa_temp, 'velocity': u, 'source': interp_source_temp,
               'absorption coefficient': interp_absorp_temp}

sal_fields = {'diffusivity': kappa_sal, 'velocity': u, 'source': interp_source_sal,
              'absorption coefficient': interp_absorp_sal}

mumps_solver_parameters = {
    'snes_monitor': True,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij',
    'snes_max_it': 100,
    'snes_atol': 2.5e-6
}


# get expressions for calculating functions used in meltrate parameterisation
# use 3 eq meltrate param including frictional velocity
q_ice, q_mixed, q_mixed_real, q_latent, q_s, wb, t_b, pfull = MeltRateParam(sal, temp, p_,cavity_depth,z,dz).three_eq_param_meltrate_friction_vel(u)



# assign values of these expressions to functions.
#  each time update the only things that change are T and S
#  so these alter the expression and give new value for functions.
Q_ice.interpolate(q_ice)
Q_mixed.interpolate(q_mixed)

Q_latent.interpolate(q_latent)
Q_s.interpolate(q_s)
melt.interpolate(wb)
Tb.interpolate(t_b)
full_pressure.interpolate(pfull)

# this is where write first file for above functions.
#Q_ice_file = File(folder+"Q_ice.pvd")
#Q_ice_file.write(Q_ice)

#Q_mixed_file = File(folder+"Qmixed.pvd")
#Q_mixed_file.write(Q_mixed)

#Qs_file = File(folder+"Q_s.pvd")
#Qs_file.write(Q_s)

#m_file = File(folder+"melt.pvd")
#m_file.write(melt)


#full_pressure_file = File(folder+"full_pressure.pvd")
#full_pressure_file.write(full_pressure)


########


n = 100
dx = 5000. / float(n)
x1 = np.array([i * dx for i in range(n)])

#y = ice_thickness(x1, 0.0, 999.0, 5000.0, 900.0)
shelf_boundary_points = []
for i in range(n):
    x_i = i * dx
    y_i = cavity_thickness(x_i, 0.0, 1.0, 5000.0, 100.0) - 0.01
    shelf_boundary_points.append([x_i, y_i])

#shelf_boundary_points = np.array(shelf_boundary_points)
import pandas as pd








df = pd.DataFrame()

df['Qice_t_0.0'] = Q_ice.at(shelf_boundary_points)
df['Qmixed_t_0.0'] = Q_mixed.at(shelf_boundary_points)
df['Qlat_t_0.0'] = Q_latent.at(shelf_boundary_points)
df['Qsalt_t_0.0'] = Q_s.at(shelf_boundary_points)
df['Melt_t_0.0'] = melt.at(shelf_boundary_points)
df['Tb_t_0.0']= Tb.at(shelf_boundary_points)
df['P_t_0.0'] = full_pressure.at(shelf_boundary_points)
df['Sal_t_0.0'] = sal.at(shelf_boundary_points)
df['Temp_t_0.0'] = temp.at(shelf_boundary_points)



if mesh.comm.rank ==0:

    print(df)
    #print(len(q_vector))
    #import matplotlib.pyplot as plt
    #plt.plot(x1,q_vector)
    #plt.savefig("q_mixed0.png")
    #Q_ice_vector = Q_ice([shelf_boundary_points])
    #Q_latent_vector = Q_latent([shelf_boundary_points])
    #Tb_vector = Tb([shelf_boundary_points])
    #Melt_vector = melt([shelf_boundary_points])

    df.to_csv(folder+"top_boundary_data.csv")


##########

temp_bcs = {2: {'q': Trestore}, 1: {'flux': -Q_mixed}}
temp_solver_parameters = mumps_solver_parameters


sal_bcs ={2: {'q': Srestore}, 1: {'flux': -Q_s}} # {1: {'q': sal_ice}}
sal_solver_parameters = mumps_solver_parameters





no_normal_flow = 0.

n = FacetNormal(mesh)
stress_open_boundary =  -n*-g*(-beta_temp*(Trestore-Tref)+beta_sal*(Srestore-Sref))*z # p = -g*(alpha(delT)+beta(delS))*z and -n because pressure into the domain! nah its the way stephan defines sigma...

ice_drag = 0.0097

# WEAKLY Enforced BCs
# top boundary: no normal flow, drag flowing over ice
# bottom boundary: no normal flow, drag flowing over bedrock
# grounding line wall (LHS): no normal flow
# open ocean (RHS): pressure to account for density difference to
up_bcs = {1: {'un': no_normal_flow, 'drag': ice_drag}, 2: {'stress': stress_open_boundary},
          3: {'un': no_normal_flow, 'drag': 0.0025}, 4: {'un': no_normal_flow}}

# STRONGLY Enforced BCs
# open ocean (RHS): no tangential flow because viscosity of outside ocean resists vertical flow.
strong_bcs = [DirichletBC(M.sub(0).sub(1), 0, 2)]  #

up_solver_parameters = mumps_solver_parameters

up_coupling = [{'pressure': 1}, {'velocity': 0}]

up_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], m, up_fields, up_coupling, dt, up_bcs,
                                                        solver_parameters=up_solver_parameters, strong_bcs=strong_bcs)
temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=temp_solver_parameters)
sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)




t = 0.0
step = 0

output_dt = 10*3600*24
output_step = output_dt/dt



#T=dt*2#10
while t < T - 0.5*dt:


    up_timestepper.advance(t)

    temp_timestepper.advance(t)
    sal_timestepper.advance(t)

    # Update melt rate functions
    melt.interpolate(wb)
    Q_mixed.interpolate(q_mixed)
    Q_ice.interpolate(q_ice)
    Q_latent.interpolate(q_latent)
    Tb.interpolate(t_b)
    full_pressure.interpolate(pfull)
    Q_s.interpolate(q_s)




    step += 1
    t += dt

    if step % output_step == 0:
        u_file.write(u_)
        p_file.write(p_)
        #d_file.write(rho)
        t_file.write(temp)
        s_file.write(sal)
        #m_file.write(melt)
        #Q_mixed_file.write(Q_mixed)
        #absorb_temp_file.write(interp_absorp_temp)
        #source_temp_file.write(interp_source_temp)
        #full_pressure_file.write(full_pressure)
        #Qs_file.write(Q_s)
        #Q_ice_file.write(Q_ice)
        t_str = str(t/(24.*3600.))
        df['Qice_t_'+t_str] = Q_ice.at(shelf_boundary_points)
        df['Qmixed_t_'+t_str] = Q_mixed.at(shelf_boundary_points)
        df['Qlat_t_'+t_str] = Q_latent.at(shelf_boundary_points)
        df['Qsalt_t_'+t_str] = Q_s.at(shelf_boundary_points)
        df['Melt_t'+t_str] = melt.at(shelf_boundary_points)
        df['Tb_t_'+t_str]= Tb.at(shelf_boundary_points)
        df['P_t_'+t_str] = full_pressure.at(shelf_boundary_points)
        df['Sal_t_' + t_str] = sal.at(shelf_boundary_points)
        df['Temp_t_' + t_str] = temp.at(shelf_boundary_points)


        if mesh.comm.rank ==0:
            df.to_csv(folder+"top_boundary_data.csv")


        with DumbCheckpoint(folder + "dump.h5", mode=FILE_UPDATE) as chk:
            chk.store(sal, name="Salinity")
            chk.store(temp, name="Temperature")
            chk.store(u_, name="Velocity")
            chk.store(p_, name="Pressure")

        PETSc.Sys.Print("t=", t)

