# ISOMIP+ setup 2d slice with extuded mesh
#Buoyancy driven overturning circulation
# beneath ice shelf.
from thwaites import *
from thwaites.utility import CombinedSurfaceMeasure
from firedrake.petsc import PETSc
from firedrake import FacetNormal
import pandas as pd
import argparse
import numpy as np
from math import ceil
from pyop2.profiling import timed_stage
##########

PETSc.Sys.popErrorHandler()

parser = argparse.ArgumentParser()
parser.add_argument("date", help="date format: dd.mm.yy")
args = parser.parse_args()

##########

#  Generate mesh
L = 100.0
dy = 25.0 
ny = round(L/dy)
dz = 2.0 
nz = L / dz

# create mesh
base_mesh = PeriodicIntervalMesh(ny, L)
mesh = ExtrudedMesh(base_mesh, nz, layer_height=dz)
x, z = SpatialCoordinate(mesh)

ds = CombinedSurfaceMeasure(mesh, 5)

PETSc.Sys.Print("Mesh dimension ", mesh.geometric_dimension())

# Set ocean surface
mesh.coordinates.dat.data[:, 1] -= 100 

print("You have Comm WORLD size = ", mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)

##########
# Set up function spaces
v_ele = FiniteElement("RTCE", mesh.ufl_cell(), 2, variant="equispaced")
#v_ele = FiniteElement("DQ", mesh.ufl_cell(), 1, variant="equispaced")
V = FunctionSpace(mesh, v_ele) # Velocity space
W = FunctionSpace(mesh, "Q", 2)  # pressure space
M = MixedFunctionSpace([V, W])

# u velocity function space.
ele = FiniteElement("DQ", mesh.ufl_cell(), 1, variant="equispaced")
U = FunctionSpace(mesh, ele)
VDG = VectorFunctionSpace(mesh, "DQ", 2) # velocity for output

##########

# Set up functions
m = Function(M)
v_, p_ = m.split()  # function: velocity, pressure
v, p = split(m)  # expression: velocity, pressure
v_._name = "velocity"
p_._name = "perturbation pressure"
vdg = Function(VDG, name="velocity")

u = Function(U, name="u_velocity")
##########

# Define a dump file

dump_file = "/data/3d_isomip_plus/first_tests/02.10.20_3d_HJ99_gammafric_dt1800.0_dtOutput86400.0_T8640000.0_ip50.0_constantTres86400.0_KMuh6.0_Muv0.01_Kv0.01_dxy4km_18layers_gl_wall_60m_closed_iterative_initial_solve_coriolis_hypre_pc_press_corr/dump.h5" 
DUMP = False
if DUMP:
    with DumbCheckpoint(dump_file, mode=FILE_UPDATE) as chk:
        # Checkpoint file open for reading and writing
        chk.load(v_, name="v_velocity")
        chk.load(p_, name="perturbation_pressure")
        chk.load(u, name="u_velocity")
else:
    # Assign Initial conditions
    v_init = zero(mesh.geometric_dimension())
    v_.assign(v_init)

    u_init = Constant(0.0)
    u.interpolate(u_init)

##########

# Set up equations
qdeg = 10

mom_eq = MomentumEquation(M.sub(0), M.sub(0), quad_degree=qdeg)
cty_eq = ContinuityEquation(M.sub(1), M.sub(1), quad_degree=qdeg)
u_eq = ScalarVelocity2halfDEquation(U, U)

##########

# Terms for equation fields

# momentum source: the buoyancy term Boussinesq approx. 


g = Constant(9.81)
# since the density is constant we don't need a bouyancy term. (also means we don't need a stress bc on the side walls
# n.b our formulation is rho'* g where rho' is rho - rho0.
mom_source = as_vector((0.,0.0))  

# coriolis frequency f plane assumption
f = Constant(1.032E-4)

# set Viscosity/diffusivity (m^2/s)
mu_h = Constant(100.0)
mu_v = Constant(0.014)
mu = as_tensor([[mu_h, 0], [0, mu_v]])

##########

# Equation fields
vp_coupling = [{'pressure': 1}, {'velocity': 0}]
vp_fields = {'viscosity': mu, 'source': mom_source,
             'coriolis_frequency': f, 'u_velocity': u}
u_fields = {'diffusivity': mu, 'velocity': v, 'coriolis_frequency': f}
##########

# Boundary conditions
# from fluidity
wind_stress = as_vector((1.775E-4, 0))
vp_bcs = {"top": {'stress': wind_stress}, "bottom": {'u': as_vector((0.0, 0.0))}}
u_bcs = {"bottom": {'q': 0.0}}

# STRONGLY Enforced BCs
strong_bcs = []

##########

# Solver parameters
mumps_solver_parameters = {
    'snes_monitor': None,
    'snes_type': 'ksponly',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    "mat_mumps_icntl_14": 200,
    'mat_type': 'aij',
    'snes_max_it': 100,
    'snes_rtol': 1e-8,
    'snes_atol': 1e-6,
}

pressure_projection_solver_parameters = {
        'snes_type': 'ksponly',
        'snes_monitor': None,
        'ksp_type': 'preonly',  # we solve the full schur complement exactly, so no need for outer krylov
        'mat_type': 'matfree',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_fact_type': 'full',
        # velocity mass block:
        'fieldsplit_0': {
            'ksp_converged_reason': None,
            'ksp_type': 'cg',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_pc_type': 'bjacobi',
            'assembled_sub_pc_type': 'sor',
            },
        # schur system: explicitly assemble the schur system
        # this only works with pressureprojectionicard if the velocity block is just the mass matrix
        # and if the velocity is DG so that this mass matrix can be inverted explicitly
        'fieldsplit_1': {
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'thwaites.LaplacePC',
            #'schur_ksp_converged_reason': None,
            'laplace_pc_type': 'ksp',
            'laplace_ksp_ksp_type': 'cg',
            'laplace_ksp_ksp_rtol': 1e-7,
            'laplace_ksp_ksp_atol': 1e-9,
            'laplace_ksp_ksp_converged_reason': None,
            'laplace_ksp_pc_type': 'python',
            'laplace_ksp_pc_python_type': 'thwaites.VerticallyLumpedPC',
        }
    }
if False:
    fs1 = pressure_projection_solver_parameters['fieldsplit_1']
    fs1['ksp_type'] = 'cg'
    fs1['ksp_rtol'] = 1e-7
    fs1['ksp_atol'] = 1e-9
    fs1['ksp_monitor_true_residual'] = None
    fs1['laplace_ksp_ksp_type'] = 'preonly'

predictor_solver_parameters = {
        'snes_monitor': None,
        'snes_type': 'ksponly',
        'ksp_type': 'gmres',
        'pc_type': 'hypre',
        'ksp_converged_reason': None,
        'ksp_rtol': 1e-5,
        'ksp_max_it': 300,
        }

gmres_solver_parameters = {
        'snes_monitor': None,
        'snes_type': 'ksponly',
        'ksp_type': 'gmres',
        'pc_type': 'sor',
        'ksp_converged_reason': None,
        'ksp_rtol': 1e-5,
        'ksp_max_it': 300,
        }



vp_solver_parameters = pressure_projection_solver_parameters
u_solver_parameters = gmres_solver_parameters

##########



# define time steps
dt = 60.
T = 3600*24*10    
output_dt = 60.
output_step = output_dt/dt

##########

# Set up time stepping routines

vp_timestepper = PressureProjectionTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
                                                          solver_parameters=vp_solver_parameters,
                                                          predictor_solver_parameters=predictor_solver_parameters,
                                                          picard_iterations=1)

# performs pseudo timestep to get good initial pressure
# this is to avoid inconsistencies in terms (viscosity and advection) that
# are meant to decouple from pressure projection, but won't if pressure is not initialised
# do this here, so we can see the initial pressure in pressure_0.pvtu
if not DUMP:
    # should not be done when picking up
    with timed_stage('initial_pressure'):
        vp_timestepper.initialize_pressure()

u_timestepper = DIRK33(u_eq, u, u_fields, dt, u_bcs, solver_parameters=u_solver_parameters)

##########

# Set up Vectorfolder
folder = "/data/ekman2.5D/extruded_meshes/"+str(args.date)+"_2.5d_ekman+_dt"+str(dt)+\
         "_dtOut"+str(output_dt)+"_T"+str(T)+"_ipdef_Muh"+str(mu_h.values()[0])+"_Muv"+str(mu_v.values()[0])


###########

# Output files for velocity, pressure, temperature and salinity
v_file = File(folder+"velocity.pvd")
v_file.write(v_)

# Output files for velocity, pressure, temperature and salinity
vdg.project(v_) # DQ2 velocity for output
vdg_file = File(folder+"dg_velocity.pvd")
vdg_file.write(vdg)

p_file = File(folder+"pressure.pvd")
p_file.write(p_)

u_file = File(folder+"u_velocity.pvd")
u_file.write(u)
##########

# Plotting depth profile.
z_profile = np.linspace(0.0, -95.0, 50)
depth_profile = []
for depth in z_profile:
    depth_profile.append([50.0, depth])


velocity_depth_profile_mp = pd.DataFrame()
velocity_depth_profile_mp['Z_profile'] = z_profile


def depth_profile_to_csv(profile, df, t_str):
    df['U_t_' + t_str] = u.at(profile)
    vw = np.array(v_.at(profile))
    vv = vw[:, 0]
    ww = vw[:, 1]
    df['V_t_' + t_str] = vv
    df['W_t_' + t_str] = ww
    if mesh.comm.rank == 0:
        df.to_csv(folder+"ekman_profile.csv")


depth_profile_to_csv(depth_profile, velocity_depth_profile_mp, '0.0')

##########
# Begin time stepping
t = 0.0
step = 0

while t < T - 0.5*dt:
    with timed_stage('velocity-pressure'):
        vp_timestepper.advance(t)
    with timed_stage('u-velocity'):
        u_timestepper.advance(t)
    
    step += 1
    t += dt

    with timed_stage('output'):
       if step % output_step == 0:
           # dumb checkpoint for starting from last timestep reached
           with DumbCheckpoint(folder+"dump.h5", mode=FILE_UPDATE) as chk:
               # Checkpoint file open for reading and writing
               chk.store(v_, name="v_velocity")
               chk.store(p_, name="perturbation_pressure")
               chk.store(u, name="u_velocity")

           
           # Write out files
           v_file.write(v_)
           vdg.project(v_)  # DQ2 velocity for plotting
           vdg_file.write(vdg)
           p_file.write(p_)
    
           u_file.write(u)
           time_str = str(step)
           depth_profile_to_csv(depth_profile, velocity_depth_profile_mp, step_str)
    
           PETSc.Sys.Print("t=", t)
    
