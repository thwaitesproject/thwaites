# Buoyancy driven circulation
# beneath ice shelf with idealised basal crevasse.
# See Jordan et al. 2014 and Ben Yeager thesis (2018)
# for further details of the setup.
from thwaites import *
from movement import *
from gadopt.utility import InteriorBC
from firedrake.petsc import PETSc
from pyop2.profiling import timed_stage
from firedrake import FacetNormal
import pandas as pd
import argparse
import numpy as np
##########


parser = argparse.ArgumentParser()
parser.add_argument("date", help="date format: dd.mm.yy")
parser.add_argument("dt", help="time step in seconds",
                    type=float)
parser.add_argument("output_dt", help="output time step in seconds",
                    type=float)
parser.add_argument("T", help="final simulation time in seconds",
                    type=float)
args = parser.parse_args()


ip_factor = Constant(50.)
restoring_time = 86400.

##########

#  Generate mesh
L = 10E3
H1 = 2.
H2 = 102.
dy = 50.0
ny = round(L/dy)
#nz = 50
dz = 1.0

# create mesh
mesh = Mesh("./Crevasse_rounded_refined.msh")
#mesh = SquareMesh(50,50, 100,100) #Mesh("./Crevasse_refined.msh")

##########

dt = args.dt
T = args.T
output_dt = args.output_dt
output_step = output_dt/dt
################

mover = LaplacianSmoother(mesh, dt)


y, z = SpatialCoordinate(mover.mesh)
##########

# Set up function spaces
V = VectorFunctionSpace(mover.mesh, "DG", 1)  # velocity space
W = FunctionSpace(mover.mesh, "CG", 2)  # pressure space
M = MixedFunctionSpace([V, W])

# u velocity function space.
U = FunctionSpace(mover.mesh, "DG", 1)

Q = FunctionSpace(mover.mesh, "DG", 1)  # melt function space
K = FunctionSpace(mover.mesh, "DG", 1)    # temperature space
S = FunctionSpace(mover.mesh, "DG", 1)    # salinity space
P1vec = VectorFunctionSpace(mover.mesh, "CG", 1)

##########

# Set up functions
m = Function(M)
v_, p_ = m.subfunctions  # function: y component of velocity, pressure
v, p = split(m)  # expression: y component of velocity, pressure
v_._name = "v_velocity"
p_._name = "perturbation pressure"

rho = Function(K, name="density")
temp = Function(K, name="temperature")
sal = Function(S, name="salinity")
melt = Function(Q, name="melt rate")

mesh_vel = Function(V, name="mesh velocity")
##########

# Assign Initial conditions
v_init = zero(mover.mesh.geometric_dimension())
v_.assign(v_init)

# baseline T3
T_200m_depth = -1.965


S_surface = 34.34 

T_restore = Constant(T_200m_depth)
S_restore = Constant(S_surface) 

temp_init = T_restore
temp.interpolate(temp_init)

sal_init = Constant(34.34)
sal.interpolate(sal_init)


##########

# Set up equations
mom_eq = MomentumEquation(M.sub(0), M.sub(0))
cty_eq = ContinuityEquation(M.sub(1), M.sub(1))
#u_eq = ScalarVelocity2halfDEquation(U, U)
temp_eq = ScalarAdvectionDiffusionEquation(K, K)
sal_eq = ScalarAdvectionDiffusionEquation(S, S)
##########

# Terms for equation fields

# momentum source: the buoyancy term Boussinesq approx. From Jordan etal 14
T_ref = Constant(-2.0)
S_ref = Constant(34.5)
beta_temp = Constant(3.87E-5)
beta_sal = Constant(7.86E-4)
g = Constant(9.81)
rho0 = 1030.
rho_ice = 920.

rho_perb = -beta_temp*(temp - T_ref) + beta_sal * (sal - S_ref)  # Linear eos (already divided by rho0)
mom_source = as_vector((0., -g)) * rho_perb
rho.interpolate(rho0 * (1 + rho_perb))
# coriolis frequency f-plane assumption at 75deg S. f = 2 omega sin (lat) = 2 * 7.2921E-5 * sin (-75 *2pi/360)
#f = Constant(-1.409E-4)

# Scalar source/sink terms at open boundary.
absorption_factor = Constant(1.0/restoring_time)
sponge_fraction = 0.06  # fraction of domain where sponge
# Temperature source term


kappa = as_tensor([[1e-3, 0], [0, 1e-3]])

kappa_temp = kappa
kappa_sal = kappa
mu = kappa


# Interior penalty term
# 3*cot(min_angle)*(p+1)*p*nu_max/nu_min

ip_alpha = Constant(3*dy/dz*2*ip_factor)
# Equation fields
vp_coupling = [{'pressure': 1}, {'velocity': 0}]
vp_fields = {'viscosity': mu, 'source': mom_source, 'mesh_velocity': mesh_vel} #, 'interior_penalty': ip_alpha}
temp_fields = {'diffusivity': kappa_temp, 'velocity': v, 'mesh_velocity': mesh_vel}
sal_fields = {'diffusivity': kappa_sal, 'velocity': v, 'mesh_velocity': mesh_vel}

##########

# Get expressions used in melt rate parameterisation
mp = ThreeEqMeltRateParam(sal, temp, p, z, velocity=pow(dot(v, v), 0.5), HJ99Gamma=True, ice_heat_flux=False)


##########

# assign values of these expressions to functions.
# so these alter the expression and give new value for functions.
melt.interpolate(mp.wb)


##########

##########

# Boundary conditions
# top boundary: no normal flow, drag flowing over ice
# bottom boundary: no normal flow, drag flowing over bedrock
# grounding line wall (LHS): no normal flow
# open ocean (RHS): pressure to account for density differences

# WEAKLY Enforced BCs
n = FacetNormal(mover.mesh)
Temperature_term = -beta_temp * ((T_restore-T_ref) * z)
Salinity_term = beta_sal * ((S_restore - S_ref) * z) # ((S_bottom - S_surface) * (pow(z, 2) / (-2.0*water_depth)) + (S_surface-S_ref) * z)
stress_open_boundary = -n*-g*(Temperature_term + Salinity_term)
no_normal_flow = 0.
ice_drag = 0.0097


vp_bcs = {4: {'un': no_normal_flow, 'drag': ice_drag}, 2: {'stress': stress_open_boundary}, 
        1: {'un': no_normal_flow}, 3: {'un': -0.025, 'drag': 2.5e-3}}

temp_bcs = {4: {'flux': -mp.T_flux_bc}, 3:{'q': T_restore}}

sal_bcs = {4: {'flux': -mp.S_flux_bc}, 3:{'q': S_restore}}


# STRONGLY Enforced BCs
# open ocean (RHS): no tangential flow because viscosity of outside ocean resists vertical flow.
strong_bcs = []#DirichletBC(M.sub(0).sub(1), 0, 2)]

##########

# Solver parameters
mumps_solver_parameters = {
    'snes_monitor': None,
    'snes_type': 'ksponly',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij',
    'snes_max_it': 100,
    'snes_rtol': 1e-8,
    'snes_atol': 1e-6,
}

pressure_projection_solver_parameters = {
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',  # we solve the full schur complement exactly, so no need for outer krylov
        'mat_type': 'matfree',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_fact_type': 'full',
        # velocity mass block:
        'fieldsplit_0': {
            'ksp_type': 'gmres',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'ksp_converged_reason': None,
            'assembled_ksp_type': 'preonly',
            'assembled_pc_type': 'bjacobi',
            'assembled_sub_pc_type': 'ilu',
            },
        # schur system: explicitly assemble the schur system
        # this only works with pressureprojectionicard if the velocity block is just the mass matrix
        # and if the velocity is DG so that this mass matrix can be inverted explicitly
        'fieldsplit_1': {
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'thwaites.AssembledSchurPC',
            'schur_ksp_type': 'cg',
            'schur_ksp_max_it': 1000,
            'schur_ksp_rtol': 1e-7,
            'schur_ksp_atol': 1e-9,
            'schur_ksp_converged_reason': None,
            'schur_pc_type': 'gamg',
            'schur_pc_gamg_threshold': 0.01
            },
        }

vp_solver_parameters = pressure_projection_solver_parameters
u_solver_parameters = mumps_solver_parameters
temp_solver_parameters = mumps_solver_parameters
sal_solver_parameters = mumps_solver_parameters


# Project melt rate (normal velocity) into a P1 vector function 
# for mesh movement
melt_top = Function(mover.coord_space)

phi = TestFunction(P1vec)
melt_trial = TrialFunction(P1vec)
a = dot(phi, melt_trial) * ds(4)
L = dot(phi,  1e5*mp.wb * n) * ds(4)

# Need to set interior nodes to zero to prevent singular matrix
# (projection is only defined on top surface)
interior_null_bc = InteriorBC(P1vec, as_vector([0., 0.]), [4])
melt_proj_problem = LinearVariationalProblem(a, L, melt_top,
                                           bcs=interior_null_bc,
                                           constant_jacobian=True)

melt_proj_solver = LinearVariationalSolver(melt_proj_problem)
melt_proj_solver.solve()

def update_boundary_velocity(t):
    melt_proj_solver.solve()
    mesh_vel.interpolate(mover.v)

#

moving_boundary = DirichletBC(mover.coord_space, melt_top, 4)
fixed_boundaries = DirichletBC(mover.coord_space, 0, [1, 2, 3])
boundary_conditions = (fixed_boundaries, moving_boundary)

##########

# Set up time stepping routines

vp_timestepper = PressureProjectionTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
                                                          solver_parameters=vp_solver_parameters,
                                                          predictor_solver_parameters=u_solver_parameters,
                                                          picard_iterations=1)

# performs pseudo timestep to get good initial pressure
# this is to avoid inconsistencies in terms (viscosity and advection) that
# are meant to decouple from pressure projection, but won't if pressure is not initialised
# do this here, so we can see the initial pressure in pressure_0.pvtu
with timed_stage('initial_pressure'):
    vp_timestepper.initialize_pressure()

#u_timestepper = DIRK33(u_eq, u, u_fields, dt, u_bcs, solver_parameters=u_solver_parameters)
temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=temp_solver_parameters)
sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)

##########

# Set up folder
folder = "/mnt/c/Users/wills/Documents/ice_ocean/moving_mesh/"+str(args.date)+"_3_eq_param_ufricHJ99_dt"+str(dt)+\
         "_dtOutput"+str(output_dt)+"_T"+str(T)+"_crevasse_moving_alecorr_nolim/"
         #+"_extended_domain_with_coriolis_stratified/"  # output folder.


###########

# Output files for velocity, pressure, temperature and salinity
v_file = VTKFile(folder+"vw_velocity.pvd")
v_file.write(v_)

p_file = VTKFile(folder+"pressure.pvd")
p_file.write(p_)


t_file = VTKFile(folder+"temperature.pvd")
t_file.write(temp)

s_file = VTKFile(folder+"salinity.pvd")
s_file.write(sal)

rho_file = VTKFile(folder+"density.pvd")
rho_file.write(rho)

##########

m_file = VTKFile(folder+"melt.pvd")
m_file.write(melt)
melt_top_file = VTKFile(folder+"melt_top.pvd")
melt_top_file.write(melt_top)
########


########
# Add limiter for DG functions
limiter = VertexBasedP1DGLimiter(S)

# Begin time stepping
t = 0.0
step = 0

while t < T - 0.5*dt:
    with timed_stage('velocity-pressure'):
        vp_timestepper.advance(t)
    with timed_stage('temperature'):
        temp_timestepper.advance(t)
    with timed_stage('salinity'):
        sal_timestepper.advance(t)
    step += 1
    t += dt


    # Move the mesh and calculate the displacement
    mover.move(
        t,
        update_boundary_velocity=update_boundary_velocity,
        boundary_conditions=boundary_conditions,
    )
#    limiter.apply(sal)
  #  limiter.apply(temp)

    print(mover.v.dat.data.max())
    with timed_stage('output'):
       if step % output_step == 0:
    
           # Update melt rate functions
           melt.interpolate(mp.wb)
    
           # Update density for plotting
           rho.interpolate(rho0*(-beta_temp*(temp - T_ref) + beta_sal * (sal - S_ref)) )

           # Write out files
           v_file.write(v_)
           p_file.write(p_)
           t_file.write(temp)
           s_file.write(sal)
           rho_file.write(rho)
           # Write melt rate functions
           m_file.write(melt)
           
           melt_top_file.write(melt_top)
           time_str = str(step)
           
           PETSc.Sys.Print("t=", t)
    
           PETSc.Sys.Print("integrated melt =", assemble(melt * ds(4)))
