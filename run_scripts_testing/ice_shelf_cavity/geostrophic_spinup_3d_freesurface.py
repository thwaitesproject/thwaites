# Geostrophic balance spinup for open box
# Ramp up a body force equivalent to applying a 
# pressure gradient (i.e. a free surface tilt)
# in the positive y direction. This should drive 
# flow to the negative x direction if everything is 
# in geostrophic balance
from thwaites import *
from movement import *
from gadopt.utility import InteriorBC
from firedrake.petsc import PETSc
from firedrake import FacetNormal
import numpy as np
from pyop2.profiling import timed_stage
import argparse
from mpi4py import MPI
##########

parser = argparse.ArgumentParser()
parser.add_argument("--nx", default=10, type=int, help="Number of horizontal cells", required=False)
parser.add_argument("--nz", default=10, type=int, help="Number of vertical layers", required=False)
parser.add_argument("--T", default=864000, type=float, help="End of simulation (s)", required=False)
parser.add_argument("--friction", action='store_true', help="Apply friction drag at ice base")
parser.add_argument("--output", action='store_true', help="Apply friction drag at ice base")
parser.add_argument("--testing", action='store_true', help="Write limits of u for testing")
args = parser.parse_args()

# define time steps
dt = 3600
T = args.T
output_dt = 86400
output_step = output_dt/dt

mesh = BoxMesh(args.nx, args.nx, args.nz, 50e3, 50e3, 100)
mesh.coordinates.dat.data[:, 2] -= 500

mover = LaplacianSmoother(mesh, dt)

x, y, z = SpatialCoordinate(mover.mesh)
PETSc.Sys.Print("Mesh dimension ", mover.mesh.geometric_dimension)
# Set up function spaces
V = VectorFunctionSpace(mover.mesh, "DG", 1)  # velocity space
W = FunctionSpace(mover.mesh, "CG", 2)  # pressure space
M = MixedFunctionSpace([V, W])

K = FunctionSpace(mover.mesh, "DG", 1, variant='equispaced')    # temperature space
P1vec = VectorFunctionSpace(mover.mesh, "CG", 1)

PETSc.Sys.Print("velocity dofs:", V.dim())
PETSc.Sys.Print("Pressure dofs:", W.dim())
PETSc.Sys.Print("scalar dofs:", K.dim())
##########

# Set up functions
m = Function(M)
v_, p_ = m.subfunctions  # function: velocity, pressure
v, p = split(m)  # expression: velocity, pressure
v_.rename("velocity")
p_.rename("pressure")
p_b = Function(W, name="balance pressure")
temp = Function(K, name="temperature")
sal = Function(K, name="salinity")

mesh_vel = Function(V, name="mesh velocity")

##########
f = Constant(-1.409E-4)
ramp = Constant(0.0)
horizontal_stress = -f * 0.01  # geostrophic stress ~ |f v| drives a flow of 0.01 m/s

# Assign Initial conditions
v_init = zero(mover.mesh.geometric_dimension)
v_.interpolate(v_init)
T_restore = -2.0
S_restore = 34.5
temp.assign(T_restore)
sal.assign(S_restore)


##########

# Set up equations
mom_eq = MomentumEquation(M.sub(0), M.sub(0))
cty_eq = ContinuityEquation(M.sub(1), M.sub(1))
temp_eq = ScalarAdvectionDiffusionEquation(K, K)
sal_eq = ScalarAdvectionDiffusionEquation(K, K)
balance_pressure_eq = BalancePressureEquation(W,W)
##########

# Terms for equation fields

# momentum source: the buoyancy term Boussinesq approx. From Jordan etal 14
T_ref = Constant(-2.0)
S_ref = Constant(34.5)
beta_temp = Constant(3.87E-5)
beta_sal = Constant(7.86E-4)
g = Constant(9.81)

rho_perb = -beta_temp*(temp - T_ref) + beta_sal * (sal - S_ref)  # Linear eos (already divided by rho0)
mom_source = as_vector((0., 0, -g)) * rho_perb
# coriolis frequency f-plane assumption at 75deg S. f = 2 omega sin (lat) = 2 * 7.2921E-5 * sin (-75 *2pi/360)

horizontal_source = as_vector((0.0, horizontal_stress, 0.0)) * ramp 

kappa = as_tensor([[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]])
mu = as_tensor([[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]])

kappa_temp = kappa
kappa_sal = kappa

# Equation fields
vp_coupling = [{'pressure': 1}, {'velocity': 0}]
vp_fields = {'viscosity': mu, 'source': mom_source+horizontal_source, 'coriolis_frequency': f, 'balance_pressure': p_b, 'mesh_velocity': mesh_vel}
temp_fields = {'diffusivity': kappa_temp, 'velocity': v, 'mesh_velocity': mesh_vel}
sal_fields = {'diffusivity': kappa_sal, 'velocity': v, 'mesh_velocity': mesh_vel}
p_b_fields = {'buoyancy': mom_source, 'mesh_velocity': mesh_vel} 

##########

# Boundary conditions
no_normal_flow = 0.

vp_bcs = {6: {'free_surface': None},
          5: {'un': no_normal_flow}}

if args.friction:
    vp_bcs[6].update({'drag': 2.5e-3})

temp_bcs = {1: {'qadv': T_restore}, 2: {'qadv': T_restore}, 3: {'qadv': T_restore}, 4: {'qadv': T_restore}}
sal_bcs = {1:{'qadv': S_restore}, 2:{'qadv': S_restore}, 3:{'qadv': S_restore}, 4:{'qadv': S_restore}}

p_b_bcs = {}

##########

# Solver parameters
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
            'schur_ksp_rtol': 1e-8,
            'schur_ksp_atol': 1e-10,
            'schur_ksp_converged_reason': None,
            'schur_pc_type': 'gamg',
            'schur_pc_gamg_threshold': 0.01
            },
        }

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
        'ksp_max_it': 1000,
        }

predictor_solver_parameters_tight = {
        'snes_monitor': None,
        'snes_type': 'ksponly',
        'ksp_type': 'gmres',
        'pc_type': 'hypre',
        'ksp_converged_reason': None,
        'ksp_rtol': 1e-7,
        'ksp_atol': 1e-9,
        'ksp_max_it': 300,
        }
vp_solver_parameters = pressure_projection_solver_parameters
temp_solver_parameters = gmres_solver_parameters
sal_solver_parameters = gmres_solver_parameters
p_b_solver_parameters = predictor_solver_parameters_tight

##########


##########


#


# Set up time stepping routines

vp_timestepper = PressureProjectionTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
                                                          solver_parameters=vp_solver_parameters,
                                                          predictor_solver_parameters=predictor_solver_parameters,
                                                          picard_iterations=1)

# Project pressure (free surface) into a P1 vector function 
# for mesh movement
fs_top = Function(mover.coord_space)

phi = TestFunction(P1vec)
fs_trial = TrialFunction(P1vec)
a = dot(phi, fs_trial) * ds(6)
L = dot(phi, (1/(g*dt)) * (vp_timestepper.solution[2]-vp_timestepper.solution_old[2]) * as_vector((0, 0, 1))) * ds(6)

# Need to set interior nodes to zero to prevent singular matrix
# (projection is only defined on top surface)
interior_null_bc = InteriorBC(P1vec, as_vector([0., 0., 0.]), [6])
fs_proj_problem = LinearVariationalProblem(a, L, fs_top,
                                           bcs=interior_null_bc,
                                           constant_jacobian=False)

fs_proj_solver = LinearVariationalSolver(fs_proj_problem)
fs_proj_solver.solve()

def update_boundary_velocity(t):
    fs_proj_solver.solve()
    mesh_vel.interpolate(mover.v)

moving_boundary = DirichletBC(mover.coord_space, fs_top, 6)
fixed_boundaries = DirichletBC(mover.coord_space, 0, [1, 2, 3, 4, 5])
boundary_conditions = (fixed_boundaries, moving_boundary)

p_b_solver = BalancePressureSolver(balance_pressure_eq, p_b, p_b_fields,p_b_bcs, solver_parameters=p_b_solver_parameters,
                                    p_b_nullspace=VectorSpaceBasis(constant=True))
p_b_solver.advance(0)
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
folder = f"geostrophic_balance_friction{args.friction}/"

###########

# Output files for velocity, pressure, temperature and salinity
if args.output:
    v_file = VTKFile(folder+"vw_velocity.pvd")
    v_file.write(v_)

    p_file = VTKFile(folder+"pressure.pvd")
    p_file.write(p_)

    p_b_file = VTKFile(folder+"balance_pressure.pvd")
    p_b_file.write(p_b)
########

# Add limiter for DG functions
limiter = VertexBasedP1DGLimiter(K)

    

# Begin time stepping
t = 0.0
step = 0

while t < T - 0.5*dt:
    with timed_stage('velocity-pressure'):
        p_b_solver.advance(t)
        vp_timestepper.advance(t)
    with timed_stage('temperature'):
        temp_timestepper.advance(t)
    with timed_stage('salinity'):
        sal_timestepper.advance(t)
    step += 1
    t += dt
    print('assemble diff p', assemble((1/(g*dt)) * (vp_timestepper.solution[2]-vp_timestepper.solution_old[2]) * ds(6)))
    # Move the mesh and calculate the displacement
    mover.move(
        t,
        update_boundary_velocity=update_boundary_velocity,
        boundary_conditions=boundary_conditions,
    )

    limiter.apply(sal)
    limiter.apply(temp)

    if t <= 86400:
        ramp.assign(t/(86400))
    if args.output:
        with timed_stage('output'):
           if step % output_step == 0:
               # dumb checkpoint for starting from last timestep reached
               with CheckpointFile(folder+"dump.h5", 'w') as chk:
                   # Checkpoint file open for reading and writing
                   chk.save_mesh(mover.mesh)
                   chk.save_function(v_, name="velocity")
                   chk.save_function(p_, name="pressure")
                   chk.save_function(temp, name="temperature")
                   chk.save_function(sal, name="salinity")
        
               # Write out files
               v_file.write(v_)
               p_file.write(p_)
               p_b_file.write(p_b)
               time_str = str(step)
        
               PETSc.Sys.Print("t=", t)
    
# Write out final vel max,min limits to check geostrophic balance
def ui_lims(u, i):
    ui_data = u.dat.data_ro[:, i]
    ui_max =  u.comm.allreduce(ui_data.max(initial=-np.inf), MPI.MAX)
    ui_min =  u.comm.allreduce(ui_data.min(initial=np.inf), MPI.MIN)

    return ui_max, ui_min

if args.testing:
    u_max, u_min = ui_lims(v_, 0)
    v_max, v_min = ui_lims(v_, 1)
    w_max, w_min = ui_lims(v_, 2)

    test = [u_max, u_min, v_max, v_min, w_max, w_min]
    
    with open('geostrophic_spinup_3d_test.log', 'w') as f:
        for line in test:
            f.write(f"{line}\n")
