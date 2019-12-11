# 2.5D modelling.
# velocity (y component : v) and pressure solved in 2D vertical slice.
# velocity (x component: u) is solved using an advection diffusion equation with a coriolis forcing term.
# use y component for velocity because the wind stress is in the y direction for ekman analytical solution
# this has already been implemented in momentum equation.
from thwaites import *
from firedrake.petsc import PETSc
import pandas as pd
import numpy as np
##########

folder = "/data/ekman2.5D/2.12.19.ekman2.5D.dt60.tau1.775E-4.layers50.ip50.with_coriolis.bottom_un=ut=0.Tfrom_spinup788400.dtoutput60/"  # output folder.

dump_file = "/data/ekman2.5D/2.12.19.ekman2.5D.dt60.tau1.775E-4.layers50.ip50.with_coriolis.bottom_un=ut=0.T10days.dtoutput3600/dump.h5"


##########

mesh = PeriodicSquareMesh(4, 50, 100, direction="x")

y, z = SpatialCoordinate(mesh)

mesh.coordinates.dat.data[:, 1] -= 100

print("You have Comm WORLD size = ", mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)


##########

# Set up function spaces
V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
W = FunctionSpace(mesh, "CG", 2)  # pressure space
M = MixedFunctionSpace([V, W])

# u velocity function space.
U = FunctionSpace(mesh, "DG", 1)

##########

# Set up functions
m = Function(M)
v_, p_ = m.split()
v, p = split(m)

u = Function(U)
##########

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
mom_eq = MomentumEquation(M.sub(0), M.sub(0))
cty_eq = ContinuityEquation(M.sub(1), M.sub(1))
u_eq = ScalarVelocity2halfDEquation(U, U)
##########

# Terms for equation fields
mu_h = 100.0
mu_v = 0.014
mu = as_tensor([[mu_h, 0.0], [0.0, mu_v]])


# momentum source: gravity (divide by pho0
g = 9.81
mom_source = as_vector((0, -g))

# coriolis frequency f-plane assumption
f = 1.032E-4

# Interior penalty term
# 3*cot(min_angle)*(p+1)*p*nu_max/nu_min
# dx/dz = 25/2 = 6.25 so 3*25/2*2 = 75
ip_alpha = 50.0#0.48
# Equation fields
vp_coupling = [{'pressure': 1}, {'velocity': 0}]
vp_fields = {'viscosity': mu, 'source': mom_source, 'interior_penalty': ip_alpha,
             'coriolis_frequency': f, 'u_velocity': u}
u_fields = {'diffusivity': mu, 'velocity': v, 'interior_penalty': ip_alpha, 'coriolis_frequency': f}
##########

# Output files for velocity, pressure, temperature and salinity
v_file = File(folder+"v_w_velocity.pvd")
v_file.write(v_)

p_file = File(folder+"pressure.pvd")
p_file.write(p_)


u_file = File(folder+"u_velocity.pvd")
u_file.write(u)

##########

# Boundary conditions
# from fluidity
wind_stress = as_vector((1.775E-4, 0))
vp_bcs = {2: {'stress': wind_stress}, 1: {'u': as_vector((0.0, 0.0))}}
u_bcs = {1: {'q': 0.0}}

# STRONGLY Enforced BCs


##########

# Solver parameters
mumps_solver_parameters = {
    'snes_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij',
    'snes_max_it': 100,
    'snes_rtol': 1e-8,
}

vp_solver_parameters = mumps_solver_parameters
u_solver_parameters = mumps_solver_parameters

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

# define time steps
dt = 60.
T = 3600*24*10    # This setup converges at T = 791460s (achieved this by 219 3600s outputs then 51 60s outputs)
output_dt = 3600.
output_step = output_dt/dt

##########

# Set up time stepping routines
vp_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
                                                        solver_parameters=vp_solver_parameters)

u_timestepper = DIRK33(u_eq, u, u_fields, dt, u_bcs, solver_parameters=u_solver_parameters)

##########

# Begin time stepping
t = 0.0
step = 0
out_step = 0
while t < T - 0.5*dt:
    vp_timestepper.advance(t)
    u_timestepper.advance(t)

    step += 1
    t += dt

    # Output files
    if step % output_step == 0:
        # dumb checkpoint for starting from spin up
        out_step += 1
        with DumbCheckpoint(folder+"dump.h5", mode=FILE_UPDATE) as chk:
            # Checkpoint file open for reading and writing
            chk.store(v_, name="v_velocity")
            chk.store(p_, name="perturbation_pressure")
            chk.store(u, name="u_velocity")

        # Write outfiles
        v_file.write(v_)
        p_file.write(p_)
        u_file.write(u)
        step_str = str(out_step)
        depth_profile_to_csv(depth_profile, velocity_depth_profile_mp, step_str)

        PETSc.Sys.Print("t=", t)

