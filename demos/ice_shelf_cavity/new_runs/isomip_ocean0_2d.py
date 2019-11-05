# Buoyancy driven overturning circulation
# beneath ice shelf. Wedge geometry. 5km
# Outside temp forcing 3.0degC above freezing point for walter (S = 34.5) at 1000m depth.
# viscosity = temp diffusivity = sal diffusivity: varies linearly over the domain.
from thwaites import *
from thwaites.utility import ice_thickness, cavity_thickness, get_top_boundary
from firedrake.petsc import PETSc
from firedrake import FacetNormal
import pandas as pd
##########

folder = "/data/isomip/5.11.19.isomip_triangle_ocean0/"  # output folder.

##########


H1 = 2.0
H2 = 400
L = 320.0E3
dx = 2.0E3
nx = round(L/dx)
nz = 36

mesh = RectangleMesh(nx, nz, L, 1)

x = mesh.coordinates.dat.data[:, 0]
mesh.coordinates.dat.data[:, 1] = mesh.coordinates.dat.data[:, 1] * (x/L*(H2-H1) + H1)

print("You have Comm WORLD size = ", mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)

x, z = SpatialCoordinate(mesh)

# shift z = 0 to surface of ocean. N.b z = 0 is outside domain.
water_depth = 600.0
z = z - water_depth

##########

# Set up function spaces
V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
W = FunctionSpace(mesh, "CG", 2)  # pressure space
M = MixedFunctionSpace([V, W])

Q = FunctionSpace(mesh, "DG", 1)  # melt function space
K = FunctionSpace(mesh, "DG", 1)    # temperature space
S = FunctionSpace(mesh, "DG", 1)    # salinity space

##########

# Set up functions
m = Function(M)
u_, p_ = m.split()
u, p = split(m)

temp = Function(K)
sal = Function(S)
melt = Function(Q)
Q_mixed = Function(Q)
Q_ice = Function(Q)
Q_latent = Function(Q)
Q_s = Function(Q)
Tb = Function(Q)
Sb = Function(Q)
full_pressure = Function(M.sub(1))

##########

# Assign Initial conditions
u_init = zero(mesh.geometric_dimension())
u_.assign(u_init)

# Isomip+ Ocean zero. Initialise = WARM, Restore = WARM
T_surface = -1.9
T_bottom = 1.0

S_surface = 33.8
S_bottom = 34.7

T_restore = T_surface + (T_bottom - T_surface) * (z / -water_depth)
S_restore = S_surface + (S_bottom - S_surface) * (z / -water_depth)

temp_init = T_restore
temp.interpolate(temp_init)

sal_init = S_restore
sal.interpolate(sal_init)

##########

# Set up equations
mom_eq = MomentumEquation(M.sub(0), M.sub(0))
cty_eq = ContinuityEquation(M.sub(1), M.sub(1))
temp_eq = ScalarAdvectionDiffusionEquation(K, K)
sal_eq = ScalarAdvectionDiffusionEquation(S, S)

##########

# Terms for equation fields

# linearly vary viscosity/diffusivity over domain.
mu = Constant(6)  # kinematic viscosity [mu] = m^2/s
kappa_temp = Constant(6)
kappa_sal = Constant(6)


# momentum source: the buoyancy term Boussinesq approx. From Ben's thesis page 31
T_ref = -1.0
S_ref = 34.2
beta_temp = 3.733*10E-5
beta_sal = 7.843*10E-4
g = 9.81
mom_source = as_vector((0, -g)) * (-beta_temp * (temp-T_ref) + beta_sal * (sal-S_ref))

# Scalar source/sink terms at open boundary.
absorption_factor = 10*24*3600

# Temperature source term
source_temp = conditional(x > 0.96 * L, absorption_factor * ((x - 0.96 * L) / (L * 0.04)) * T_restore, 0.0)

# Salinity source term
source_sal = conditional(x > 0.96 * L, absorption_factor * ((x - 0.96 * L) / (L * 0.04)) * S_restore, 0.0)

# Temperature absorption term
absorp_temp = conditional(x > 0.96 * L, absorption_factor * ((x - 0.96 * L) / (L * 0.04)), 0.0)

# Salinity absorption term
absorp_sal = conditional(x > 0.96 * L, absorption_factor * ((x - 0.96 * L)/(L * 0.04)), 0.0)

# Equation fields
up_coupling = [{'pressure': 1}, {'velocity': 0}]
up_fields = {'viscosity': mu, 'source': mom_source}
temp_fields = {'diffusivity': kappa_temp, 'velocity': u, 'source': source_temp,
               'absorption coefficient': absorp_temp}
sal_fields = {'diffusivity': kappa_sal, 'velocity': u, 'source': source_sal,
              'absorption coefficient': absorp_sal}

##########

# Output files for velocity, pressure, temperature and salinity
u_file = File(folder+"velocity.pvd")
u_file.write(u_)

p_file = File(folder+"pressure.pvd")
p_file.write(p_)

t_file = File(folder+"temperature.pvd")
t_file.write(temp)

s_file = File(folder+"salinity.pvd")
s_file.write(sal)

##########

# Get expressions used in melt rate parameterisation
mp = ThreeEqMeltRateParam(sal, temp, p_, z, u_)

##########

# assign values of these expressions to functions.
# so these alter the expression and give new value for functions.
Q_ice.interpolate(mp.Q_ice)
Q_mixed.interpolate(mp.Q_mixed)
Q_latent.interpolate(mp.Q_latent)
Q_s.interpolate(mp.S_flux_bc)
melt.interpolate(mp.wb)
Tb.interpolate(mp.Tb)
Sb.interpolate(mp.Sb)
full_pressure.interpolate(mp.P_full)


##########

# Output files for melt functions
Q_ice_file = File(folder+"Q_ice.pvd")
Q_ice_file.write(Q_ice)

Q_mixed_file = File(folder+"Q_mixed.pvd")
Q_mixed_file.write(Q_mixed)

Qs_file = File(folder+"Q_s.pvd")
Qs_file.write(Q_s)

m_file = File(folder+"melt.pvd")
m_file.write(melt)

full_pressure_file = File(folder+"full_pressure.pvd")
full_pressure_file.write(full_pressure)


########
# Plotting top boundary.
shelf_boundary_points = get_top_boundary(L, H2)
top_boundary_mp = pd.DataFrame()


def top_boundary_to_csv(boundary_points, df, t_str):
    df['Qice_t_' + t_str] = Q_ice.at(boundary_points)
    df['Qmixed_t_' + t_str] = Q_mixed.at(boundary_points)
    df['Qlat_t_' + t_str] = Q_latent.at(boundary_points)
    df['Qsalt_t_' + t_str] = Q_s.at(boundary_points)
    df['Melt_t' + t_str] = melt.at(boundary_points)
    df['Tb_t_' + t_str] = Tb.at(boundary_points)
    df['P_t_' + t_str] = full_pressure.at(boundary_points)
    df['Sal_t_' + t_str] = sal.at(boundary_points)
    df['Temp_t_' + t_str] = temp.at(boundary_points)
    df["integrated_melt_t_ " + t_str] = assemble(melt * ds(1))

    if mesh.comm.rank == 0:
        top_boundary_mp.to_csv(folder+"top_boundary_data.csv")


top_boundary_to_csv(shelf_boundary_points, top_boundary_mp, '0.0')

##########

# Boundary conditions
# top boundary: no normal flow, drag flowing over ice
# bottom boundary: no normal flow, drag flowing over bedrock
# grounding line wall (LHS): no normal flow
# open ocean (RHS): pressure to account for density differences

# WEAKLY Enforced BCs
n = FacetNormal(mesh)
Temperature_term = -beta_temp * ((T_bottom - T_surface) * (pow(z, 2) / -water_depth) + (T_surface-T_ref) * z)
Salinity_term = beta_sal * ((S_bottom - S_surface) * (pow(z, 2) / -water_depth) + (S_surface-S_ref) * z)
stress_open_boundary = -n*-g*(Temperature_term + Salinity_term)
no_normal_flow = 0.


up_bcs = {1: {'un': no_normal_flow}, 2: {'stress': stress_open_boundary},
          3: {'un': no_normal_flow, 'drag': 0.0025}, 4: {'un': no_normal_flow, 'drag': 0.0025}}

temp_bcs = {2: {'q': T_restore}, 4: {'flux': -mp.T_flux_bc}}

sal_bcs = {2: {'q': S_restore}, 4: {'flux': -mp.S_flux_bc}}


# STRONGLY Enforced BCs
# open ocean (RHS): no tangential flow because viscosity of outside ocean resists vertical flow.
strong_bcs = [DirichletBC(M.sub(0).sub(1), 0, 2)]

##########

# Solver parameters
mumps_solver_parameters = {
    'snes_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij',
    'snes_max_it': 100,
    'snes_atol': 1e-16,
}

up_solver_parameters = mumps_solver_parameters
temp_solver_parameters = mumps_solver_parameters
sal_solver_parameters = mumps_solver_parameters

##########

# define time steps
T = 3600*24*365
dt = 3660.
output_dt = dt*24.
output_step = output_dt/dt

##########

# Set up time stepping routines
up_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], m, up_fields, up_coupling, dt, up_bcs,
                                                        solver_parameters=up_solver_parameters, strong_bcs=strong_bcs)
temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=temp_solver_parameters)
sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)

##########

# Begin time stepping
t = 0.0
step = 0

while t < T - 0.5*dt:
    up_timestepper.advance(t)
    temp_timestepper.advance(t)
    sal_timestepper.advance(t)

    # Update melt rate functions
    Q_ice.interpolate(mp.Q_ice)
    Q_mixed.interpolate(mp.Q_mixed)
    Q_latent.interpolate(mp.Q_latent)
    Q_s.interpolate(mp.S_flux_bc)
    melt.interpolate(mp.wb)
    Tb.interpolate(mp.Tb)
    Sb.interpolate(mp.Sb)
    full_pressure.interpolate(mp.P_full)

    step += 1
    t += dt

    # Output files
    if step % output_step == 0:
        u_file.write(u_)
        p_file.write(p_)
        t_file.write(temp)
        s_file.write(sal)
        m_file.write(melt)
        Q_mixed_file.write(Q_mixed)
        full_pressure_file.write(full_pressure)
        Qs_file.write(Q_s)
        Q_ice_file.write(Q_ice)

        time_str = str(t/(24.*3600.))
        top_boundary_to_csv(shelf_boundary_points, top_boundary_mp, time_str)

        PETSc.Sys.Print("t=", t)

        PETSc.Sys.Print("integrated melt =", assemble(melt * ds(1)))
