# Buoyancy driven overturning circulation
# beneath ice shelf. Wedge geometry. 5km
# Outside temp forcing stratified according to ocean0 isomip.
# with turbulence
# viscosity = temp diffusivity = sal diffusivity: varies linearly over the domain, vertical is 10x weaker.
from thwaites import *
#from thwaites.utility import get_top_boundary
from firedrake.petsc import PETSc
from firedrake import FacetNormal
import pandas as pd
import argparse
import os.path
##########


parser = argparse.ArgumentParser()
parser.add_argument("date", help="date format: dd.mm.yy")
parser.add_argument("dx", help="horizontal mesh resolution in m",
                    type=float)
parser.add_argument("nz", help="no. of layers in vertical",
                    type=int)
parser.add_argument("Kh", help="horizontal eddy viscosity/diffusivity in m^2/s",
                    type=float)
parser.add_argument("Kv", help="vertical eddy viscosity/diffusivity in m^2/s",
                    type=float)
parser.add_argument("restoring_factor", help="restoring time = 1/restoring_factor",
                    type=float)
parser.add_argument("ip_factor", help="dimensionless constant multiplying interior penalty alpha factor",
                    type=float)
parser.add_argument("dt", help="time step in seconds",
                    type=float)
parser.add_argument("dt_output", help="output time step in seconds",
                    type=float)
parser.add_argument("T", help="final simulation time in seconds",
                    type=float)
args = parser.parse_args()


folder = "/data/argparse_tests/"+str(args.date)+".3eq_param.dt"+str(args.dt)+\
         ".dt_output"+str(args.dt_output)+".T"+str(args.T)+".ip"+str(args.ip_factor)+\
         ".alpha"+str(args.restoring_factor)+\
         ".Kh"+str(args.Kh)+".Kv"+str(args.Kv)+\
         ".dx"+str(args.dx)+".nz"+str(args.nz)+"/"  # output folder.

##########


L = 50*1e3
H1 = 100
H2 = 900
dx = args.dx
nx = round(L/dx)
nz = args.nz
dz = H2/nz

mesh = RectangleMesh(nx, nz, L, 1)
x = mesh.coordinates.dat.data[:,0]
y = mesh.coordinates.dat.data[:,1]
mesh.coordinates.dat.data[:,1] = ((x/L)*(H2-H1) + H1)*y


print("You have Comm WORLD size = ", mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)

x, z = SpatialCoordinate(mesh)

# shift z = 0 to surface of ocean. N.b z = 0 is outside domain.


water_depth = 1100.0

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

# from holland et al 2008b. constant T below 200m depth. varying sal.
T_200m_depth = 1.0


S_200m_depth = 34.5
S_bottom = 34.8
salinity_gradient = (S_bottom-S_200m_depth)/-H2
S_surface = S_200m_depth - (salinity_gradient*(H2-water_depth))

T_restore = Constant(T_200m_depth)
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

# linearly vary viscosity/diffusivity over domain. reduce vertical/diffusion
kappa_h = Constant(args.Kh)
kappa_v = Constant(args.Kv)
kappa = as_tensor([[kappa_h, 0], [0, kappa_v]])
kappa_temp = kappa
kappa_sal = kappa
mu = kappa

# momentum source: the buoyancy term Boussinesq approx. From Kimura et al 2013
T_ref = Constant(1.0)#0.0
S_ref = Constant(35.0) #34.8
beta_temp = Constant(1.0E-4) #3.87*10E-5
beta_sal = Constant(7.6E-4) #7.86*10E-4
g = 9.81
rho_prime = -beta_temp * (temp-T_ref) + beta_sal * (sal-S_ref)
mom_source = as_vector((0, -g)) * rho_prime
tke0 = Constant(1e-7)
psi0 = Constant(1.4639e-8)

# Scalar source/sink terms at open boundary.
L = 50000.  # Length of domain
absorption_factor = args.restoring_factor # T = 1 hour
sponge_fraction = 0.2  # fraction of domain where sponge
# Temperature source term
source_temp = conditional(x > (1.0-sponge_fraction) * L,
                          absorption_factor * ((x - (1.0-sponge_fraction) * L) / (L * sponge_fraction)) * T_restore,
                          0.0)

# Salinity source term
source_sal = conditional(x > (1.0-sponge_fraction) * L,
                         absorption_factor * ((x - (1.0-sponge_fraction) * L) / (L * sponge_fraction)) * S_restore,
                         0.0)

# Temperature absorption term
absorp_temp = conditional(x > (1.0-sponge_fraction) * L,
                          absorption_factor * ((x - (1.0-sponge_fraction) * L) / (L * sponge_fraction)),
                          0.0)

# Salinity absorption term
absorp_sal = conditional(x > (1.0-sponge_fraction) * L,
                         absorption_factor * ((x - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
                         0.0)

# Interior penalty term
# 3*cot(min_angle)*(p+1)*p*nu_max/nu_min

ip_alpha = Constant(3*dx/dz*2*args.ip_factor)
# Equation fields
up_coupling = [{'pressure': 1}, {'velocity': 0}]
up_fields = {'viscosity': mu, 'source': mom_source, 'interior_penalty': ip_alpha}
temp_fields = {'diffusivity': kappa_temp, 'velocity': u, 'interior_penalty': ip_alpha, 'source': source_temp,
               'absorption coefficient': absorp_temp}
sal_fields = {'diffusivity': kappa_sal, 'velocity': u, 'interior_penalty': ip_alpha, 'source': source_sal,
              'absorption coefficient': absorp_sal, }
rans_fields = {'velocity': u_, 'viscosity': mu, 'diffusivity': kappa, 'density': rho_prime}

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
mp = ThreeEqMeltRateParam(sal, temp, p, z)

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
# shelf_boundary_points = get_top_boundary()
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


# top_boundary_to_csv(shelf_boundary_points, top_boundary_mp, '0.0')

##########

# Boundary conditions
# top boundary: no normal flow, drag flowing over ice
# bottom boundary: no normal flow, drag flowing over bedrock
# grounding line wall (LHS): no normal flow
# open ocean (RHS): pressure to account for density differences

# WEAKLY Enforced BCs
n = FacetNormal(mesh)
Temperature_term = -beta_temp * ((T_restore-T_ref) * z)
Salinity_term = beta_sal * ((S_bottom - S_surface) * (pow(z, 2) / (-2.0*water_depth)) + (S_surface-S_ref) * z)
stress_open_boundary = -n*-g*(Temperature_term + Salinity_term)
no_normal_flow = 0.
ice_drag = 0.0097


# test stress open_boundary
sop = Function(W)
sop.interpolate(-g*(Temperature_term + Salinity_term))
sop_file = File(folder+"boundary_stress.pvd")
sop_file.write(sop)


up_bcs = {4: {'un': no_normal_flow, 'drag': ice_drag}, 2: {'stress': stress_open_boundary, 'tke': Constant(0.0), 'psi': Constant(0.0)},
          3: {'un': no_normal_flow, 'drag': 0.0025}, 1: {'un': no_normal_flow}}

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
    'snes_rtol': 1e-8,
}

up_solver_parameters = mumps_solver_parameters
temp_solver_parameters = mumps_solver_parameters
sal_solver_parameters = mumps_solver_parameters
rans_solver_parameters = mumps_solver_parameters

##########

# define time steps
dt = args.dt
T = args.T
output_dt = args.dt_output
output_step = output_dt/dt

##########

rans = RANSModel(rans_fields, mesh, bcs=up_bcs, options={'l_max': H2})
rans._create_integrators(BackwardEuler, dt, up_bcs, rans_solver_parameters)
rans.initialize(rans_tke=tke0, rans_psi=psi0)

up_fields['rans_eddy_viscosity'] = rans.eddy_viscosity
sal_fields['rans_eddy_diffusivity'] = rans.eddy_diffusivity
temp_fields['rans_eddy_diffusivity'] = rans.eddy_diffusivity

rans_file = File(os.path.join(folder, "rans.pvd"))
rans_output_fields = (
    rans.tke, rans.psi,
    rans.fields.rans_eddy_viscosity,
    rans.production, rans.rate_of_strain,
    rans.eddy_viscosity,
    rans.fields.rans_mixing_length,
    rans.sqrt_tke, rans.gamma1,
    rans.grad_tke, rans.grad_psi,
    rans.C_mu, rans.C_mu_p,
    rans.fields.N2, rans.fields.N2_neg, rans.fields.N2_pos_over_k, rans.fields.N2_pos,
    rans.fields.M2,
    rans.u_tau, rans.y_plus, rans.u_plus)
rans_file.write(*rans_output_fields)

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
    rans.advance(t)

    step += 1
    t += dt

    # Output files
    if step % output_step == 0:
        # dumb checkpoint for starting from spin up

        with DumbCheckpoint(folder+"dump.h5", mode=FILE_UPDATE) as chk:
            # Checkpoint file open for reading and writing
            chk.store(u_, name="velocity")
            chk.store(p_, name="perturbation_pressure")
            chk.store(temp, name="temperature")
            chk.store(sal, name="salinity")

        # Update melt rate functions
        Q_ice.interpolate(mp.Q_ice)
        Q_mixed.interpolate(mp.Q_mixed)
        Q_latent.interpolate(mp.Q_latent)
        Q_s.interpolate(mp.S_flux_bc)
        melt.interpolate(mp.wb)
        Tb.interpolate(mp.Tb)
        Sb.interpolate(mp.Sb)
        full_pressure.interpolate(mp.P_full)

        # Write melt rate functions
        u_file.write(u_)
        p_file.write(p_)
        t_file.write(temp)
        s_file.write(sal)
        rans_file.write(*rans_output_fields)

        m_file.write(melt)
        Q_mixed_file.write(Q_mixed)
        full_pressure_file.write(full_pressure)
        Qs_file.write(Q_s)
        Q_ice_file.write(Q_ice)

        time_str = str(t/(24.*3600.))
        #top_boundary_to_csv(shelf_boundary_points, top_boundary_mp, time_str)

        PETSc.Sys.Print("t=", t)

        PETSc.Sys.Print("integrated melt =", assemble(melt * ds(4)))
