# Buoyancy driven circulation
# beneath ice shelf with idealised basal crevasse.
# See Jordan et al. 2014 and Ben Yeager thesis (2018)
# for further details of the setup.
from thwaites import *
from thwaites.utility import get_top_boundary, cavity_thickness
from firedrake.petsc import PETSc
from firedrake import FacetNormal
import pandas as pd
import argparse
import numpy as np
from pyop2.profiling import timed_stage
from thwaites.utility import FrazilRisingVelocity
##########


parser = argparse.ArgumentParser()
parser.add_argument("date", help="date format: dd.mm.yy")
#parser.add_argument("dy", help="horizontal mesh resolution in m",
                  #  type=float)
#parser.add_argument("nz", help="no. of layers in vertical",
#                    type=int)
#parser.add_argument("Kh", help="horizontal eddy viscosity/diffusivity in m^2/s",
 #                   type=float)
#parser.add_argument("Kv", help="vertical eddy viscosity/diffusivity in m^2/s",
   #                 type=float)
#parser.add_argument("restoring_time", help="restoring time in s",
                   # type=float)
#parser.add_argument("ip_factor", help="dimensionless constant multiplying interior penalty alpha factor",
                  #  type=float)
parser.add_argument("dt", help="time step in seconds",
                    type=float)
parser.add_argument("output_dt", help="output time step in seconds",
                    type=float)
parser.add_argument("T", help="final simulation time in seconds",
                    type=float)
args = parser.parse_args()

#nz = args.nz #10

ip_factor = Constant(50.)
#dt = 1.0
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
mesh = Mesh("./Crevasse_refined_3d.msh")

PETSc.Sys.Print("Mesh dimension ", mesh.geometric_dimension())

# shift z = 0 to surface of ocean. N.b z = 0 is outside domain.
#PETSc.Sys.Print("Length of lhs", assemble(Constant(1.0)*ds(3, domain=mesh)))

#PETSc.Sys.Print("Length of rhs", assemble(Constant(1.0)*ds(2, domain=mesh)))

PETSc.Sys.Print("Length of bottom", assemble(Constant(1.0)*ds(1, domain=mesh)))

PETSc.Sys.Print("Length of top", assemble(Constant(1.0)*ds(4, domain=mesh)))




print("You have Comm WORLD size = ", mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)

x, y, z = SpatialCoordinate(mesh)
water_depth = 400
##########

# Set up function spaces
V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
W = FunctionSpace(mesh, "CG", 2)  # pressure space
M = MixedFunctionSpace([V, W])

# u velocity function space.
U = FunctionSpace(mesh, "DG", 1)

Q = FunctionSpace(mesh, "DG", 1)  # melt function space
K = FunctionSpace(mesh, "DG", 1)    # temperature space
S = FunctionSpace(mesh, "DG", 1)    # salinity space

##########

# Set up functions
m = Function(M)
v_, p_ = m.subfunctions  # function: y component of velocity, pressure
v, p = split(m)  # expression: y component of velocity, pressure
v_._name = "v_velocity"
p_._name = "perturbation pressure"
#u = Function(U, name="x velocity")  # x component of velocity

rho = Function(K, name="density")
temp = Function(K, name="temperature")
sal = Function(S, name="salinity")
melt = Function(Q, name="melt rate")
Q_mixed = Function(Q, name="ocean heat flux")
Q_ice = Function(Q, name="ice heat flux")
Q_latent = Function(Q, name="latent heat")
Q_s = Function(Q, name="ocean salt flux")
Tb = Function(Q, name="boundary freezing temperature")
Sb = Function(Q, name="boundary salinity")
full_pressure = Function(M.sub(1), name="full pressure")

frazil = Function(Q, name="frazil ice concentration") # should this really be P0dg to prevent negative frazil ice?
frazil_flux = Function(Q, name="frazil ice flux") 
##########

# Define a dump file
dump_file = "/data/2d_crevasse/17.02.22_3_eq_param_ufricHJ99_dt5.0_dtOutput3600.0_T864000.0_isotropicdx5to25m_open_iterative_0.025inflow_qice=0_400mdepth_frazil_sharpmesh_3changedensity_allsource_salsource_limfraz5e-9/dump_step_172800.h5"

DUMP = False
if DUMP:
    with DumbCheckpoint(dump_file, mode=FILE_UPDATE) as chk:
        # Checkpoint file open for reading and writing
        chk.load(v_, name="v_velocity")
        chk.load(p_, name="perturbation_pressure")
        #chk.load(u, name="u_velocity")
        chk.load(sal, name="salinity")
        chk.load(temp, name="temperature")
        chk.load(frazil, name="frazil ice concentration")

        T_200m_depth = -1.965

        S_200m_depth = 34.34
        #S_bottom = 34.8
        #salinity_gradient = (S_bottom - S_200m_depth) / -H2
        #S_surface = S_200m_depth - (salinity_gradient * (H2 - water_depth))  # projected linear slope to surface.

        T_restore = Constant(T_200m_depth)
        S_restore = Constant(S_200m_depth) #S_surface + (S_bottom - S_surface) * (z / -water_depth)


else:
    # Assign Initial conditions
    v_init = zero(mesh.geometric_dimension())
    v_.assign(v_init)

    #u_init = Constant(0.0)
    #u.interpolate(u_init)

    # baseline T3
    T_200m_depth = -1.965


    #S_bottom = 34.8
    #salinity_gradient = (S_bottom - S_200m_depth) / -H2
    S_surface = 34.34 #S_200m_depth - (salinity_gradient * (H2 - water_depth))  # projected linear slope to surface.

    T_restore = Constant(T_200m_depth)
    S_restore = Constant(S_surface) #S_surface + (S_bottom - S_surface) * (z / -water_depth)

    temp_init = T_restore
    temp.interpolate(temp_init)

    sal_init = Constant(34.34)
    #sal_init = S_restore
    sal.interpolate(sal_init)
    
    frazil_init = Constant(5e-9) # initialise with a minimum frazil ice concentration
    frazil.interpolate(frazil_init)


##########

# Set up equations
mom_eq = MomentumEquation(M.sub(0), M.sub(0))
cty_eq = ContinuityEquation(M.sub(1), M.sub(1))
#u_eq = ScalarVelocity2halfDEquation(U, U)
temp_eq = ScalarAdvectionDiffusionEquation(K, K)
sal_eq = ScalarAdvectionDiffusionEquation(S, S)
frazil_eq = FrazilAdvectionDiffusionEquation(Q,Q)
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
mom_source = as_vector((0., 0, -g)) * (rho_perb - frazil * (1 + rho_perb) + frazil * (rho_ice / rho0))
rho.interpolate(rho0*((1-frazil) * (1 + rho_perb)) + frazil * rho_ice)
# coriolis frequency f-plane assumption at 75deg S. f = 2 omega sin (lat) = 2 * 7.2921E-5 * sin (-75 *2pi/360)
#f = Constant(-1.409E-4)

# Scalar source/sink terms at open boundary.
absorption_factor = Constant(1.0/restoring_time)
sponge_fraction = 0.06  # fraction of domain where sponge
# Temperature source term


kappa = as_tensor([[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]])

kappa_temp = kappa
kappa_sal = kappa
kappa_frazil = kappa
mu = kappa

FRV = FrazilRisingVelocity(0.1)  # initial velocity guess needs to be >0
w_i = FRV.frazil_rising_velocity() # Picard iterations converge to value for w_i (which only depends on crystal size, here we assume r =7.5e-4m

frazil_mp = FrazilMeltParam(sal, temp, p, z, frazil)
temp_source = (frazil_mp.Tc - temp - frazil_mp.Lf/frazil_mp.c_p_m) * frazil_mp.wc
temp_absorption = 0 
sal_source = -sal *frazil_mp.wc
sal_absorption = 0 
frazil_source =  -frazil_mp.wc
frazil_absorption = 0

# Interior penalty term
# 3*cot(min_angle)*(p+1)*p*nu_max/nu_min

ip_alpha = Constant(3*dy/dz*2*ip_factor)
# Equation fields
vp_coupling = [{'pressure': 1}, {'velocity': 0}]
vp_fields = {'viscosity': mu, 'source': mom_source} #, 'interior_penalty': ip_alpha}
#u_fields = {'diffusivity': mu, 'velocity': v, 'interior_penalty': ip_alpha, 'coriolis_frequency': f}
temp_fields = {'diffusivity': kappa_temp, 'velocity': v, 'source': temp_source, 'absorption coefficient': temp_absorption}
sal_fields = {'diffusivity': kappa_sal, 'velocity': v, 'source': sal_source, 'absorption coefficient': sal_absorption, }
frazil_fields = {'diffusivity': kappa_frazil, 'velocity': v, 'w_i': Constant(w_i), 'source': frazil_source, 'absorption coefficient': frazil_absorption}

##########

# Get expressions used in melt rate parameterisation
mp = ThreeEqMeltRateParam(sal, temp, p, z, velocity=pow(dot(v, v), 0.5), HJ99Gamma=True, ice_heat_flux=False)


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
frazil_flux.interpolate(w_i*frazil)
##########

# Plotting top boundary.
shelf_boundary_points = get_top_boundary(cavity_length=L, cavity_height=H2, water_depth=water_depth, n=400)
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
    df["integrated_melt_t_ " + t_str] = assemble(melt * ds(4))

    if mesh.comm.rank == 0:
        top_boundary_mp.to_csv(folder+"top_boundary_data.csv")


##########

# Boundary conditions
# top boundary: no normal flow, drag flowing over ice
# bottom boundary: no normal flow, drag flowing over bedrock
# grounding line wall (LHS): no normal flow
# open ocean (RHS): pressure to account for density differences

# WEAKLY Enforced BCs
n = FacetNormal(mesh)
Temperature_term = -beta_temp * ((T_restore-T_ref) * z)
Salinity_term = beta_sal * ((S_restore - S_ref) * z) # ((S_bottom - S_surface) * (pow(z, 2) / (-2.0*water_depth)) + (S_surface-S_ref) * z)
stress_open_boundary = -n*-g*(Temperature_term + Salinity_term)
no_normal_flow = 0.
ice_drag = 0.0097


# test stress open_boundary
#sop = Function(W)
#sop.interpolate(-g*(Temperature_term + Salinity_term))
#sop_file = File(folder+"boundary_stress.pvd")
#sop_file.write(sop)


vp_bcs = {4: {'un': no_normal_flow, 'drag': ice_drag}, 2: {'stress': stress_open_boundary}, 
        3: {'un': -0.025}, 1: {'un': no_normal_flow, 'drag': 2.5e-3},
        5: {'un': no_normal_flow}, 6: {'un': no_normal_flow},}
#u_bcs = {2: {'q': Constant(0.0)}}

temp_bcs = {4: {'flux': -mp.T_flux_bc}, 3:{'q': T_restore}}

sal_bcs = {4: {'flux': -mp.S_flux_bc}, 3:{'q': S_restore}}

frazil_bcs = {}

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

predictor_solver_parameters = {
        'snes_monitor': None,
        'snes_type': 'ksponly',
        'ksp_type': 'gmres',
#        'pc_type': 'gamg',
        'pc_type': 'hypre',
#        'pc_hypre_boomeramg_strong_threshold': 0.6,
        'ksp_converged_reason': None,
#        'ksp_monitor_true_residual': None,
        'ksp_rtol': 1e-5,
        'ksp_max_it': 300,
        }

gmres_solver_parameters = {
        'snes_monitor': None,
        'snes_type': 'ksponly',
        'ksp_type': 'gmres',
        'pc_type': 'sor',
        'ksp_converged_reason': None,
#        'ksp_monitor_true_residual': None,
        'ksp_rtol': 1e-5,
        'ksp_max_it': 300,
        }
vp_solver_parameters = pressure_projection_solver_parameters
u_solver_parameters = mumps_solver_parameters
temp_solver_parameters = gmres_solver_parameters
sal_solver_parameters = gmres_solver_parameters
frazil_solver_parameters = gmres_solver_parameters

##########

# Plotting depth profiles.
z500m = cavity_thickness(5E2, 0., H1, L, H2)
z1km = cavity_thickness(1E3, 0., H1, L, H2)
z2km = cavity_thickness(2E3, 0., H1, L, H2)
z4km = cavity_thickness(4E3, 0., H1, L, H2)
z6km = cavity_thickness(6E3, 0., H1, L, H2)


z_profile500m = np.linspace(z500m-water_depth-1., 1.-water_depth, 50)
z_profile1km = np.linspace(z1km-water_depth-1., 1.-water_depth, 50)
z_profile2km = np.linspace(z2km-water_depth-1., 1.-water_depth, 50)
z_profile4km = np.linspace(z4km-water_depth-1., 1.-water_depth, 50)
z_profile6km = np.linspace(z6km-water_depth-1., 1.-water_depth, 50)


depth_profile500m = []
depth_profile1km = []
depth_profile2km = []
depth_profile4km = []
depth_profile6km = []

for d5e2, d1km, d2km, d4km, d6km in zip(z_profile500m, z_profile1km, z_profile2km, z_profile4km, z_profile6km):
    depth_profile500m.append([5E2, d5e2])
    depth_profile1km.append([1E3, d1km])
    depth_profile2km.append([2E3, d2km])
    depth_profile4km.append([4E3, d4km])
    depth_profile6km.append([6E3, d6km])

velocity_depth_profile500m = pd.DataFrame()
velocity_depth_profile1km = pd.DataFrame()
velocity_depth_profile2km = pd.DataFrame()
velocity_depth_profile4km = pd.DataFrame()
velocity_depth_profile6km = pd.DataFrame()

velocity_depth_profile500m['Z_profile'] = z_profile500m
velocity_depth_profile1km['Z_profile'] = z_profile1km
velocity_depth_profile2km['Z_profile'] = z_profile2km
velocity_depth_profile4km['Z_profile'] = z_profile4km
velocity_depth_profile6km['Z_profile'] = z_profile6km


def depth_profile_to_csv(profile, df, depth, t_str):
    #df['U_t_' + t_str] = u.at(profile)
    vw = np.array(v_.at(profile))
    vv = vw[:, 0]
    ww = vw[:, 1]
    df['V_t_' + t_str] = vv
    df['W_t_' + t_str] = ww
    if mesh.comm.rank == 0:
        df.to_csv(folder+depth+"_profile.csv")




##########

# define time steps
dt = args.dt
T = args.T
output_dt = args.output_dt
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

#u_timestepper = DIRK33(u_eq, u, u_fields, dt, u_bcs, solver_parameters=u_solver_parameters)
temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=temp_solver_parameters)
sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)
frazil_timestepper = DIRK33(frazil_eq, frazil, frazil_fields, dt, frazil_bcs, solver_parameters=frazil_solver_parameters)

##########

# Set up folder
folder = "/data/3d_crevasse/"+str(args.date)+"_3_eq_param_ufricHJ99_dt"+str(dt)+\
         "_dtOutput"+str(output_dt)+"_T"+str(T)+"_3d_isotropicdx10to25m_open_0.025inflow_qice0_400mdepth_frazil_sharpmesh_50m_wide_iterative_nodragside/"
         #+"_extended_domain_with_coriolis_stratified/"  # output folder.


###########

# Output files for velocity, pressure, temperature and salinity
v_file = File(folder+"vw_velocity.pvd")
v_file.write(v_)

p_file = File(folder+"pressure.pvd")
p_file.write(p_)

#u_file = File(folder+"u_velocity.pvd")
#u_file.write(u)

t_file = File(folder+"temperature.pvd")
t_file.write(temp)

s_file = File(folder+"salinity.pvd")
s_file.write(sal)

rho_file = File(folder+"density.pvd")
rho_file.write(rho)

frazil_file = File(folder+"frazil.pvd")
frazil_file.write(frazil)
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

frazil_flux_file = File(folder+"frazil_flux.pvd")
frazil_flux_file.write(frazil_flux)
########

# Extra outputs for plotting
# Melt rate functions along ice-ocean boundary
#top_boundary_to_csv(shelf_boundary_points, top_boundary_mp, '0.0')

# Depth profiles
#depth_profile_to_csv(depth_profile500m, velocity_depth_profile500m, "500m", '0.0')
#depth_profile_to_csv(depth_profile1km, velocity_depth_profile1km, "1km", '0.0')
#depth_profile_to_csv(depth_profile2km, velocity_depth_profile2km, "2km", '0.0')
#depth_profile_to_csv(depth_profile4km, velocity_depth_profile4km, "4km", '0.0')
#depth_profile_to_csv(depth_profile6km, velocity_depth_profile6km, "6km", '0.0')

########

# Extra outputs for matplotlib plotting

def matplotlib_out(t):

    v_array = v_.dat.data[:, 0]
    w_array = v_.dat.data[:, 1]
    temp_array = temp.dat.data
    sal_array = sal.dat.data
    rho_array = rho.dat.data
        
    # Gather all pieces to one array. 
    v_array = mesh.comm.gather(v_array, root=0)
    w_array = mesh.comm.gather(w_array, root=0)
    temp_array = mesh.comm.gather(temp_array, root=0)
    sal_array = mesh.comm.gather(sal_array, root=0)
    rho_array = mesh.comm.gather(rho_array, root=0)

    if mesh.comm.rank == 0:
        # concatenate arrays
        v_array_f = np.concatenate(v_array)
        w_array_f = np.concatenate(w_array)
        vel_mag_array_f = np.sqrt(v_array_f**2 + w_array_f**2)
        temp_array_f = np.concatenate(temp_array)
        sal_array_f = np.concatenate(sal_array)
        rho_array_f = np.concatenate(rho_array)
            
        # Add concatenated arrays to data frame
        matplotlib_df['v_array_{:.0f}hours'.format(t/3600)] = v_array_f
        matplotlib_df['w_array_{:.0f}hours'.format(t/3600)] = w_array_f
        matplotlib_df['vel_mag_array_{:.0f}hours'.format(t/3600)] = vel_mag_array_f
        matplotlib_df['temp_array_{:.0f}hours'.format(t/3600)] = temp_array_f
        matplotlib_df['sal_array_{:.0f}hours'.format(t/3600)] = sal_array_f
        matplotlib_df['rho_array_{:.0f}hours'.format(t/3600)] = rho_array_f
        
        # write dataframe to output file
        matplotlib_df.to_hdf(folder+"matplotlib_arrays.h5", key="0")

MATPLOTLIB_OUT = False

if MATPLOTLIB_OUT:
    
    # Interpolate coordinates to arrays
    y_array, z_array = interpolate(y, Function(U)).dat.data, interpolate(z, Function(U)).dat.data

    # Gather pieces of array to process zero
    y_array = mesh.comm.gather(y_array, root=0)
    z_array = mesh.comm.gather(z_array, root=0) 

    if mesh.comm.rank == 0:
        # Concatanate arrays to have one complete array
        y_array_f = np.concatenate(y_array)
        z_array_f = np.concatenate(z_array)

        # Create a data frame to store arrays for matplotlib plotting later
        matplotlib_df = pd.DataFrame()

        # Add concatenated arrays to data frame
        matplotlib_df['y_array'] = y_array_f
        matplotlib_df['z_array'] = z_array_f

        # Write data frame to file
        matplotlib_df.to_hdf(folder+"matplotlib_arrays.h5", key="0")
    
    # Add initial conditions for v, w, temp, sal, and rho to data frame
    matplotlib_out(0)

########
# Add limiter for DG functions
limiter = VertexBasedP1DGLimiter(S)

# Begin time stepping
t = 0.0
step = 0

while t < T - 0.5*dt:
    with timed_stage('velocity-pressure'):
        vp_timestepper.advance(t)
        #u_timestepper.advance(t)
    with timed_stage('temperature'):
        temp_timestepper.advance(t)
    with timed_stage('salinity'):
        sal_timestepper.advance(t)
    with timed_stage('frazil'):
        frazil_timestepper.advance(t)
    step += 1
    t += dt

    limiter.apply(sal)
    limiter.apply(temp)
    limiter.apply(frazil)
    frazil.interpolate(conditional(frazil < 5e-9, 5e-9, frazil))
    with timed_stage('output'):
       if step % output_step == 0:
           # dumb checkpoint for starting from last timestep reached
           with DumbCheckpoint(folder+"dump.h5", mode=FILE_UPDATE) as chk:
               # Checkpoint file open for reading and writing
               chk.store(v_, name="v_velocity")
               chk.store(p_, name="perturbation_pressure")
               #chk.store(u, name="u_velocity")
               chk.store(temp, name="temperature")
               chk.store(sal, name="salinity")
               chk.store(frazil, name="frazil ice concentration")
    
           # Update melt rate functions
           Q_ice.interpolate(mp.Q_ice)
           Q_mixed.interpolate(mp.Q_mixed)
           Q_latent.interpolate(mp.Q_latent)
           Q_s.interpolate(mp.S_flux_bc)
           melt.interpolate(mp.wb)
           Tb.interpolate(mp.Tb)
           Sb.interpolate(mp.Sb)
           full_pressure.interpolate(mp.P_full)
           frazil_flux.interpolate(w_i*frazil)
    
           # Update density for plotting
           rho.interpolate(rho0*((1-frazil)*(-beta_temp*(temp - T_ref) + beta_sal * (sal - S_ref)) + (rho_ice / rho0) * frazil))

           if MATPLOTLIB_OUT:
               # Write v, w, |u| temp, sal, rho to file for plotting later with matplotlib
               matplotlib_out(t)
           
           else:
               # Write out files
               v_file.write(v_)
               p_file.write(p_)
               #u_file.write(u)
               t_file.write(temp)
               s_file.write(sal)
               rho_file.write(rho)
               frazil_file.write(frazil)   
               # Write melt rate functions
               m_file.write(melt)
               Q_mixed_file.write(Q_mixed)
               full_pressure_file.write(full_pressure)
               Qs_file.write(Q_s)
               Q_ice_file.write(Q_ice)
               frazil_flux_file.write(frazil_flux)
           time_str = str(step)
           #top_boundary_to_csv(shelf_boundary_points, top_boundary_mp, time_str)
    
   #        depth_profile_to_csv(depth_profile500m, velocity_depth_profile500m, "500m", time_str)
    #       depth_profile_to_csv(depth_profile1km, velocity_depth_profile1km, "1km", time_str)
     #      depth_profile_to_csv(depth_profile2km, velocity_depth_profile2km, "2km", time_str)
      #     depth_profile_to_csv(depth_profile4km, velocity_depth_profile4km, "4km", time_str)
       #    depth_profile_to_csv(depth_profile6km, velocity_depth_profile6km, "6km", time_str)
    
           PETSc.Sys.Print("t=", t)
    
           PETSc.Sys.Print("integrated melt =", assemble(melt * ds(4)))

    if step % (output_step * 24) == 0:
        with DumbCheckpoint(folder+"dump_step_{}.h5".format(step), mode=FILE_CREATE) as chk:
            # Checkpoint file open for reading and writing at regular interval
            chk.store(v_, name="v_velocity")
            chk.store(p_, name="perturbation_pressure")
            #chk.store(u, name="u_velocity")
            chk.store(temp, name="temperature")
            chk.store(sal, name="salinity")
            chk.store(frazil, name="frazil ice concentration")
