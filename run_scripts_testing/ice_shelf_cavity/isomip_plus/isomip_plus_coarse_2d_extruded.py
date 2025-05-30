# ISOMIP+ setup 2d slice with extuded mesh
#Buoyancy driven overturning circulation
# beneath ice shelf.
from thwaites import *
from thwaites.utility import get_top_surface, cavity_thickness, CombinedSurfaceMeasure
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
parser.add_argument("dy", help="horizontal mesh resolution in m",
                    type=float)
parser.add_argument("dz", help="vertical mesh resolution in m",
                    type=float)
#parser.add_argument("Kh", help="horizontal eddy viscosity/diffusivity in m^2/s",
#                    type=float)
#parser.add_argument("Kv", help="vertical eddy viscosity/diffusivity in m^2/s",
 #                   type=float)
parser.add_argument("dt", help="time step in seconds",
                    type=float)
parser.add_argument("output_dt", help="output time step in seconds",
                    type=float)
parser.add_argument("T", help="final simulation time in seconds",
                    type=float)
args = parser.parse_args()

restoring_time = Constant(0.1*86400.)
##########

#  Generate mesh
L = 480E3
shelf_length = 320E3
H1 = 80.
H2 = 600.
H3 = 720.
dy = args.dy 
ny = round(L/dy)
nz_cavity = 15
nz_ocean = 18
dz = args.dz 

# create mesh
mesh1d = IntervalMesh(ny, L)
layers = []
cell = 0
yr = 0
min_dz = 0.5*dz # if top cell is thinner than this, merge with cell below
tiny_dz = 0.01*dz # workaround zero measure facet issue (fd issue #1858)

for i in range(ny):
    yr += dy  # y of right-node (assumed to be the higher one)
    if yr <= shelf_length:
        height = H1 + yr/shelf_length * (H2-H1)
    else:
        height = H3
    ncells = ceil((height-min_dz)/dz)
    layers.append([0, ncells])

mesh = ExtrudedMesh(mesh1d, layers, layer_height=dz)

nlayers_column1 = layers[0][1]
# top-left corner node should be at least tiny_dz above height of node below
min_height_corner_node = (nlayers_column1-1)*dz + tiny_dz

# move top nodes to correct position:
cfs = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
bc = DirichletBC(cfs, as_vector((x, Max(conditional(x>shelf_length, H3, H1+x/shelf_length * (H2-H1)), min_height_corner_node))), "top")
bc.apply(mesh.coordinates)

ds = CombinedSurfaceMeasure(mesh, 5)

PETSc.Sys.Print("Mesh dimension ", mesh.geometric_dimension())

# Set ocean surface
water_depth = 720.0
mesh.coordinates.dat.data[:, 1] -= water_depth

print("You have Comm WORLD size = ", mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)

x, z = SpatialCoordinate(mesh)

PETSc.Sys.Print("Length of South side (Gl wall) should be 80.04m: ", assemble((Constant(1.0)*ds(1, domain=mesh))))

PETSc.Sys.Print("Length of North side (open ocean) should be 720m: ", assemble(Constant(1.0)*ds(2, domain=mesh)))

PETSc.Sys.Print("Length of bottom: should be 480e3m: ", assemble(Constant(1.0)*ds("bottom", domain=mesh)))

PETSc.Sys.Print("length of ocean surface should be 160e3m", assemble(conditional(x > shelf_length, Constant(1.0), 0.0)*ds("top", domain=mesh)))

PETSc.Sys.Print("Length of iceslope: should be 320000.64m: ", assemble(conditional(x < shelf_length, Constant(1.0), 0.0)*ds("top", domain=mesh)))
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
vdg_ele = FiniteElement("DQ", mesh.ufl_cell(), 1, variant="equispaced")
VDG1 = VectorFunctionSpace(mesh, vdg_ele) # velocity for output

Q = FunctionSpace(mesh, ele)
K = FunctionSpace(mesh, ele)
S = FunctionSpace(mesh, ele)

P1 = FunctionSpace(mesh, "CG", 1)
##########

# Set up functions
m = Function(M)
v_, p_ = m.subfunctions  # function: velocity, pressure
v, p = split(m)  # expression: velocity, pressure
v_._name = "velocity"
p_._name = "perturbation pressure"
vdg = Function(VDG, name="velocity")
vdg1 = Function(VDG1, name="velocity")

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

rho_anomaly = Function(P1, name="density anomaly")
##########

# Define a dump file
dump_file = "/data/2d_isomip_plus/first_tests/extruded_meshes/10.12.20_2d_HJ99_gammafric_dt1800.0_dtOut3600.0_T8640000.0_ipdef_StratLinTres86400.0_KMuh6.0_MuKvSwitch0.1limProject0.001_dx4kmdz40_closed_iterlump_RTCE2/dump.h5"
DUMP = False
if DUMP:
    with DumbCheckpoint(dump_file, mode=FILE_UPDATE) as chk:
        # Checkpoint file open for reading and writing
        chk.load(v_, name="velocity")
        chk.load(p_, name="perturbation_pressure")
        chk.load(sal, name="salinity")
        chk.load(temp, name="temperature")

     #   vdg1.project(v_) # DQ1 velocity for 
        
        # ISOMIP+ warm conditions .
        T_surface = -1.9
        T_bottom = 1.0

        S_surface = 33.8
        S_bottom = 34.7
        
        T_restore = T_surface + (T_bottom - T_surface) * (z / -water_depth)
        S_restore = S_surface + (S_bottom - S_surface) * (z / -water_depth)



else:
    # Assign Initial conditions
    v_init = zero(mesh.geometric_dimension())
    v_.assign(0.0)

    #vdg1.project(v_) # DQ1 velocity for 
    # ISOMIP+ warm conditions .
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
qdeg = 10

mom_eq = MomentumEquation(M.sub(0), M.sub(0), quad_degree=qdeg)
cty_eq = ContinuityEquation(M.sub(1), M.sub(1), quad_degree=qdeg)
#u_eq = ScalarVelocity2halfDEquation(U, U)
temp_eq = ScalarAdvectionDiffusionEquation(K, K, quad_degree=qdeg)
sal_eq = ScalarAdvectionDiffusionEquation(S, S, quad_degree=qdeg)

##########

# Terms for equation fields

# momentum source: the buoyancy term Boussinesq approx. 
T_ref = Constant(-1.0)
S_ref = Constant(34.2)
beta_temp = Constant(3.733E-5)
beta_sal = Constant(7.843E-4)
g = Constant(9.81)
mom_source = as_vector((0.,-g))*(-beta_temp*(temp - T_ref) + beta_sal * (sal - S_ref)) 

rho0 = 1027.51
rho.interpolate(rho0*(1.0-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref)))
rho_anomaly.project(-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref))

gradrho = Function(P1)  # vertical component of gradient of density anomaly units m^-1
gradrho.project(Dx(rho_anomaly, mesh.geometric_dimension() - 1))





# coriolis frequency f-plane assumption at 75deg S. f = 2 omega sin (lat) = 2 * 7.2921E-5 * sin (-75 *2pi/360)
f = Constant(-1.409E-4)

# Scalar source/sink terms at open boundary.
absorption_factor = Constant(1.0/restoring_time)
sponge_fraction = 0.02  # fraction of domain where sponge
# Temperature source term
source_temp = conditional(x > (1.0-sponge_fraction) * L,
                           absorption_factor * T_restore *((x - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
                          0.0)

# Salinity source term
source_sal = conditional(x > (1.0-sponge_fraction) * L,
                         absorption_factor * S_restore *((x - (1.0-sponge_fraction) * L)/(L * sponge_fraction)), 
                         0.0)

# Temperature absorption term
absorp_temp = conditional(x > (1.0-sponge_fraction) * L,
                          absorption_factor*((x - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
                          0.0)

# Salinity absorption term
absorp_sal = conditional(x > (1.0-sponge_fraction) * L,
                         absorption_factor * ((x - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
                         0.0)

# set Viscosity/diffusivity (m^2/s)
mu_h = Constant(6.0)
mu_v = Constant(1e-3)
kappa_h = Constant(1.0)
kappa_v = Constant(5e-5)

# Code below for Kv switch
#kappa_v = Function(P1) 
DeltaS = Constant(1.0)  # rough order of magnitude estimate of change in salinity over restoring region
gradrho_scale = DeltaS * beta_sal / water_depth  # rough oredr of magnitude estimate for vertical gradient of density anomaly. units m^-1
#kappa_v.assign(conditional((gradrho / gradrho_scale) < 1e-3, 1e-3, 1e-1))

mu = as_tensor([[mu_h, 0], [0, mu_v]])
kappa = as_tensor([[kappa_h, 0], [0, kappa_v]])

kappa_temp = kappa
kappa_sal = kappa


# Interior penalty term
# 3*cot(min_angle)*(p+1)*p*nu_max/nu_min

#dz_gl = Constant(H1/nz_cavity)
#dz_ocean = Constant(water_depth/nz_ocean)
#dz = Function(Q).interpolate(conditional(x < shelf_length, dz_gl + x * (dz_ocean - dz_gl) / shelf_length, dz_ocean))
# Equation fields
vp_coupling = [{'pressure': 1}, {'velocity': 0}]
vp_fields = {'viscosity': mu, 'source': mom_source}
temp_fields = {'diffusivity': kappa_temp, 'velocity': v, 'source': source_temp,
               'absorption coefficient': absorp_temp}
sal_fields = {'diffusivity': kappa_sal, 'velocity': v, 'source': source_sal,
              'absorption coefficient': absorp_sal}

##########

# Get expressions used in melt rate parameterisation
mp = ThreeEqMeltRateParam(sal, temp, p, z, velocity=pow(dot(vdg, vdg), 0.5), ice_heat_flux=False)

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

# Plotting top boundary.
shelf_boundary_points =  get_top_surface(cavity_xlength=5000.,cavity_ylength=L, cavity_height=H2, water_depth=water_depth, dx=500.0,dy=500.) 
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
    df["integrated_melt_t_ " + t_str] = assemble(melt * ds(106))

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


# test stress open_boundary
#sop = Function(W)
#sop.interpolate(-g*(Temperature_term + Salinity_term))
#sop_file = File(folder+"boundary_stress.pvd")
#sop_file.write(sop)


vp_bcs = {"top": {'un': no_normal_flow, 'drag': conditional(x < shelf_length, 2.5E-3, 0.0)}, 
        1: {'un': no_normal_flow}, 2: {'un': no_normal_flow}, 
        "bottom": {'un': no_normal_flow, 'drag': 2.5E-3}} 

temp_bcs = {"top": {'flux': conditional(x < shelf_length, -mp.T_flux_bc, 0.0)}}

sal_bcs = {"top": {'flux':  conditional(x < shelf_length, -mp.S_flux_bc, 0.0)}}


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
        'ksp_type': 'preonly',  # we solve the full schur complement exactly, so no need for outer krylov
        'mat_type': 'matfree',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_fact_type': 'full',
        # velocity mass block:
        'fieldsplit_0': {
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_ksp_type': 'preonly',
            'assembled_pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps'
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
            'laplace_ksp_ksp_monitor_true_residual': None,
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
temp_solver_parameters = gmres_solver_parameters
sal_solver_parameters = gmres_solver_parameters

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
    depth_profile500m.append([2500, 5E2, d5e2])
    depth_profile1km.append([2500, 1E3, d1km])
    depth_profile2km.append([2500, 2E3, d2km])
    depth_profile4km.append([2500, 4E3, d4km])
    depth_profile6km.append([2500, 6E3, d6km])

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
    uvw = np.array(v_.at(profile))
    uu = uvw[:, 0]
    vv = uvw[:, 1]
    ww = uvw[:, 2]
    df['U_t_' + t_str] = uu
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
                                                          predictor_solver_parameters=u_solver_parameters,
                                                          picard_iterations=1,
                                                          pressure_nullspace=VectorSpaceBasis(constant=True))

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

##########

# Set up folder
folder = "/data/2d_isomip_plus/first_tests/extruded_meshes/"+str(args.date)+"_2d_isomip+_dt"+str(dt)+\
         "_dtOut"+str(output_dt)+"_T"+str(T)+"_ipdef_StratLinTres"+str(restoring_time.values()[0])\
         +"_Muh"+str(mu_h.values()[0])+"_Muv"+str(mu_v.values()[0])+"_Kh"+str(kappa_h.values()[0])+"_Kv"+str(kappa_v.values()[0])\
         +"_dx"+str(round(1e-3*dy))+"kmdz"+str(round(dz))+"m_closed_iterlump_P1dglimSquTritracers_eos/"
#folder = 'tmp/'


###########

# Output files for velocity, pressure, temperature and salinity
v_file = File(folder+"velocity.pvd")
v_file.write(v_)

# Output files for velocity, pressure, temperature and salinity


vdg.project(v_)  # DQ2 velocity for plotting
vdg_file = File(folder+"dg_velocity.pvd")
vdg_file.write(vdg)

vdg1.project(v_) # DQ1 velocity for 
vdg1_file = File(folder+"P1dg_velocity_lim.pvd")
vdg1_file.write(vdg1)

p_file = File(folder+"pressure.pvd")
p_file.write(p_)

t_file = File(folder+"temperature.pvd")
t_file.write(temp)

s_file = File(folder+"salinity.pvd")
s_file.write(sal)

rho_file = File(folder+"density.pvd")
rho_file.write(rho)

rhograd_file = File(folder+"density_anomaly_grad.pvd")
rhograd_file.write(gradrho)

#kappav_file = File(folder+"kappav.pvd")
#kappav_file.write(kappa_v)
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

##########

with DumbCheckpoint(folder+"initial_pressure_dump", mode=FILE_UPDATE) as chk:
    # Checkpoint file open for reading and writing
    chk.store(v_, name="velocity")
    chk.store(p_, name="perturbation_pressure")
    chk.store(temp, name="temperature")
    chk.store(sal, name="salinity")



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

    u_array = v_.dat.data[:, 0]
    v_array = v_.dat.data[:, 1]
    w_array = v_.dat.data[:, 2]
    temp_array = temp.dat.data
    sal_array = sal.dat.data
    rho_array = rho.dat.data
        
    # Gather all pieces to one array. 
    u_array = mesh.comm.gather(u_array, root=0)
    v_array = mesh.comm.gather(v_array, root=0)
    w_array = mesh.comm.gather(w_array, root=0)
    temp_array = mesh.comm.gather(temp_array, root=0)
    sal_array = mesh.comm.gather(sal_array, root=0)
    rho_array = mesh.comm.gather(rho_array, root=0)

    if mesh.comm.rank == 0:
        # concatenate arrays
        u_array_f = np.concatenate(u_array)
        v_array_f = np.concatenate(v_array)
        w_array_f = np.concatenate(w_array)
        vel_mag_array_f = np.sqrt(u_array_f**2+v_array_f**2 + w_array_f**2)
        temp_array_f = np.concatenate(temp_array)
        sal_array_f = np.concatenate(sal_array)
        rho_array_f = np.concatenate(rho_array)
            
        # Add concatenated arrays to data frame
        matplotlib_df['u_array_{:.0f}hours'.format(t/3600)] = u_array_f
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
    x_array = interpolate(x, Function(U)).dat.data
    y_array = interpolate(y, Function(U)).dat.data
    z_array = interpolate(z, Function(U)).dat.data

    # Gather pieces of array to process zero
    x_array = mesh.comm.gather(x_array, root=0)
    y_array = mesh.comm.gather(y_array, root=0)
    z_array = mesh.comm.gather(z_array, root=0) 

    if mesh.comm.rank == 0:
        # Concatanate arrays to have one complete array
        x_array_f = np.concatenate(x_array)
        y_array_f = np.concatenate(y_array)
        z_array_f = np.concatenate(z_array)

        # Create a data frame to store arrays for matplotlib plotting later
        matplotlib_df = pd.DataFrame()

        # Add concatenated arrays to data frame
        matplotlib_df['x_array'] = x_array_f
        matplotlib_df['y_array'] = y_array_f
        matplotlib_df['z_array'] = z_array_f

        # Write data frame to file
        matplotlib_df.to_hdf(folder+"matplotlib_arrays.h5", key="0")
    
    # Add initial conditions for v, w, temp, sal, and rho to data frame
    matplotlib_out(0)

####################

# Add limiter for DG functions
limiter = VertexBasedP1DGLimiter(U, squeezed_triangles=True)
v_comp = Function(U)
w_comp = Function(U)
########

# Begin time stepping
t = 0.0
step = 0

while t < T - 0.5*dt:
    with timed_stage('velocity-pressure'):
        vp_timestepper.advance(t)
        vdg.project(v_)  # DQ2 velocity for plotting
#        vdg1.project(v_) # DQ1 velocity for 
#        v_comp.interpolate(vdg1[0])
#        limiter.apply(v_comp)
#        w_comp.interpolate(vdg1[1])
#        limiter.apply(w_comp)
#        vdg1.interpolate(as_vector((v_comp, w_comp)))
    with timed_stage('temperature'):
        temp_timestepper.advance(t)
    with timed_stage('salinity'):
        sal_timestepper.advance(t)

    limiter.apply(sal)
    limiter.apply(temp)

    rho_anomaly.project(-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref))
    gradrho.project(Dx(rho_anomaly, mesh.geometric_dimension() - 1))
    #kappa_v.assign(conditional((gradrho / gradrho_scale) < 1e-1, 1e-3, 1e-1))

    step += 1
    t += dt

    with timed_stage('output'):
       if step % output_step == 0:
           # dumb checkpoint for starting from last timestep reached
           with DumbCheckpoint(folder+"dump.h5", mode=FILE_UPDATE) as chk:
               # Checkpoint file open for reading and writing
               chk.store(v_, name="velocity")
               chk.store(p_, name="perturbation_pressure")
               chk.store(temp, name="temperature")
               chk.store(sal, name="salinity")

           # Update melt rate functions
           rho.interpolate(rho0*(1.0-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref)))
           Q_ice.interpolate(mp.Q_ice)
           Q_mixed.interpolate(mp.Q_mixed)
           Q_latent.interpolate(mp.Q_latent)
           Q_s.interpolate(mp.S_flux_bc)
           melt.interpolate(mp.wb)
           Tb.interpolate(mp.Tb)
           Sb.interpolate(mp.Sb)
           full_pressure.interpolate(mp.P_full)
    
           # Update density for plotting

           if MATPLOTLIB_OUT:
               # Write u, v, w, |u| temp, sal, rho to file for plotting later with matplotlib
               matplotlib_out(t)
           
           else:
               # Write out files
               v_file.write(v_)
               vdg_file.write(vdg)
#               vdg1_file.write(vdg1)
               p_file.write(p_)
               t_file.write(temp)
               s_file.write(sal)
               rho_file.write(rho)
               
               rhograd_file.write(gradrho)
#               kappav_file.write(kappa_v)
               # Write melt rate functions
               m_file.write(melt)
               Q_mixed_file.write(Q_mixed)
               full_pressure_file.write(full_pressure)
               Qs_file.write(Q_s)
               Q_ice_file.write(Q_ice)
    
           time_str = str(step)
           #top_boundary_to_csv(shelf_boundary_points, top_boundary_mp, time_str)
    
           #depth_profile_to_csv(depth_profile500m, velocity_depth_profile500m, "500m", time_str)
           #depth_profile_to_csv(depth_profile1km, velocity_depth_profile1km, "1km", time_str)
           #depth_profile_to_csv(depth_profile2km, velocity_depth_profile2km, "2km", time_str)
           #depth_profile_to_csv(depth_profile4km, velocity_depth_profile4km, "4km", time_str)
           #depth_profile_to_csv(depth_profile6km, velocity_depth_profile6km, "6km", time_str)
    
           PETSc.Sys.Print("t=", t)
    
           PETSc.Sys.Print("integrated melt =", assemble(conditional(x < shelf_length, melt, 0.0) * ds("top")))

    if step % (output_step * 24) == 0:
        with DumbCheckpoint(folder+"dump_step_{}.h5".format(step), mode=FILE_CREATE) as chk:
            # Checkpoint file open for reading and writing at regular interval
            chk.store(v_, name="velocity")
            chk.store(p_, name="perturbation_pressure")
            chk.store(temp, name="temperature")
            chk.store(sal, name="salinity")
