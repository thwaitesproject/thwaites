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
#parser.add_argument("dy", help="horizontal mesh resolution in m",
                  #  type=float)
#parser.add_argument("nz", help="no. of layers in vertical",
#                    type=int)
#parser.add_argument("Kh", help="horizontal eddy viscosity/diffusivity in m^2/s",
#                    type=float)
parser.add_argument("Kv", help="vertical eddy viscosity/diffusivity in m^2/s",
                    type=float)
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
Kv = args.Kv
##########

#  Generate mesh
L = 480E3
Ly = 80E3
shelf_length = 320E3
H1 = 80.
H2 = 600.
H3 = 720.
dy = 4000.0
nx = round(L/dy)
ny = round(Ly/dy)
nz_cavity = 15.
nz_ocean = 18.
dz = Constant(H2/nz_cavity)

mesh2d = RectangleMesh(nx,ny, L, Ly, diagonal='left')
x2d = SpatialCoordinate(mesh2d)
layers = []
cell = 0
yr = 0
min_dz = 0.5*dz # if top cell is thinner than this, merge with cell below
tiny_dz = 0.01*dz # workaround zero measure facet issue (fd issue #1858)

P1_bathy_FS = FunctionSpace(mesh2d, "CG", 1)
P0dg_cells_FS = FunctionSpace(mesh2d, "DG", 0)

#p1_bathy = conditional(x1d[0] < shelf_length, H2, H3)
P1_bathy = Function(P1_bathy_FS)
P1_bathy.interpolate(conditional(x2d[0] + 0.5*dy < shelf_length, H2, H3))

P0dg_cells = Function(P0dg_cells_FS)
tmp = P1_bathy.copy()
P0dg_cells.assign(np.finfo(0.).min)
par_loop("""for (int i=0; i<bathy.dofs; i++) {
        bathy_max[0] = fmax(bathy[i], bathy_max[0]);
        }""",
        dx, {'bathy_max': (P0dg_cells, RW), 'bathy': (tmp, READ)})

P0dg_cells /= dz

P0dg_cells_array = P0dg_cells.dat.data[:]

for i in P0dg_cells_array:
    layers.append([0, i])

mesh = ExtrudedMesh(mesh2d, layers, layer_height=dz)

### Testing mesh
P1_extruded = FunctionSpace(mesh, "CG", 1)
P1_bathy_ext = Function(P1_extruded)

x, y, z = SpatialCoordinate(mesh)
P1_bathy_ext.interpolate(conditional(x + 0.5*dy < shelf_length, H2, H3))
# move top nodes to correct position:
cfs = mesh.coordinates.function_space()
bc = DirichletBC(cfs, as_vector((x, y, P1_bathy_ext)), "top")
bc.apply(mesh.coordinates)

# Scale the mesh to make ice shelf slope
Vc = mesh.coordinates.function_space()
x, y, z = SpatialCoordinate(mesh)
f = Function(Vc).interpolate(as_vector([x, y, conditional(x < shelf_length, ((x/shelf_length)*(H2-H1) + H1)*z/H2, z)]))
mesh.coordinates.assign(f)

ds = CombinedSurfaceMeasure(mesh, 5)

PETSc.Sys.Print("Mesh dimension ", mesh.geometric_dimension())

# Set ocean surface
water_depth = 720.0
mesh.coordinates.dat.data[:, 2] -= water_depth

print("You have Comm WORLD size = ", mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)

x, y, z = SpatialCoordinate(mesh)

PETSc.Sys.Print("Area of South side (Gl wall) should be {:.0f}m^2: ".format(H1*Ly), assemble((Constant(1.0)*ds(1, domain=mesh))))

PETSc.Sys.Print("Area of North side (open ocean) should be {:.0f}m^2: ".format(H3*Ly), assemble(Constant(1.0)*ds(2, domain=mesh)))

PETSc.Sys.Print("Area of bottom: should be {:.0f}m^2: ".format(L*Ly), assemble(Constant(1.0)*ds("bottom", domain=mesh)))

PETSc.Sys.Print("Area of ocean surface should be {:.0f}m^2".format((L-shelf_length)*Ly), assemble(conditional(x > shelf_length, Constant(1.0), 0.0)*ds("top", domain=mesh)))

PETSc.Sys.Print("Area of iceslope: should be {:.0f}m^2: ".format(sqrt(shelf_length**2 + (H2-H1)**2)*Ly), assemble(conditional(x < shelf_length, Constant(1.0), 0.0)*ds("top", domain=mesh)))
n = FacetNormal(mesh)
print("ds_v",assemble(avg(dot(n,n))*dS_v(domain=mesh)))
##########

PETSc.Sys.Print("mesh cell type", mesh.ufl_cell())

# Set up function spaces
# based on (quite old?) firedrake presentation...
# https://fenicsproject.org/pub/presentations/fenics14-paris/presentation_ATTM.pdf

U0 = FiniteElement("CG", triangle, 2)
U1 = FiniteElement("RT", triangle, 2)

V0 = FiniteElement("CG", interval, 2)
V1 = FiniteElement("DG", interval, 1)

W0 = TensorProductElement(U0, V0)  # pressure
W1 = HCurl(TensorProductElement(U1, V0)) + HCurl(TensorProductElement(U0, V1))  # grad W0, i.e grad P so velocity!

V = FunctionSpace(mesh, W1) # Velocity space
W = FunctionSpace(mesh, W0)  # pressure space
M = MixedFunctionSpace([V, W])

# u velocity function space.
scalar_hor_ele = FiniteElement("DG", triangle, 1)
scalar_vert_ele = FiniteElement("DG", interval, 1, variant="equispaced")
scalar_ele = TensorProductElement(scalar_hor_ele, scalar_vert_ele)
#scalar_ele = FiniteElement("DQ", mesh.ufl_cell(), 1, variant="equispaced")
U = FunctionSpace(mesh, scalar_ele)
VDG = VectorFunctionSpace(mesh, "DG", 2) # velocity for output

Q = FunctionSpace(mesh, scalar_ele)
K = FunctionSpace(mesh, scalar_ele)
S = FunctionSpace(mesh, scalar_ele)

##########

# Set up functions
m = Function(M)
v_, p_ = m.split()  # function: velocity, pressure
v, p = split(m)  # expression: velocity, pressure
v_._name = "velocity"
p_._name = "perturbation pressure"
vdg = Function(VDG, name="velocity")

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

##########

# Define a dump file

dump_file = "/data/3d_isomip_plus/first_tests/02.10.20_3d_HJ99_gammafric_dt1800.0_dtOutput86400.0_T8640000.0_ip50.0_constantTres86400.0_KMuh6.0_Muv0.01_Kv0.01_dxy4km_18layers_gl_wall_60m_closed_iterative_initial_solve_coriolis_hypre_pc_press_corr/dump.h5" 

DUMP = False
if DUMP:
    with DumbCheckpoint(dump_file, mode=FILE_UPDATE) as chk:
        # Checkpoint file open for reading and writing
        chk.load(v_, name="velocity")
        chk.load(p_, name="perturbation_pressure")
        chk.load(sal, name="salinity")
        chk.load(temp, name="temperature")

        # from holland et al 2008b. constant T below 200m depth. varying sal.
        T_200m_depth = 1.0

        S_200m_depth = 34.4
        #S_bottom = 34.8
        #salinity_gradient = (S_bottom - S_200m_depth) / -H2
        #S_surface = S_200m_depth - (salinity_gradient * (H2 - water_depth))  # projected linear slope to surface.

        T_restore = Constant(T_200m_depth)
        S_restore = Constant(S_200m_depth) #S_surface + (S_bottom - S_surface) * (z / -water_depth)


else:
    # Assign Initial conditions
    v_init = zero(mesh.geometric_dimension())
    v_.assign(v_init)


    # from holland et al 2008b. constant T below 200m depth. varying sal.
    T_200m_depth = 1.0


    #S_bottom = 34.8
    #salinity_gradient = (S_bottom - S_200m_depth) / -H2
    S_surface = 34.4 #S_200m_depth - (salinity_gradient * (H2 - water_depth))  # projected linear slope to surface.

    T_restore = Constant(T_200m_depth)
    S_restore = Constant(S_surface) #S_surface + (S_bottom - S_surface) * (z / -water_depth)

    temp_init = T_restore
    temp.interpolate(temp_init)

    sal_init = Constant(34.4)
    #sal_init = S_restore
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

# momentum source: the buoyancy term Boussinesq approx. From mitgcm default
T_ref = Constant(0.0)
S_ref = Constant(35)
beta_temp = Constant(2.0E-4)
beta_sal = Constant(7.4E-4)
g = Constant(9.81)
mom_source = as_vector((0., 0.,-g))*(-beta_temp*(temp - T_ref) + beta_sal * (sal - S_ref)) 

rho0 = 1030.
rho.interpolate(rho0*(1.0-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref)))
# coriolis frequency f-plane assumption at 75deg S. f = 2 omega sin (lat) = 2 * 7.2921E-5 * sin (-75 *2pi/360)
f = Constant(-1.409E-4)

# Scalar source/sink terms at open boundary.
absorption_factor = Constant(1.0/restoring_time)
sponge_fraction = 0.02  # fraction of domain where sponge
# Temperature source term
source_temp = conditional(x > (1.0-sponge_fraction) * L,
                           absorption_factor * T_restore,# *((y - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
                          0.0)

# Salinity source term
source_sal = conditional(x > (1.0-sponge_fraction) * L,
                         absorption_factor * S_restore, # *((y - (1.0-sponge_fraction) * L)/(L * sponge_fraction)), 
                         0.0)

# Temperature absorption term
absorp_temp = conditional(x > (1.0-sponge_fraction) * L,
                          absorption_factor, #*((y - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
                          0.0)

# Salinity absorption term
absorp_sal = conditional(x > (1.0-sponge_fraction) * L,
                         absorption_factor, #* ((y - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
                         0.0)


# linearly vary viscosity/diffusivity over domain. reduce vertical/diffusion
kappa_h = Constant(6.0)
open_ocean_kappa_v = Constant(Kv)
grounding_line_kappa_v = Constant(open_ocean_kappa_v * H1/H2)
kappa_v_grad = (open_ocean_kappa_v - grounding_line_kappa_v) / shelf_length
kappa_v = conditional(x < shelf_length, grounding_line_kappa_v + x * kappa_v_grad, open_ocean_kappa_v)
mu_v = kappa_v
#kappa_v = Constant(args.Kh*dz/dy)
#grounding_line_kappa_v = Constant(open_ocean_kappa_v*H1/H2)
#kappa_v_grad = (open_ocean_kappa_v-grounding_line_kappa_v)/L
#kappa_v = grounding_line_kappa_v + y*kappa_v_grad

#sponge_kappa_h = conditional(y > (1.0-sponge_fraction) * L,
#                             1000. * kappa_h * ((y - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
#                             kappa_h)

#sponge_kappa_v = conditional(y > (1.0-sponge_fraction) * L,
#                             1000. * kappa_v * ((y - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
#                             kappa_v)


kappa = as_tensor([[kappa_h, 0, 0], [0, kappa_h, 0], [0, 0, kappa_v]])

kappa_temp = kappa
kappa_sal = kappa
mu = as_tensor([[kappa_h, 0, 0], [0, kappa_h, 0], [0, 0, mu_v]])


# Interior penalty term
# 3*cot(min_angle)*(p+1)*p*nu_max/nu_min

dz_gl = Constant(H1/nz_cavity)
dz_ocean = Constant(water_depth/nz_ocean)
#ip_dz = Function(Q).interpolate(conditional(x < shelf_length, dz_gl + x * (dz_ocean - dz_gl) / shelf_length, dz_ocean))
ip_dz = Constant(20.0)
ip_alpha = 3*dy/ip_dz*2*ip_factor
# Equation fields
vp_coupling = [{'pressure': 1}, {'velocity': 0}]
vp_fields = {'viscosity': mu, 'source': mom_source, 'interior_penalty': ip_alpha}
temp_fields = {'diffusivity': kappa_temp, 'velocity': v, 'interior_penalty': ip_alpha, 'source': source_temp,
               'absorption coefficient': absorp_temp}
sal_fields = {'diffusivity': kappa_sal, 'velocity': v, 'interior_penalty': ip_alpha, 'source': source_sal,
              'absorption coefficient': absorp_sal}

##########

# Get expressions used in melt rate parameterisation
mp = ThreeEqMeltRateParam(sal, temp, p, z, velocity=pow(dot(v_, v_), 0.5), HJ99Gamma=True)

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
ice_drag = 0.0097


# test stress open_boundary
#sop = Function(W)
#sop.interpolate(-g*(Temperature_term + Salinity_term))
#sop_file = File(folder+"boundary_stress.pvd")
#sop_file.write(sop)


vp_bcs = {"top": {'un': no_normal_flow, 'drag': conditional(x < shelf_length, ice_drag, 0.0)}, 
        1: {'un': no_normal_flow}, 2: {'un': no_normal_flow}, 
        "bottom": {'un': no_normal_flow, 'drag': 0.0025},
        3: {'un': no_normal_flow}, 4: {'un': no_normal_flow}} 

temp_bcs = {"top": {'flux': conditional(x < shelf_length, -mp.T_flux_bc, 0.0)}}

sal_bcs = {"top": {'flux':  conditional(x < shelf_length, -mp.S_flux_bc, 0.0)}}


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
            'assembled_ksp_type': 'cg',
            'assembled_ksp_converged_reason': None # just for initial debugging, remove when things are working
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
        vp_timestepper.initialize_pressure(solver_parameters=gmres_solver_parameters)

#u_timestepper = DIRK33(u_eq, u, u_fields, dt, u_bcs, solver_parameters=u_solver_parameters)
temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=temp_solver_parameters)
sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)

##########

# Set up folder
folder = "/data/3d_isomip_plus/extruded_meshes/"+str(args.date)+"_3d_HJ99_gammafric_dt"+str(dt)+\
         "_dtOut"+str(output_dt)+"_T"+str(T)+"_ip"+str(ip_factor.values()[0])+\
         "_Tres"+str(restoring_time)+"_KMuh"+str(kappa_h.values()[0])+"_ooMuv"+str(open_ocean_kappa_v.values()[0])+"_ooKv"+str(open_ocean_kappa_v.values()[0])\
         +"_dx10km_lay5_closed_iter_lump/"
         #+"_extended_domain_with_coriolis_stratified/"  # output folder.
#folder = 'tmp/'


###########

# Output files for velocity, pressure, temperature and salinity
v_file = File(folder+"velocity.pvd")
v_file.write(v_)

# Output files for velocity, pressure, temperature and salinity
vdg_file = File(folder+"dg_velocity.pvd")
vdg_file.write(vdg)

p_file = File(folder+"pressure.pvd")
p_file.write(p_)

t_file = File(folder+"temperature.pvd")
t_file.write(temp)

s_file = File(folder+"salinity.pvd")
s_file.write(sal)

rho_file = File(folder+"density.pvd")
rho_file.write(rho)

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


# Add limiter for DG functions
limiter = VertexBasedLimiter(U)
v_comp = Function(U)
w_comp = Function(U)
########

# Begin time stepping
t = 0.0
step = 0

while t < T - 0.5*dt:
    with timed_stage('velocity-pressure'):
        vp_timestepper.advance(t)
        vdg.project(v_)
    with timed_stage('temperature'):
        temp_timestepper.advance(t)
    with timed_stage('salinity'):
        sal_timestepper.advance(t)

    limiter.apply(sal)
    limiter.apply(temp)
#    v_comp.interpolate(v[0])
#    limiter.apply(v_comp)
#    w_comp.interpolate(v[1])
#    limiter.apply(w_comp)
#    v_.interpolate(as_vector((v_comp, w_comp)))

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
           Q_ice.interpolate(mp.Q_ice)
           Q_mixed.interpolate(mp.Q_mixed)
           Q_latent.interpolate(mp.Q_latent)
           Q_s.interpolate(mp.S_flux_bc)
           melt.interpolate(mp.wb)
           Tb.interpolate(mp.Tb)
           Sb.interpolate(mp.Sb)
           full_pressure.interpolate(mp.P_full)
    
           # Update density for plotting
           rho.interpolate(rho0*(1.0-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref)))

           if MATPLOTLIB_OUT:
               # Write u, v, w, |u| temp, sal, rho to file for plotting later with matplotlib
               matplotlib_out(t)
           
           else:
               # Write out files
               v_file.write(v_)
               vdg_file.write(vdg)
               p_file.write(p_)
               t_file.write(temp)
               s_file.write(sal)
               rho_file.write(rho)
               
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
