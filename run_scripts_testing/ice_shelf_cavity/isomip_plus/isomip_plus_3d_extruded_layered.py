# ISOMIP+ setup 2d slice with extuded mesh
#Buoyancy driven overturning circulation
# beneath ice shelf.
from thwaites import *
from thwaites.utility import get_top_surface, cavity_thickness, CombinedSurfaceMeasure, ExtrudedFunction
from firedrake.petsc import PETSc
from firedrake import FacetNormal
import pandas as pd
import argparse
import numpy as np
from math import ceil
from pyop2.profiling import timed_stage
import rasterio
from thwaites.interpolate import interpolate as interpolate_data
##########
PETSc.Sys.popErrorHandler()

parser = argparse.ArgumentParser()
parser.add_argument("date", help="date format: dd.mm.yy")
parser.add_argument("dy", help="horizontal mesh resolution in m",
                    type=float)
parser.add_argument("nz", help="no. of layers in vertical",
                    type=int)
#parser.add_argument("Kh", help="horizontal eddy viscosity/diffusivity in m^2/s",
#                    type=float)
#parser.add_argument("Kv", help="vertical eddy viscosity/diffusivity in m^2/s",
#                    type=float)
#parser.add_argument("restoring_time", help="restoring time in s",
                   # type=float)
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
L = 800E3
Ly = 80E3
shelf_length = 640E3
H1 = 130.
H2 = 600.
H3 = 720.
water_depth = H3
dy = args.dy 
nx = round(L/dy)
ny = round(Ly/dy)
dz = H2/args.nz #40.0

# create mesh
base_mesh = Mesh("./isomip_outline_mesh_res_alignedinterioricefront.msh") 
layers = []
cell = 0
yr = 0
min_dz = 0.5*dz # if top cell is thinner than this, merge with cell below
tiny_dz = 0.01*dz # workaround zero measure facet issue (fd issue #1858)

x_base = SpatialCoordinate(base_mesh)

P1 = FunctionSpace(base_mesh, "CG", 1)
ocean_thickness = Function(P1)
ocean_thickness.interpolate(conditional(x_base[0] + 0.5*dy < shelf_length, H2, H3))

def extruded_cavity_mesh(base_mesh, ocean_thickness):
    P0dg = FunctionSpace(base_mesh, "DG", 0)
    P0dg_cells = Function(P0dg)
    tmp = ocean_thickness.copy()
    P0dg_cells.assign(-1.0)#np.finfo(0.).min)
    par_loop("""for (int i=0; i<bathy.dofs; i++) {
            bathy_max[0] = fmax(bathy[i], bathy_max[0]);
            }""",
            dx, {'bathy_max': (P0dg_cells, RW), 'bathy': (tmp, READ)})

    P0dg_cells /= dz

    P0dg_cells_array = P0dg_cells.dat.data_ro_with_halos[:]



    for i in P0dg_cells_array:
        layers.append([0, i])

    mesh = ExtrudedMesh(base_mesh, layers, layer_height=dz)
    return mesh 

mesh = extruded_cavity_mesh(base_mesh, ocean_thickness)
x, y, z = SpatialCoordinate(mesh)


P0_extruded = FunctionSpace(mesh, 'DG', 0)
p0mesh_cells = Function(P0_extruded)
PETSc.Sys.Print("number of cells:", len(p0mesh_cells.dat.data[:]))


P1_extruded = FunctionSpace(mesh, 'CG', 1)

# Redefine thickness (without ice slope!) on extruded mesh
ocean_thickness_extruded = ExtrudedFunction(ocean_thickness, mesh_3d=mesh)

# move top nodes to correct position:
cfs = mesh.coordinates.function_space()
bc = DirichletBC(cfs, as_vector((x, y, ocean_thickness_extruded.view_3d)), "top")
bc.apply(mesh.coordinates)

# Bathymetry 
x_bar = Constant(300E3) # Characteristic along flow length scale of the bedrock
x_tilda = x / x_bar  # isomip+ x coordinate used for defining along flow bathymetry/bedrock topography 
B0 = Constant(-150.0) # Bedrock topography at x = 0 (in the ice domain!)
B2 = Constant(-728.8) # Second bedrock topography coefficient 
B4 = Constant(343.91) # Third bedrock topography coefficient
B6 = Constant(-50.57) # Forth bedrock topography coefficient

bathy_x = B0 + B2 * pow(x_tilda, 2) + B4 * pow(x_tilda, 4) + B6 * pow(x_tilda, 6)

d_c =  Constant(500.0) # Depth of the trough compared with the side walls
w_c = Constant(24E3) # Half width of the trough 
f_c = Constant(4E3) # Characteristic width of the side of the channel

bathy_y = d_c / (1 + exp(-2 * (y - 0.5 * Ly - w_c) / f_c))  + d_c / (1 + exp(2 * (y - 0.5 * Ly + w_c) / f_c))

bathymetry = Function(P1_extruded)
bathymetry.interpolate(conditional(bathy_x + bathy_y < -water_depth,
                        -water_depth,
                        bathy_x + bathy_y))

print("max bathy : ", bathymetry.dat.data[:].max())

ice_draft_file = rasterio.open(f'netcdf:Ocean1_input_geom_v1.01.nc:lowerSurface', 'r')
#ice_draft.interpolate(conditional(x - 0.5*dy < shelf_length, (x/shelf_length)*(H2-H1) + H1,H3) - water_depth) 


#P1 = FunctionSpace(base_mesh, "CG", 1)
ice_draft_base = interpolate_data(ice_draft_file, P1)
print("max icedraft : ",ice_draft_base.dat.data[:].max())
print("min icedraft : ",ice_draft_base.dat.data[:].min())

ocean_thickness = Function(P1_extruded)

ice_draft = ExtrudedFunction(ice_draft_base, mesh_3d=mesh)
print("max icedraft extruded : ",ice_draft.view_3d.dat.data[:].max())
print("min icedraft extruded : ",ice_draft.view_3d.dat.data[:].min())
ocean_thickness.interpolate(ice_draft.view_3d - bathymetry)

print("max thickness : ", ocean_thickness.dat.data[:].max())
print("min thickness : ", ocean_thickness.dat.data[:].min())
ocean_thickness.interpolate(conditional(ice_draft.view_3d - bathymetry < Constant(10),
                                        Constant(10),
                                        ice_draft.view_3d - bathymetry)) 
print("max thickness : ", ocean_thickness.dat.data[:].max())
print("min thickness : ", ocean_thickness.dat.data[:].min())

# Scale the mesh to make ice shelf slope
Vc = mesh.coordinates.function_space()
x, y, z = SpatialCoordinate(mesh)
f = Function(Vc).interpolate(as_vector([x, y, conditional(x + 0.5*dy < shelf_length, ocean_thickness*z/H2, ocean_thickness*z/H3) - -bathymetry]))
#f = Function(Vc).interpolate(as_vector([x, y, ocean_thickness*z/H2 - -bathymetry]))
mesh.coordinates.assign(f)

ds = CombinedSurfaceMeasure(mesh, 5)

PETSc.Sys.Print("Mesh dimension ", mesh.geometric_dimension())


print("You have Comm WORLD size = ", mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)

x, y, z = SpatialCoordinate(mesh)

PETSc.Sys.Print("Area of South side (Gl wall) should be {:.0f}m^2: ".format(H1*Ly), assemble((Constant(1.0)*ds(1, domain=mesh))))

PETSc.Sys.Print("Area of North side (open ocean) should be {:.0f}m^2: ".format(H3*Ly), assemble(Constant(1.0)*ds(2, domain=mesh)))

PETSc.Sys.Print("Area of bottom: should be {:.0f}m^2: ".format(L*Ly), assemble(Constant(1.0)*ds("bottom", domain=mesh)))

PETSc.Sys.Print("Area of ocean surface should be {:.0f}m^2".format((L-shelf_length)*Ly), assemble(conditional(x > shelf_length, Constant(1.0), 0.0)*ds("top", domain=mesh)))

PETSc.Sys.Print("Area of iceslope: should be {:.0f}m^2: ".format(sqrt(shelf_length**2 + (H2-H1)**2)*Ly), assemble(conditional(x < shelf_length, Constant(1.0), 0.0)*ds("top", domain=mesh)))

n = FacetNormal(mesh)
print(assemble(avg(dot(n,n))*dS_v(domain=mesh)))
print(assemble(avg(dot(n,n))*dS_h(domain=mesh)))
mesh_file = File("ocean_thickness_icedraftfile.pvd")
mesh_file.write(ocean_thickness)
mesh_file = File("bathy_icedraftfile.pvd")
mesh_file.write(bathymetry)
mesh_file = File("icedraft_icedraftfile.pvd")
mesh_file.write(ice_draft)

p0mesh_cells.interpolate(CellVolume(mesh))
PETSc.Sys.Print("max cell volume:", p0mesh_cells.dat.data[:].max())
PETSc.Sys.Print("min cell volume:", p0mesh_cells.dat.data[:].min())
mesh_cellvol_file = File("mesh_cell_vol.pvd")
mesh_cellvol_file.write(p0mesh_cells)
##########
PETSc.Sys.Print("mesh cell type", mesh.ufl_cell())
# Set up function spaces
# Nedelec wedge element. Discontinuous in normal component
# continuous in tangential component.
N2_1 = FiniteElement("N2curl", triangle, 1, variant="integral")
CG_2 = FiniteElement("CG", interval, 2)
N2CG = TensorProductElement(N2_1, CG_2)
Ned_horiz = HCurlElement(N2CG)
P2tr = FiniteElement("CG", triangle, 2)
P1dg = FiniteElement("DG", interval, 1)
P2P1 = TensorProductElement(P2tr, P1dg)
Ned_vert = HCurlElement(P2P1)
Ned_wedge = Ned_horiz + Ned_vert
V = FunctionSpace(mesh, Ned_wedge)

W0 = TensorProductElement(P2tr, CG_2) 

W = FunctionSpace(mesh, W0)  # pressure space
M = MixedFunctionSpace([V, W])

# u velocity function space.
scalar_hor_ele = FiniteElement("DG", triangle, 1, variant="equispaced")
scalar_vert_ele = FiniteElement("DG", interval, 1, variant="equispaced")
scalar_ele = TensorProductElement(scalar_hor_ele, scalar_vert_ele)
#scalar_ele = FiniteElement("DQ", mesh.ufl_cell(), 1, variant="equispaced")
S = FunctionSpace(mesh, scalar_ele)
VDG = VectorFunctionSpace(mesh, "DQ", 2) # velocity for output
#VDG1 = VectorFunctionSpace(mesh, "DQ", 1) # velocity for output

PETSc.Sys.Print("vel dofs:", V.dim())
PETSc.Sys.Print("pressure dofs:", W.dim())
PETSc.Sys.Print("combined dofs:", M.dim())
PETSc.Sys.Print("scalar dofs:", S.dim())
PETSc.Sys.Print("P1 dofs (no of nodes):", P1_extruded.dim())
##########
# Set up functions
m = Function(M)
v_, p_ = m.split()  # function: velocity, pressure
v, p = split(m)  # expression: velocity, pressure
v_._name = "velocity"
p_._name = "perturbation pressure"
vdg = Function(VDG, name="velocity")
#vdg1 = Function(VDG1, name="velocity")
#u_pred_dg = Function(VDG, name="pred_velocity")

rho = Function(S, name="density")
temp = Function(S, name="temperature")
sal = Function(S, name="salinity")
melt = Function(S, name="melt rate")
#Q_mixed = Function(Q, name="ocean heat flux")
#Q_ice = Function(Q, name="ice heat flux")
#Q_latent = Function(Q, name="latent heat")
#Q_s = Function(Q, name="ocean salt flux")
#Tb = Function(Q, name="boundary freezing temperature")
#Sb = Function(Q, name="boundary salinity")
#full_pressure = Function(M.sub(1), name="full pressure")

rho_anomaly = Function(P1_extruded, name="density anomaly")
##########

# Define a dump file

dump_file = "/data/3d_isomip_plus/extruded_meshes/21.04.21_3d_isomip+_dt900.0_dtOut43200.0_T1728000.0_ipdef_StratLinTres8640.0_Muh6.0_fixMuv0.001_Kh1.0_fixKv5e-05_dx2km_lay30_glwall80mLy4km_closed_iterlump_deg2ip3_predrtol1e-5_offsetmelt/dump_step_1440.h5" 

DUMP = False
if DUMP:
    with DumbCheckpoint(dump_file, mode=FILE_UPDATE) as chk:
        # Checkpoint file open for reading and writing
        chk.load(v_, name="velocity")
        chk.load(p_, name="perturbation_pressure")
        chk.load(sal, name="salinity")
        chk.load(temp, name="temperature")

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
    v_.assign(v_init)


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
degree = V.ufl_element().degree()
PETSc.Sys.Print("velocity: degree", degree )
PETSc.Sys.Print("velocity: hor/vert degree", max(degree[0], degree[1]))
# Set up equations
qdeg = 10

mom_eq = MomentumEquation(M.sub(0), M.sub(0), quad_degree=qdeg)
cty_eq = ContinuityEquation(M.sub(1), M.sub(1), quad_degree=qdeg)
temp_eq = ScalarAdvectionDiffusionEquation(S, S, quad_degree=qdeg)
sal_eq = ScalarAdvectionDiffusionEquation(S, S, quad_degree=qdeg)

##########

# Terms for equation fields

# momentum source: the buoyancy term Boussinesq approx. 
T_ref = Constant(-1.0)
S_ref = Constant(34.2)
beta_temp = Constant(3.733E-5)
beta_sal = Constant(7.843E-4)
g = Constant(9.81)
mom_source = as_vector((0.,0.,-g))*(-beta_temp*(temp - T_ref) + beta_sal * (sal - S_ref)) 

rho0 = 1027.51
rho.interpolate(rho0*(1.0-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref)))
rho_anomaly_projector = Projector(-beta_temp * (temp - T_ref) + beta_sal * (sal - S_ref), rho_anomaly)

gradrho = Function(P1_extruded)  # vertical component of gradient of density anomaly units m^-1
gradrho_projector = Projector(Dx(rho_anomaly, mesh.geometric_dimension() - 1), gradrho)


# coriolis frequency f-plane assumption at 75deg S. f = 2 omega sin (lat) = 2 * 7.2921E-5 * sin (-75 *2pi/360)
f = Constant(-1.409E-4)

class VerticalDensityGradientSolver:
    """Computes vertical density gradient.
                                                                                                                                                                                                                       """
    def __init__(self, rho, solution):
        self.rho = rho
        self.solution = solution
        
        self.fs = self.solution.function_space()
        self.mesh = self.fs.mesh()
        self.n = FacetNormal(self.mesh)
        
        test = TestFunction(self.fs)
        tri = TrialFunction(self.fs)
        vert_dim = self.mesh.geometric_dimension()-1
        
        a = test*tri*dx
        L = -Dx(test, vert_dim)*self.rho*dx + test*self.n[vert_dim]*self.rho*ds_tb #+ avg(rho) * jump(gradrho_test, n[dim]) * dS_h (this is zero because jump(phi,n) = 0 for continuous P1 test function!)
       
        prob = LinearVariationalProblem(a, L, self.solution, constant_jacobian=True)
        self.weak_grad_solver = LinearVariationalSolver(prob) # #, solver_parameters=solver_parameters)
       
    def solve(self):
        self.weak_grad_solver.solve()

#gradrho = Function(P1_extruded)
#grad_rho_solver = VerticalDensityGradientSolver(rho, gradrho)        

#grad_rho_solver.solve()

# Scalar source/sink terms at open boundary.
absorption_factor = Constant(1.0/restoring_time)
sponge_fraction = 0.0125  # fraction of domain where sponge
# Temperature source term
source_temp = conditional(x > (1.0-sponge_fraction) * L,
                           absorption_factor * T_restore *((x - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
                          0.0)

# Salinity source term
source_sal = conditional(x > (1.0-sponge_fraction) * L,
                         absorption_factor * S_restore  *((x - (1.0-sponge_fraction) * L)/(L * sponge_fraction)), 
                         0.0)

# Temperature absorption term
absorp_temp = conditional(x > (1.0-sponge_fraction) * L,
                          absorption_factor * ((x - (1.0-sponge_fraction) * L)/(L * sponge_fraction)),
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


#kappa_v = Function(P1_extruded) 
DeltaS = Constant(1.0)  # rough order of magnitude estimate of change in salinity over restoring region
gradrho_scale = DeltaS * beta_sal / water_depth  # rough order of magnitude estimate for vertical gradient of density anomaly. units m^-1
#kappa_v.assign(conditional(gradrho / gradrho_scale < 1e-1, 1e-3, 1e-1))

mu = as_tensor([[mu_h, 0, 0], [0, mu_h, 0], [0, 0, mu_v]])
kappa = as_tensor([[kappa_h, 0, 0], [0, kappa_h, 0], [0, 0, kappa_v]])

kappa_temp = kappa
kappa_sal = kappa

##########

# Equation fields
vp_coupling = [{'pressure': 1}, {'velocity': 0}]
vp_fields = {'viscosity': mu, 'source': mom_source, 'interior_penalty': Constant(3.0),
            'coriolis_frequency': f}
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
#Q_ice.interpolate(mp.Q_ice)
#Q_mixed.interpolate(mp.Q_mixed)
#Q_latent.interpolate(mp.Q_latent)
#Q_s.interpolate(mp.S_flux_bc)
melt.interpolate(mp.wb)
#Tb.interpolate(mp.Tb)
#Sb.interpolate(mp.Sb)
#full_pressure.interpolate(mp.P_full)

##########



##########

# Boundary conditions
# top boundary: no normal flow, drag flowing over ice
# bottom boundary: no normal flow, drag flowing over bedrock
# grounding line wall (LHS): no normal flow
# open ocean (RHS): pressure to account for density differences

# WEAKLY Enforced BCs
n = FacetNormal(mesh)
Temperature_term = -beta_temp * z * (T_surface + 0.5 * (T_bottom - T_surface) * (z / -water_depth))
Salinity_term = beta_sal * z * (S_surface + 0.5 * (S_bottom - S_surface) * (z / -water_depth))
stress_open_boundary = -n*-g*(Temperature_term + Salinity_term)
no_normal_flow = 0.


# test stress open_boundary
#sop = Function(W)
#sop.interpolate(-g*(Temperature_term + Salinity_term))
#sop_file = File(folder+"boundary_stress.pvd")
#sop_file.write(sop)


vp_bcs = {"top": {'un': no_normal_flow, 'drag': conditional(x < shelf_length, 2.5E-3, 0.0)}, 
        1: {'un': no_normal_flow}, 2: {'stress': stress_open_boundary}, 
        3: {'un': no_normal_flow}, 4: {'un': no_normal_flow}, 
        "bottom": {'un': no_normal_flow, 'drag': 2.5E-3}} 

temp_bcs = {"top": {'flux': conditional(x + 5.0 * dy < shelf_length, -mp.T_flux_bc, 0.0)}}

sal_bcs = {"top": {'flux':  conditional(x + 5.0 * dy < shelf_length, -mp.S_flux_bc, 0.0)}}


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
        'snes_monitor': None,
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',  # we solve the full schur complement exactly, so no need for outer krylov
#        'ksp_monitor_true_residual': None,
        'mat_type': 'matfree',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_fact_type': 'full',
        # velocity mass block:
        'fieldsplit_0': {
            'ksp_converged_reason': None,
#            'ksp_monitor_true_residual': None,
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
            'laplace_pc_type': 'ksp',
            'laplace_ksp_ksp_type': 'cg',
            'laplace_ksp_ksp_rtol': 1e-7,
            'laplace_ksp_ksp_atol': 1e-9,
            'laplace_ksp_ksp_converged_reason': None,
#            'laplace_ksp_ksp_monitor_true_residual': None,
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
        'pc_hypre_boomeramg_strong_threshold': 0.6,
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
temp_solver_parameters = gmres_solver_parameters
sal_solver_parameters = gmres_solver_parameters



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
                                                          theta=0.5,
                                                          predictor_solver_parameters=predictor_solver_parameters,
                                                          picard_iterations=1,
                                                          pressure_nullspace=VectorSpaceBasis(constant=True))
# performs pseudo timestep to get good initial pressure
# this is to avoid inconsistencies in terms (viscosity and advection) that
# are meant to decouple from pressure projection, but won't if pressure is not initialised
# do this here, so we can see the initial pressure in pressure_0.pvtu
if not DUMP:
    # should not be done when picking up
    with timed_stage('initial_pressure'):
        for i in range(1):
            vp_timestepper.initialize_pressure()

temp_timestepper = DIRK33(temp_eq, temp, temp_fields, dt, temp_bcs, solver_parameters=temp_solver_parameters)
sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)

##########

# Set up Vectorfolder
folder = "/data/3d_isomip_plus/extruded_meshes/"+str(args.date)+"_3d_isomip+_dt"+str(dt)+\
         "_dtOut"+str(output_dt)+"_T"+str(T)+"_StratLinTres"+str(restoring_time.values()[0])+\
         "_Muh"+str(mu_h.values()[0])+"_fixMuv"+str(mu_v.values()[0])+"_Kh"+str(kappa_h.values()[0])+"_fixKv"+str(kappa_v.values()[0])+\
         "_dx"+str(round(1e-3*dy))+"km_lay"+str(args.nz)+"_closed_coriolis_tracerlims_ip3_alignicefront_cranknic/"
         #+"_extended_domain_with_coriolis_stratified/"  # output folder.
#folder = 'tmp/'


###########

# Output files for velocity, pressure, temperature and salinity
#v_file = File(folder+"velocity.pvd")  # for some reason velocity doesn't work in paraview - to do with squeezed triangles/wedges?
#v_file.write(v_)

# Output files for velocity, pressure, temperature and salinity
#vdg.project(v_) # DQ2 velocity for output
vdg_projector = Projector(v_, vdg)
vdg_projector.project()
vdg_file = File(folder+"dg_velocity.pvd")
vdg_file.write(vdg)

#vdg1.project(v_) # DQ1 velocity 
#vdg1_file = File(folder+"P1dg_velocity_lim.pvd")
#vdg1_file.write(vdg1)

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
#Q_ice_file = File(folder+"Q_ice.pvd")
#Q_ice_file.write(Q_ice)

#Q_mixed_file = File(folder+"Q_mixed.pvd")
#Q_mixed_file.write(Q_mixed)

#Qs_file = File(folder+"Q_s.pvd")
#Qs_file.write(Q_s)

m_file = File(folder+"melt.pvd")
m_file.write(melt)

#full_pressure_file = File(folder+"full_pressure.pvd")
#full_pressure_file.write(full_pressure)

#u_pred_star_file = File(folder+"u_star.pvd")
#u_pred_dg.project(vp_timestepper.u_star) # DQ2 velocity for output
#u_pred_star_file.write(u_pred_dg)


##########

with DumbCheckpoint(folder+"initial_pressure_dump", mode=FILE_UPDATE) as chk:
    # Checkpoint file open for reading and writing
    chk.store(v_, name="velocity")
    chk.store(p_, name="perturbation_pressure")
    chk.store(temp, name="temperature")
    chk.store(sal, name="salinity")


####################

# Add limiter for DG functions
limiter = VertexBasedP1DGLimiter(S)
v_comp = Function(S)
w_comp = Function(S)
########

# Begin time stepping
t = 0.0
step = 0

while t < T - 0.5*dt:
    with timed_stage('velocity-pressure'):
        vp_timestepper.advance(t)
        #vdg.project(v_)  # DQ2 velocity for melt and plotting
        vdg_projector.project()
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

    rho_anomaly_projector.project()
    gradrho_projector.project()
    #kappa_v_projector.project()
    
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
  #         Q_ice.interpolate(mp.Q_ice)
  #         Q_mixed.interpolate(mp.Q_mixed)
  #         Q_latent.interpolate(mp.Q_latent)
  #         Q_s.interpolate(mp.S_flux_bc)
           melt.interpolate(mp.wb)
  #         Tb.interpolate(mp.Tb)
  #         Sb.interpolate(mp.Sb)
  #         full_pressure.interpolate(mp.P_full)
    

           
            # Write out files
 #          u_pred_dg.project(vp_timestepper.u_star) # DQ2 velocity for output
 #          u_pred_star_file.write(u_pred_dg)
    #       v_file.write(v_)
           vdg_file.write(vdg)
   #        vdg1_file.write(vdg1)
           p_file.write(p_)
           t_file.write(temp)
           s_file.write(sal)
           rho_file.write(rho)
               
           rhograd_file.write(gradrho)
#          kappav_file.write(kappa_v)
          # Write melt rate functions
           m_file.write(melt)
 #          Q_mixed_file.write(Q_mixed)
 #          full_pressure_file.write(full_pressure)
 #          Qs_file.write(Q_s)
 #          Q_ice_file.write(Q_ice)
    
           time_str = str(step)
           #top_boundary_to_csv(shelf_boundary_points, top_boundary_mp, time_str)
    
           #depth_profile_to_csv(depth_profile500m, velocity_depth_profile500m, "500m", time_str)
           #depth_profile_to_csv(depth_profile1km, velocity_depth_profile1km, "1km", time_str)
           #depth_profile_to_csv(depth_profile2km, velocity_depth_profile2km, "2km", time_str)
           #depth_profile_to_csv(depth_profile4km, velocity_depth_profile4km, "4km", time_str)
           #depth_profile_to_csv(depth_profile6km, velocity_depth_profile6km, "6km", time_str)
    
           PETSc.Sys.Print("t=", t)
    
           PETSc.Sys.Print("integrated melt =", assemble(conditional(x < shelf_length, melt, 0.0) * ds("top")))

    if t % (3600 * 24) == 0:
        with DumbCheckpoint(folder+"dump_step_{}.h5".format(step), mode=FILE_CREATE) as chk:
            # Checkpoint file open for reading and writing at regular interval
            chk.store(v_, name="velocity")
            chk.store(p_, name="perturbation_pressure")
            chk.store(temp, name="temperature")
            chk.store(sal, name="salinity")
