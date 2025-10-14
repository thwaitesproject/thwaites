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
#from firedrake.meshadapt import *
#from pyroteus import *
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
parser.add_argument("angle", help="angle of the crevasse wrt flow direction (along/across/negative/positive)")
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
dy = 25.0
ny = round(L/dy)
#nz = 50
dz = 1.0

# Define a dump file
dump_file = "/g/data/xd2/ws9229/ice_ocean/3d_crevasse/07.05.24_5kmflume_3_eq_param_ufricHJ99_dt60.0_dtOutput86400.0_T8640000.0_3d_along_openbox_bodyforce_posv_qice0_dz5m_dx20mto250m_scalehorvis_consTS_qadv_balp_nocutoff/dump.h5"

DUMP = False

# create mesh
#mesh = Mesh("./5km_flume_nocrevasse_refined.msh")
#mesh = Mesh("./3d_crevasse_flume_dx250mto20m_dz5m_crevdxz5m.msh")

# do initialisation again.... 
water_depth = 600
if not DUMP:
   # mesh = Mesh(f"./3d_crevasse_flume_dx250mto20m_dz5m_crevdxz5m_{args.angle}.msh", name=f"3dcrevasse_{args.angle}")
    mesh = Mesh(f"./3d_crevasse_flume_dx250mto20m_dz2to5to10m_crevdxz5m_along.msh", name=f"3dcrevasse_{args.angle}")
    mesh.coordinates.dat.data[:, 2] -= 500 #water_depth
else:
    with CheckpointFile(dump_file, 'r') as chk:
        mesh = chk.load_mesh(f"3dcrevasse_{args.angle}") 

x, y, z = SpatialCoordinate(mesh)
#mesh = BoxMesh(50,50,20, 5000,5000,100)
PETSc.Sys.Print("Mesh dimension ", mesh.geometric_dimension())
# shift z = 0 to surface of ocean. N.b z = 0 is outside domain.
#PETSc.Sys.Print("Length of lhs", assemble(Constant(1.0)*ds(3, domain=mesh)))

#PETSc.Sys.Print("Length of rhs", assemble(Constant(1.0)*ds(2, domain=mesh)))

PETSc.Sys.Print("Length of bottom", assemble(Constant(1.0)*ds(1, domain=mesh)))

PETSc.Sys.Print("Length of top", assemble(Constant(1.0)*ds(4, domain=mesh)))

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
P0 = FunctionSpace(mesh,"DG",0)
P1 = FunctionSpace(mesh,"CG",1)
print("velocity dofs:", V.dim())
print("Pressure dofs:", W.dim())
print("scalar dofs:", Q.dim())
##########

# Set up functions
m = Function(M)
v_, p_ = m.subfunctions  # function: y component of velocity, pressure
v, p = split(m)  # expression: y component of velocity, pressure
v_.rename("velocity")
p_.rename("pressure")
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
p_b = Function(W, name="balance pressure")

frazil = Function(Q, name="frazil") # should this really be P0dg to prevent negative frazil ice?
frazil_flux = Function(Q, name="frazil ice flux") 
##########

if DUMP:
    with CheckpointFile(dump_file, 'r') as chk:
        # Checkpoint file open for reading and writing
        v_ = chk.load_function(mesh, "velocity")
        p_ = chk.load_function(mesh, "pressure")
        #chk.load(u, name="u_velocity")
        sal = chk.load_function(mesh, "salinity")
        temp = chk.load_function(mesh, "temperature")
        frazil = chk.load_function(mesh, "frazil")

        # ISOMIP+ warm conditions .
        T_surface = -1.96  # This gives T = -0.4degC at 520m depth
        T_bottom = -0.1

        S_surface = 33.725 # This gives S = 34.375 PSU at 520m depth
        S_bottom = 34.5
        
        T_restore = T_bottom #T_surface + (T_bottom - T_surface) * (z / -water_depth)
        S_restore = S_bottom # S_surface + (S_bottom - S_surface) * (z / -water_depth)


else:
    # Assign Initial conditions
    v_init = zero(mesh.geometric_dimension())
    v_.assign(v_init)

    T_surface = -1.96  # This gives T = -0.4degC at 520m depth
    T_bottom = -0.1

    S_surface = 33.725 # This gives S = 34.375 PSU at 520m depth
    S_bottom = 34.5
    
    T_restore = T_bottom #T_surface + (T_bottom - T_surface) * (z / -water_depth)
    S_restore = S_bottom #S_surface + (S_bottom - S_surface) * (z / -water_depth)
    # baseline T3


    temp_init = T_restore
    temp.interpolate(temp_init)

    sal_init = S_restore
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
balance_pressure_eq = BalancePressureEquation(W,W)
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
f = Constant(-1.409E-4)
horizontal_stress = -f * 0.01  # geostrophic stress ~ |f v| drives a flow of 0.01 m/s?

ramp = Constant(0.0)
if DUMP:
    ramp.assign(1)
#horizontal_source = as_vector((0.0, conditional(z<-500, horizontal_stress, 0.0), 0.0)) * ramp 
horizontal_source = as_vector((0.0, horizontal_stress, 0.0)) * ramp 
#horizontal_source = as_vector((conditional(z<-500, horizontal_stress, 0.0), conditional(z<-500, horizontal_stress, 0.0), 0.0)) * ramp * Constant(1./np.sqrt(2.)) 
#horizontal_source = as_vector((0.0,horizontal_stress,0.0)) * ramp
# Scalar source/sink terms at open boundary.
absorption_factor = Constant(1.0/restoring_time)
sponge_fraction = 0.06  # fraction of domain where sponge
# Temperature source term

class SmagorinskyViscosity(object):
    r"""
    Computes Smagorinsky subgrid scale horizontal viscosity

    This formulation is according to Ilicak et al. (2012) and
    Griffies and Hallberg (2000).

    .. math::
        \nu = (C_s \Delta x)^2 |S|

    with the deformation rate

    .. math::
        |S| &= \sqrt{D_T^2 + D_S^2} \\
        D_T &= \frac{\partial u}{\partial x} - \frac{\partial v}{\partial y} \\
        D_S &= \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}

    :math:`\Delta x` is the horizontal element size and :math:`C_s` is the
    Smagorinsky coefficient.

    To match a certain mesh Reynolds number :math:`Re_h` set
    :math:`C_s = 1/\sqrt{Re_h}`.

    Ilicak et al. (2012). Spurious dianeutral mixing and the role of
    momentum closure. Ocean Modelling, 45-46(0):37-58.
    http://dx.doi.org/10.1016/j.ocemod.2011.10.003

    Griffies and Hallberg (2000). Biharmonic friction with a
    Smagorinsky-like viscosity for use in large-scale eddy-permitting
    ocean models. Monthly Weather Review, 128(8):2935-2946.
    http://dx.doi.org/10.1175/1520-0493(2000)128%3C2935:BFWASL%3E2.0.CO;2
    """
    def __init__(self, uv, output, c_s, h_elem_size, max_val, min_val=1e-10,
                 weak_form=True, solver_parameters=None):
        """
        :arg uv_3d: horizontal velocity
        :type uv_3d: 3D vector :class:`Function`
        :arg output: Smagorinsky viscosity field
        :type output: 3D scalar :class:`Function`
        :arg c_s: Smagorinsky coefficient
        :type c_s: float or :class:`Constant`
        :arg h_elem_size: field that defines the horizontal element size
        :type h_elem_size: 3D scalar :class:`Function` or :class:`Constant`
        :arg float max_val: Maximum allowed viscosity. Viscosity will be clipped at
            this value.
        :kwarg float min_val: Minimum allowed viscosity. Viscosity will be clipped at
            this value.
        :kwarg bool weak_form: Compute velocity shear by integrating by parts.
            Necessary for some function spaces (e.g. P0).
        :kwarg dict solver_parameters: PETSc solver options
        """
        if solver_parameters is None:
            solver_parameters = {}
        solver_parameters.setdefault('ksp_atol', 1e-12)
        solver_parameters.setdefault('ksp_rtol', 1e-16)
#        assert max_val.function_space() == output.function_space(), \
#            'max_val function must belong to the same space as output'
        self.max_val = max_val
        self.min_val = min_val
        self.output = output
        self.weak_form = weak_form


        if self.weak_form:
            # solve grad(u) weakly
            mesh = output.function_space().mesh()
            fs_grad = get_functionspace(mesh, 'DP', 1, 'DP', 1, vector=True, dim=4)
            self.grad = Function(fs_grad, name='uv_grad')

            tri_grad = TrialFunction(fs_grad)
            test_grad = TestFunction(fs_grad)

            normal = FacetNormal(mesh)
            a = inner(tri_grad, test_grad)*dx

            rhs_terms = []
            for iuv in range(2):
                for ix in range(2):
                    i = 2*iuv + ix
                    vol_term = -inner(Dx(test_grad[i], ix), uv[iuv])*dx
                    int_term = inner(avg(uv[iuv]), jump(test_grad[i], normal[ix]))*dS_v
                    ext_term = inner(uv[iuv], test_grad[i]*normal[ix])*ds_v
                    rhs_terms.extend([vol_term, int_term, ext_term])
            l = sum(rhs_terms)
            prob = LinearVariationalProblem(a, l, self.grad)
            self.weak_grad_solver = LinearVariationalSolver(prob, solver_parameters=solver_parameters)

            # rate of strain tensor
            d_t = self.grad[0] - self.grad[3]
            d_s = self.grad[1] + self.grad[2]
        else:
            # rate of strain tensor
            d_t = Dx(uv[0], 0) - Dx(uv[1], 1)
            d_s = Dx(uv[0], 1) + Dx(uv[1], 0)

        fs = output.function_space()
        tri = TrialFunction(fs)
        test = TestFunction(fs)

        nu = c_s**2*h_elem_size**2 * sqrt(d_t**2 + d_s**2)

        a = test*tri*dx
        l = test*nu*dx
        self.prob = LinearVariationalProblem(a, l, output)
        self.solver = LinearVariationalSolver(self.prob, solver_parameters=solver_parameters)

    def solve(self):
        """Compute viscosity"""
        if self.weak_form:
            self.weak_grad_solver.solve()
        self.solver.solve()
        # remove negative values

smag_solver_parameters = {
        'snes_monitor': None,
        'snes_type': 'ksponly',
        'ksp_type': 'gmres',
        'pc_type': 'sor',
        'ksp_converged_reason': None,
#        'ksp_monitor_true_residual': None,
        'ksp_rtol': 1e-5,
        'ksp_max_it': 300,
        }

smag_visc = Function(P0)
c_s = Constant(1./np.sqrt(2.))  # Grid Re = 2
max_nu = 10 * 1.0 * dy / 2.0  # 10x U dx / 2 to have grid reynolds 
smag_visc_solver = SmagorinskyViscosity(v_, smag_visc, c_s, Constant(dy), max_nu, weak_form=False, solver_parameters=smag_solver_parameters)
smag_visc_solver.solve()

cell_size = Function(P1)
cell_size.interpolate(CellDiameter(mesh)/5.0)

kappa = as_tensor([[1e-3*cell_size, 0, 0], [0, 1e-3*cell_size, 0], [0, 0, 1e-3]])
mu = as_tensor([[1e-3*cell_size, 0, 0], [0, 1e-3*cell_size, 0], [0, 0, 1e-3]])

kappa_temp = kappa
kappa_sal = kappa
kappa_frazil = kappa

FRV = FrazilRisingVelocity(0.1)  # initial velocity guess needs to be >0
w_i = FRV.frazil_rising_velocity() # Picard iterations converge to value for w_i (which only depends on crystal size, here we assume r =7.5e-4m

frazil_mp = FrazilMeltParam(sal, temp, p, z, frazil)
temp_source =  0 #(frazil_mp.Tc - temp - frazil_mp.Lf/frazil_mp.c_p_m) * frazil_mp.wc
temp_absorption = 0 
sal_source = 0 #-sal *frazil_mp.wc
sal_absorption = 0 
frazil_source = 0 # -frazil_mp.wc
frazil_absorption = 0

# Interior penalty term
# 3*cot(min_angle)*(p+1)*p*nu_max/nu_min

ip_alpha = Constant(3*dy/dz*2*ip_factor)
# Equation fields
vp_coupling = [{'pressure': 1}, {'velocity': 0}]
vp_fields = {'viscosity': mu, 'source': mom_source+horizontal_source, 'coriolis_frequency': f, 'balance_pressure': p_b} #, 'interior_penalty': ip_alpha}
#u_fields = {'diffusivity': mu, 'velocity': v, 'interior_penalty': ip_alpha, 'coriolis_frequency': f}
temp_fields = {'diffusivity': kappa_temp, 'velocity': v, 'source': temp_source, 'absorption coefficient': temp_absorption}
sal_fields = {'diffusivity': kappa_sal, 'velocity': v, 'source': sal_source, 'absorption coefficient': sal_absorption, }
frazil_fields = {'diffusivity': kappa_frazil, 'velocity': v, 'w_i': Constant(w_i), 'source': frazil_source, 'absorption coefficient': frazil_absorption}
p_b_fields = {'buoyancy': mom_source} #, 'velocity': v}

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
#Temperature_term = -beta_temp * (T_surface * z + 0.5 * (T_bottom - T_surface) * (pow(z,2) / -water_depth) - T_ref * z)
#Salinity_term = beta_sal *  (S_surface * z + 0.5 * (S_bottom - S_surface) * (pow(z,2) / -water_depth) - S_ref * z)
stress_open_boundary = -n*-g*(Temperature_term + Salinity_term)
no_normal_flow = 0.
ice_drag = 0.0097


# test stress open_boundary
#sop = Function(W)
#sop.interpolate(-g*(Temperature_term + Salinity_term))
#sop_file = File(folder+"boundary_stress.pvd")
#sop_file.write(sop)


vp_bcs = {4: {'un': no_normal_flow, 'drag': ice_drag}, 5: {}, 
        6: {}, 1: {'un': no_normal_flow, 'drag': 2.5e-3},
        2: {}, 3: {}} # gmsh bcs
#vp_bcs = {1: {'un': no_normal_flow, 'drag': ice_drag}, 2: {'stress': stress_open_boundary}, # gmsh no crevasse extruded
#        3: {'stress': stress_open_boundary}, 6: {'un': no_normal_flow, 'drag': 2.5e-3}, # gmsh no crevasse extruded
#        5: {'stress': stress_open_boundary}, 4: {'stress': stress_open_boundary}} # gmsh no crevasse extruded
#u_bcs = {2: {'q': Constant(0.0)}}

temp_bcs = {4: {'flux': -mp.T_flux_bc}, 3: {'qadv': T_restore}, 5: {'qadv': T_restore}, 6: {'qadv': T_restore}, 2: {'qadv': T_restore}}
#temp_bcs = {3: {'q': T_restore}, 5: {'q': T_restore}, 6: {'q': T_restore}, 2: {'q': T_restore}} # gmsh bcs
#temp_bcs = {1: {'flux': -mp.T_flux_bc}, 5: {'q': T_restore}, 2: {'q': T_restore}, 3: {'q': T_restore}, 4: {'q': T_restore}} # gmsh no crevasse extruded

sal_bcs = {4: {'flux': -mp.S_flux_bc}, 3:{'qadv': S_restore}, 5:{'qadv': S_restore}, 6:{'qadv': S_restore}, 2:{'qadv': S_restore}}
#sal_bcs = {3:{'q': S_restore}, 5:{'q': S_restore}, 6:{'q': S_restore}, 2:{'q': S_restore}} # gmsh bcs
#sal_bcs = {1: {'flux': -mp.T_flux_bc}, 5:{'q': S_restore}, 2:{'q': S_restore}, 3:{'q': S_restore}, 4:{'q': S_restore}} # gmsh no crevasse extruded

frazil_bcs = {}
p_b_bcs = {}

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
        'ksp_max_it': 1000,
        }
vp_solver_parameters = pressure_projection_solver_parameters
u_solver_parameters = mumps_solver_parameters
temp_solver_parameters = gmres_solver_parameters
sal_solver_parameters = gmres_solver_parameters
frazil_solver_parameters = gmres_solver_parameters
p_b_solver_parameters = predictor_solver_parameters

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

p_b_solver = BalancePressureSolver(balance_pressure_eq, p_b, p_b_fields,p_b_bcs, solver_parameters=p_b_solver_parameters,
                                    p_b_nullspace=VectorSpaceBasis(constant=True))
p_b_solver.advance(0)
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
folder = f"/g/data/xd2/ws9229/ice_ocean/3d_crevasse/{args.date}_5kmflume_3_eq_param_ufricHJ99_dt{dt}_dtOutput{output_dt}_T{T}_3d_{args.angle}_openbox_bodyforce_posv_qice0_dz5m_dx20mto250m_scalehorvis_consTS_qadv_balp_nocutoff_refdz2out/"
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


smag_visc_file = File(folder+"smag_visc.pvd")
smag_visc_file.write(smag_visc)
cellsize_file = File(folder+"cell_size.pvd")
cellsize_file.write(cell_size)

p_b_file = File(folder+"balance_pressure.pvd")
p_b_file.write(p_b)
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

if DUMP:
    t+=7*86400

while t < T - 0.5*dt:
    with timed_stage('velocity-pressure'):
        p_b_solver.advance(t)
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

    smag_visc_solver.solve()
    if not DUMP:
        if t <= 86400:
            ramp.assign(t/86400)
    with timed_stage('output'):
       if step % output_step == 0:
           # dumb checkpoint for starting from last timestep reached
           with CheckpointFile(folder+"dump.h5", 'w') as chk:
               # Checkpoint file open for reading and writing
               chk.save_mesh(mesh)
               chk.save_function(v_, name="velocity")
               chk.save_function(p_, name="pressure")
               #chk.store(u, name="u_velocity")
               chk.save_function(temp, name="temperature")
               chk.save_function(sal, name="salinity")
               chk.save_function(frazil, name="frazil")
    
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
             #  frazil_file.write(frazil)   
               # Write melt rate functions
               m_file.write(melt)
            #   Q_mixed_file.write(Q_mixed)
             #  full_pressure_file.write(full_pressure)
             #  Qs_file.write(Q_s)
             #  Q_ice_file.write(Q_ice)
             #  frazil_flux_file.write(frazil_flux)
               smag_visc_file.write(smag_visc)
               p_b_file.write(p_b)
           time_str = str(step)
           #top_boundary_to_csv(shelf_boundary_points, top_boundary_mp, time_str)
    
   #        depth_profile_to_csv(depth_profile500m, velocity_depth_profile500m, "500m", time_str)
    #       depth_profile_to_csv(depth_profile1km, velocity_depth_profile1km, "1km", time_str)
     #      depth_profile_to_csv(depth_profile2km, velocity_depth_profile2km, "2km", time_str)
      #     depth_profile_to_csv(depth_profile4km, velocity_depth_profile4km, "4km", time_str)
       #    depth_profile_to_csv(depth_profile6km, velocity_depth_profile6km, "6km", time_str)
    
           PETSc.Sys.Print("t=", t)
    
           PETSc.Sys.Print("integrated melt =", assemble(melt * ds(4)))

    #if step % (output_step ) == 0:
    #    with DumbCheckpoint(folder+"dump_step_{}.h5".format(step), mode=FILE_CREATE) as chk:
    #        # Checkpoint file open for reading and writing at regular interval
    #        chk.store(v_, name="v_velocity")
    #        chk.store(p_, name="perturbation_pressure")
    #        #chk.store(u, name="u_velocity")
    #        chk.store(temp, name="temperature")
    #        chk.store(sal, name="salinity")
    #        chk.store(frazil, name="frazil ice concentration")
