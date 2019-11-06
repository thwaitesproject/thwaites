# Buoyancy driven overturning circulation
# beneath ice shelf. Wedge geometry. 5km
# Outside temp forcing 3.0degC above freezing point for walter (S = 34.5) at 1000m depth.
# viscosity = temp diffusivity = sal diffusivity: varies linearly over the domain.
from thwaites import *
import numpy as np
from firedrake.petsc import PETSc
from firedrake import FacetNormal
from thwaites.coupled_integrators import CoupledEquationsTimeIntegrator

folder = "output_coupled/"  # output folder.
L = 5*1e3
H1 = 1
H2 = 100
dx = 50
nx = round(L/dx)
nz = 10
dz = H1/nz

mesh = RectangleMesh(nx, nz, L, 1)
x = mesh.coordinates.dat.data[:,0]
y = mesh.coordinates.dat.data[:,1]
mesh.coordinates.dat.data[:,1] = ((x/L)*(H2-H1) + H1)*y

print("You have Comm WORLD size = ", mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)

# Set up function spaces
V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
W = FunctionSpace(mesh, "CG", 2)  # pressure space

Q = FunctionSpace(mesh, "DG", 1)  # density space
K = FunctionSpace(mesh,"DG",1)    # temperature space
S = FunctionSpace(mesh,"DG",1)    # salinity space
M = MixedFunctionSpace([V,W,K,S])

# Set up functions
m = Function(M)
u_, p_, T_, S_ = m.split()
u, p, T, S = split(m)

melt = Function(Q)
Q_mixed = Function(Q)
Q_ice = Function(Q)
Q_latent = Function(Q)
Q_s = Function(K)
Tb = Function(Q)
Sb = Function(Q)
full_pressure = Function(M.sub(1))


# ice thickness shouldn't be used!!! unless the water depth is not relative to 1km!!!
def ice_thickness(x,x0,y0,x1,y1):
    m = (y1-y0)/(x1-x0)
    return y0 + m*x

def cavity_thickness(x,x0,y0,x1,y1):
    m = (y1-y0)/(x1-x0)
    return y0 + m*x

x,z = SpatialCoordinate(mesh)

# shift z = 0 to surface of ocean. This is outside domain.
h_ice = ice_thickness(x,0.0,999.0,5000.0,900.0)
h_cav = cavity_thickness(x,0.0,1.0,5000.0,100.0)
water_depth = 1000.0
cavity_depth = h_cav - water_depth
z = z -water_depth

u_init = Constant((1e-7, 1e-7))
u_.assign(u_init)


# linearly vary viscosity/diffusivities over domain.
kappa_h = Constant(20)
kappa_v = 2*(3.96E-4*x + 0.02)
kappa = as_tensor([[kappa_h, 0], [0, kappa_v]])
kappa_temp = kappa
kappa_sal = kappa
mu_h = Constant(20)
mu_v = 2*(3.96E-4*x + 0.02)
mu = as_tensor([[mu_h, 0], [0, mu_v]])


Freezing_temp_at_GL = MeltRateParam(34.5, 0, 0,0,-water_depth,1).freezing_point()
delT = 3.0

Trestore = Freezing_temp_at_GL+delT
Srestore = 34.5 #PSU

temp_init = Constant(Trestore)   # Tinit =Tres = 0.37 deg C from ben's thesis page 54
T_.interpolate(temp_init)

sal_init = Constant(34.4) # stationay flow S = Srest#set this to slightly less than Sres. horizontal pressure gradient increases as depth increases. pressure greater outside domain so drives flow in!
S_.interpolate(sal_init)


# We declare the output filenames, and write out the initial conditions. ::
u_file = File(folder+"velocity.pvd")
u_file.write(u_)
p_file = File(folder+"pressure.pvd")
p_file.write(p_)

t_file = File(folder+"temperature.pvd")
t_file.write(T_)
s_file = File(folder+"salinity.pvd")
s_file.write(S_)

#
mom_eq = MomentumEquation(M.sub(0), M.sub(0))
cty_eq = ContinuityEquation(M.sub(1), M.sub(1))
temp_eq = ScalarAdvectionDiffusionEquation(M.sub(2), M.sub(2))
sal_eq = ScalarAdvectionDiffusionEquation(M.sub(3), M.sub(3))

# momentum source: the buoyancy term boussinesq approx. From Ben's thesis page 31
Tref= 0.0
Sref=34.8 #PSU
beta_temp = 3.87*10E-5 #5.0E-3
beta_sal=7.86*10E-4
g = 9.81

mom_source = as_vector((0, -g))*(-beta_temp*(T-Tref)+beta_sal*(S-Sref))

# 3*cot(min_angle)*(p+1)*p*nu_max/nu_min
ip_alpha = 3*dx/dz*2

up_fields = {'viscosity': mu, 'source': mom_source, 'interior_penalty': ip_alpha, 'pressure': p, 'velocity': u}
temp_fields = {'diffusivity': kappa_temp, 'velocity': u, 'interior_penalty': ip_alpha}
sal_fields = {'diffusivity': kappa_sal, 'velocity': u, 'interior_penalty': ip_alpha}
upts_fields = [up_fields, up_fields, temp_fields, sal_fields]

mumps_solver_parameters = {
    'snes_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij',
    'snes_max_it': 1000,
    'snes_atol': 1e-16,
}

# Get expressions for calculating functions used in meltrate parameterisation
dz = 20 # This really needs to change. implement as classes. Oct 2019
q_ice, q_mixed, q_mixed_real, q_latent, q_s, wb, t_b, s_b, pfull = MeltRateParam(S, T, p,cavity_depth,z,dz).three_eq_param_meltrate()

# assign values of these expressions to functions.
# so these alter the expression and give new value for functions.
Q_ice.interpolate(q_ice)
Q_mixed.interpolate(q_mixed)

Q_latent.interpolate(q_latent)
Q_s.interpolate(q_s)
melt.interpolate(wb)
Tb.interpolate(t_b)
Sb.interpolate(s_b)
full_pressure.interpolate(pfull)

# this is where write first file for above functions.
Q_ice_file = File(folder+"Q_ice.pvd")
Q_ice_file.write(Q_ice)

Q_mixed_file = File(folder+"Qmixed.pvd")
Q_mixed_file.write(Q_mixed)

Qs_file = File(folder+"Q_s.pvd")
Qs_file.write(Q_s)

m_file = File(folder+"melt.pvd")
m_file.write(melt)

full_pressure_file = File(folder+"full_pressure.pvd")
full_pressure_file.write(full_pressure)


########
# Plotting top boundary. really ought to be a separate function in utility

n = 100
dx = 5000. / float(n)
x1 = np.array([i * dx for i in range(n)])

shelf_boundary_points = []
for i in range(n):
    x_i = i * dx
    y_i = cavity_thickness(x_i, 0.0, 1.0, 5000.0, 100.0) - 0.01
    shelf_boundary_points.append([x_i, y_i])

import pandas as pd
df = pd.DataFrame()
df['Qice_t_0.0'] = Q_ice.at(shelf_boundary_points)
df['Qmixed_t_0.0'] = Q_mixed.at(shelf_boundary_points)
df['Qlat_t_0.0'] = Q_latent.at(shelf_boundary_points)
df['Qsalt_t_0.0'] = Q_s.at(shelf_boundary_points)
df['Melt_t_0.0'] = melt.at(shelf_boundary_points)
df['Tb_t_0.0']= Tb.at(shelf_boundary_points)
df['P_t_0.0'] = full_pressure.at(shelf_boundary_points)
df['Sal_t_0.0'] = S_.at(shelf_boundary_points)
df['Temp_t_0.0'] = T_.at(shelf_boundary_points)

if mesh.comm.rank ==0:
    print(df)
    df.to_csv(folder+"top_boundary_data.csv")

##########

temp_bcs = {2: {'q': Trestore}, 4: {'flux': -Q_mixed}}
temp_solver_parameters = mumps_solver_parameters

sal_bcs ={2: {'q': Srestore}, 4: {'flux': -Q_s}}
sal_solver_parameters = mumps_solver_parameters

n = FacetNormal(mesh)
stress_open_boundary = -n*-g*(-beta_temp*(Trestore-Tref)+beta_sal*(Srestore-Sref))*z
no_normal_flow = 0.
ice_drag = 0.0097

# WEAKLY Enforced BCs
# top boundary: no normal flow, drag flowing over ice
# bottom boundary: no normal flow, drag flowing over bedrock
# grounding line wall (LHS): no normal flow
# open ocean (RHS): pressure to account for density difference
up_bcs = {4: {'un': no_normal_flow, 'drag': ice_drag}, 2: {'stress': stress_open_boundary},
          3: {'un': no_normal_flow, 'drag': 0.0025}, 1: {'un': no_normal_flow}}
upts_bcs = [up_bcs, up_bcs, temp_bcs, sal_bcs]

# STRONGLY Enforced BCs
# open ocean (RHS): no tangential flow because viscosity of outside ocean resists vertical flow.
strong_bcs = [DirichletBC(M.sub(0).sub(1), 0, 2)]

upts_solver_parameters = mumps_solver_parameters


## define time steps
T = 3600*24*120
dt = 3600.
output_dt = dt
output_step = output_dt/dt

upts_timestepper = CoupledEquationsTimeIntegrator([mom_eq, cty_eq, temp_eq, sal_eq], m, upts_fields, dt, upts_bcs,
                                                        solver_parameters=upts_solver_parameters, mass_terms=[True, False, True, True], strong_bcs=strong_bcs)

t = 0.0
step = 0

while t < T - 0.5*dt:
    upts_timestepper.advance(t)

    # Update melt rate functions
    melt.interpolate(wb)
    Q_mixed.interpolate(q_mixed)
    Q_ice.interpolate(q_ice)
    Q_latent.interpolate(q_latent)
    Tb.interpolate(t_b)
    Sb.interpolate(s_b)
    full_pressure.interpolate(pfull)
    Q_s.interpolate(q_s)

    step += 1
    t += dt

    if step % output_step == 0:
        u_file.write(u_)
        p_file.write(p_)
        #d_file.write(rho)
        t_file.write(T_)
        s_file.write(S_)
        m_file.write(melt)
        Q_mixed_file.write(Q_mixed)
        full_pressure_file.write(full_pressure)
        Qs_file.write(Q_s)
        Q_ice_file.write(Q_ice)
        t_str = str(t/(24.*3600.))
        df['Qice_t_'+t_str] = Q_ice.at(shelf_boundary_points)
        df['Qmixed_t_'+t_str] = Q_mixed.at(shelf_boundary_points)
        df['Qlat_t_'+t_str] = Q_latent.at(shelf_boundary_points)
        df['Qsalt_t_'+t_str] = Q_s.at(shelf_boundary_points)
        df['Melt_t'+t_str] = melt.at(shelf_boundary_points)
        df['Tb_t_'+t_str]= Tb.at(shelf_boundary_points)
        df['P_t_'+t_str] = full_pressure.at(shelf_boundary_points)
        df['Sal_t_' + t_str] = S_.at(shelf_boundary_points)
        df['Temp_t_' + t_str] = T_.at(shelf_boundary_points)


        if mesh.comm.rank ==0:
            df.to_csv(folder+"top_boundary_data.csv")


        PETSc.Sys.Print("t=", t)

