# Demo for scalar advection-diffusion

from thwaites import *
from thwaites.rans import RANSModel
from math import pi
mesh = Mesh('bao.msh')

# We set up a function space of discontinous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
W = FunctionSpace(mesh, "CG", 2)  # pressure space
Z = MixedFunctionSpace([V,W])

z = Function(Z)
u_, p_ = z.split()

# We set up the initial velocity field using a simple analytic expression. ::

x, y = SpatialCoordinate(mesh)
u_init = Constant((0.0, 0.0))
u_.assign(u_init)

# the viscosity
Re = 2.2e4
V_in = 1.0
H1 = 1.0
mu_visc = V_in*H1/Re

mu0 = 0.1
tke0 = Constant((mu0/H1)**2)
psi0 = Constant(0.09*tke0**(3/2)/H1)


# We declare the output filenames, and write out the initial conditions. ::
u_file = File("velocity.pvd")
u_file.write(u_)
p_file = File("pressure.pvd")
p_file.write(p_)


# time period and time step
T = 1000.
dt = T/1000.

mom_eq = MomentumEquation(Z.sub(0), Z.sub(0))
cty_eq = ContinuityEquation(Z.sub(1), Z.sub(1))

u, p = split(z)

up_fields = {'viscosity': mu_visc}

mumps_solver_parameters = {
    'snes_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij',
}

no_normal_flow = {'un': 0.}
uv_in = Constant((V_in, 0))
inflow_bc = {'u': uv_in, 'tke': Constant(0.0), 'psi': Constant(0.09*(0.0)**(3/2.))}
outflow_bc = {}
wall_bc = {'un': 0., 'wall_law': 0.}
# NOTE: in the current implementation of the momentum advection term,
# there's a subtle difference between not specifying a boundary, and
# specifying an empty {} boundary - is equivalent to specifying a 
# (0,0) 'u' value - it only adds a stabilisation on the outflow - need to check correctness
up_bcs = {1: inflow_bc, 2: outflow_bc, 3: no_normal_flow, 4: no_normal_flow, 5: wall_bc}
up_solver_parameters = mumps_solver_parameters

up_coupling = [{'pressure': 1}, {'velocity': 0}]

up_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], z, up_fields, up_coupling, dt, up_bcs, solver_parameters=up_solver_parameters)

rans_fields = {'velocity': u_,}
rans_solver_parameters = mumps_solver_parameters

rans = RANSModel(rans_fields, mesh, bcs=up_bcs, options={'l_max': 10.})
rans._create_integrators(BackwardEuler, dt, up_bcs, rans_solver_parameters)
rans.initialize(rans_tke=tke0, rans_psi=psi0)

up_fields['rans_eddy_viscosity'] = rans.fields.rans_eddy_viscosity


# output files for RANS
# we write initial states, so the indexing is in sync with the velocity/pressure files
rans_file = File("rans.pvd")
rans_output_fields = (
    rans.fields.rans_tke, rans.fields.rans_psi,
    rans.fields.rans_eddy_viscosity,
    rans.production, rans.rate_of_strain,
    rans.eddy_viscosity,
    rans.fields.rans_mixing_length,
    rans.sqrt_tke,
    rans.u_tau, rans.y_plus, rans.u_plus)
rans_file.write(*rans_output_fields)

def diagnostics():
    n = -FacetNormal(mesh)
    tau = (mu_visc+rans.fields.rans_eddy_viscosity)*sym(grad(u_))
    integrand = dot(n, tau) - p_*n
    F_D = assemble(integrand[0]*ds(5))
    F_p = assemble(p_*n[0]*ds(5))
    F_L = assemble(integrand[1]*ds(5))
    C_D = 2*F_D/(V_in**2*H1)
    C_p = 2*F_p/(V_in**2*H1)
    C_L = 2*F_L/(V_in**2*H1)
    return C_D, C_p, C_L

t = 0.0
step = 0
while t < T - 0.5*dt:

    up_timestepper.advance(t)
    rans.advance(t)

    step += 1
    t += dt

    if t>100:
        dt = 0.002

    if step % 1 == 0:
        u_file.write(u_)
        p_file.write(p_)
        rans_file.write(*rans_output_fields)
        print("t=", t, diagnostics())
