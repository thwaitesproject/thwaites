# Demo for scalar advection-diffusion

from thwaites import *
from math import pi
mesh = Mesh('channel_with_cylinder.msh')

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
mu_visc = Constant(1e-3)


D = 0.1


# We declare the output filenames, and write out the initial conditions. ::
u_file = File("velocity_norans.pvd")
u_file.write(u_)
p_file = File("pressure_norans.pvd")
p_file.write(p_)


# time period and time step
T = 100.
dt = 0.04

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
H=0.41
Um = 0.3
Ubar = 2*Um/3
uv_in = as_vector((4*Um*y*(H-y)/H**2,0))
inflow_bc = {'u': uv_in, 'tke': Constant(0.01), 'psi': Constant(0.09*(0.01)**(3/2.))}
outflow_bc = {}
noslip_bc = {'u': Constant((0.0, 0.0))}
wall_bc = {'un': 0., 'wall_law': 0.}
# NOTE: in the current implementation of the momentum advection term,
# there's a subtle difference between not specifying a boundary, and
# specifying an empty {} boundary - is equivalent to specifying a 
# (0,0) 'u' value - it only adds a stabilisation on the outflow - need to check correctness
up_bcs = {1: inflow_bc, 2: outflow_bc, 3: noslip_bc, 4: noslip_bc, 5: noslip_bc}
up_solver_parameters = mumps_solver_parameters

up_coupling = [{'pressure': 1}, {'velocity': 0}]

up_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], z, up_fields, up_coupling, dt, up_bcs, solver_parameters=up_solver_parameters)


eps = 1e-5
xy_a = [0.15-eps, 0.2]
xy_e = [0.25+eps, 0.2]
def diagnostics():
    n = -FacetNormal(mesh)
    tau = mu_visc*sym(grad(u_))
    integrand = dot(n, tau) - p_*n
    F_D = assemble(integrand[0]*ds(5))
    F_L = assemble(integrand[1]*ds(5))
    C_D = 2*F_D/(Ubar**2*D)
    C_L = 2*F_L/(Ubar**2*D)
    delta_p = p_(xy_a) -p_(xy_e)
    return C_D, C_L, delta_p

t = 0.0
step = 0
while t < T - 0.5*dt:

    up_timestepper.advance(t)

    step += 1
    t += dt

    if step % 1 == 0:
        u_file.write(u_)
        p_file.write(p_)
        print("t=", t, diagnostics())
