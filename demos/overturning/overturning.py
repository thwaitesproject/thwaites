# Demo for scalar advection-diffusion

from thwaites import *
from math import pi
mesh = UnitSquareMesh(40, 40)

# We set up a function space of discontinous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
W = FunctionSpace(mesh, "CG", 2)  # pressure space
Z = MixedFunctionSpace([V,W])

Q = FunctionSpace(mesh, "DG", 1)  # density space

z = Function(Z)
u_, p_ = z.split()
rho = Function(Q)


# We set up the initial velocity field using a simple analytic expression. ::

x, y = SpatialCoordinate(mesh)
u_init = Constant((0.0, 0.0))
u_.assign(u_init)

# the diffusivity and viscosity
kappa = Constant(1e-3)
mu = Constant(1e-3)

bell_r0 = 0.15; bell_x0 = 0.25; bell_y0 = 0.5
cone_r0 = 0.15; cone_x0 = 0.5; cone_y0 = 0.25
cyl_r0 = 0.15; cyl_x0 = 0.5; cyl_y0 = 0.75
slot_left = 0.475; slot_right = 0.525; slot_top = 0.85

bell = 0.25*(1+cos(pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
             conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
               0.0, 1.0), 0.0)

# the tracer function and its initial condition
rho_init = Constant(0.0)+slot_cyl
rho.interpolate(rho_init)

# We declare the output filenames, and write out the initial conditions. ::
u_file = File("velocity.pvd")
u_file.write(u_)
p_file = File("pressure.pvd")
p_file.write(p_)
d_file = File("density.pvd")
d_file.write(rho)

# time period and time step
T = 10.
dt = T/1000.

u_test, p_test = TestFunctions(Z)
mom_eq = MomentumEquation(u_test, Z.sub(0))
cty_eq = ContinuityEquation(p_test, Z.sub(1))

rho_test = TestFunction(Q)
rho_eq = ScalarAdvectionDiffusionEquation(rho_test, Q)

u, p = split(z)

source = as_vector((0, -1.0))*rho  # momentum source: the buoyancy term
up_fields = {'viscosity': mu, 'source': source}
rho_fields = {'diffusivity': kappa, 'velocity': u}

mumps_solver_parameters = {
    'snes_monitor': True,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij'
}
# weakly applied dirichlet bcs on top and bottom for density
rho_top = 1.0
rho_bottom = 0.0
rho_bcs = {3: {'q': rho_bottom}, 4: {'q': rho_top}}
rho_solver_parameters = mumps_solver_parameters

no_normal_flow = {'un': 0.}
up_bcs = {1: no_normal_flow, 2: no_normal_flow, 3: no_normal_flow}
up_solver_parameters = mumps_solver_parameters

up_coupling = [{'pressure': 1}, {'velocity': 0}]

up_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], z, up_fields, up_coupling, dt, up_bcs, solver_parameters=up_solver_parameters)
rho_timestepper = DIRK33(rho_eq, rho, rho_fields, dt, rho_bcs, solver_parameters=rho_solver_parameters)

t = 0.0
step = 0
while t < T - 0.5*dt:

    up_timestepper.advance(t)
    rho_timestepper.advance(t)

    step += 1
    t += dt

    if step % 1 == 0:
        u_file.write(u_)
        p_file.write(p_)
        d_file.write(rho)
        print("t=", t)
