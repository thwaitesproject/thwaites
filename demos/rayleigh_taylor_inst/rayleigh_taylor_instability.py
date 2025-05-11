# Rayleigh-Taylor instability. see fluidity document for incomplete description.
# notebook has set up with derivation of non dimensional form of momentum equation.
#
from thwaites import *
from math import pi

d=1
mesh = RectangleMesh(50, 400,0.5,4)

import mpi4py
from mpi4py import MPI
print("You have Comm WORLD size = ",mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)


# setup functionspaces and functions for velocity, pressure and density
V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
W = FunctionSpace(mesh, "CG", 2)  # pressure space
Z = MixedFunctionSpace([V,W])

Q = FunctionSpace(mesh, "DG", 1)  # density space


z = Function(Z)
u_, p_ = z.subfunctions
rho = Function(Q)


x,y = SpatialCoordinate(mesh)

# convert y: 0->4 to -2->2
y = y-2

u_init = Constant((0.0, 0.0))
u_.assign(u_init)

# the diffusivity and viscosity
kappa = Constant(0)  # no diffusion of density
Reynolds_number = Constant(1000.0)
mu = Constant(1/Reynolds_number)

# height of initial interface:
def eta(x,d):
    return -0.1*d*cos((2*pi*x)/d)


# the tracer function and its initial condition
# change from rho to temp...
# these numbers come from At = (rho_max - rho_min)/(rho_max+rho_min) = 0.75
# and dimensionalising N.S (see notebook - monday 28th october)
# rho_min is used to non dimensionalise rho. so rho_min = 1 and rho_max = 7

At = 0.5 # must be less than 1.0

rho_min=1.0 # dimensionless rho_min
rho_max = rho_min*(1.0+At)/(1.0-At)


rho_init = 0.5*(rho_max+rho_min) + 0.5*(rho_max-rho_min)*tanh((y-eta(x,d))/(0.01*d))
rho.interpolate(rho_init)

# folder to store output files
folder = "output/"

# We declare the output filenames, and write out the initial conditions. ::
u_file = File(folder+"velocity.pvd")
u_file.write(u_)
p_file = File(folder+"pressure.pvd")
p_file.write(p_)
d_file = File(folder+"density.pvd")
d_file.write(rho)

# time period and time step
T = 5.
dt = 0.01

# the equations to solve:
# the equation objects provide the UFL, the symbolic expression of the weak form of the equations
# that we want to solve
u_test, p_test = TestFunctions(Z)

# Navier Stokes momentum equation
mom_eq = MomentumEquation(Z.sub(0), Z.sub(0))
# incompressible (div u=0) continuity equation
cty_eq = ContinuityEquation(Z.sub(1), Z.sub(1))


rho_test = TestFunction(Q)
# advection-diffusion equation for density
rho_eq = ScalarAdvectionDiffusionEquation(Q, Q)


# fields and source terms used in the equations:
rho_mean = 0.5*(rho_min+rho_max)
mom_source = as_vector((0, -1.0))*(rho-rho_min) # momentum source: the buoyancy term boussinesq approx

u, p = split(z)
# fields used in the velocity-pressure solve
up_fields = {'viscosity': mu, 'source': mom_source}
# fields used in the density solve:
rho_fields = {'diffusivity': kappa, 'velocity': u}

# options to solve the discretised system, here configured to use a direct solver (using the MUMPS library)
mumps_solver_parameters = {
    'snes_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij'
}

# weakly applied dirichlet bcs on top and bottom for density

# 1: top, 2: rhs, 3: bottom, 4: lhs

rho_bcs = {}
rho_solver_parameters = mumps_solver_parameters


no_normal_flow = {'un': 0.}
no_normal_no_slip_flow = {'u': as_vector((0,0))}

up_bcs = {1: no_normal_no_slip_flow, 2: no_normal_flow, 3: no_normal_no_slip_flow, 4: no_normal_flow}# no_normal_no_slip_flow}

# these are the solver parameters for the velocity-pressure system
up_solver_parameters = {
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
        'schur_ksp_max_it': 100,
        'schur_ksp_converged_reason': None,
        'schur_pc_type': 'gamg',
    },
}


# since all our boundaries are closed, we have Neumann boundary conditions
# for pressure on all sides, which means that we can add an arbitray constant to
# pressure, i.o.w. the equations have a nullspace in pressure - which we need to provide to the solver
pressure_nullspace = VectorSpaceBasis(constant=True)

# here we tie it all together, combining the equation with a time integrator

# the momentum and continuity equations are solved simultaneously for velocity and pressure
# in a so called pressure projection approach: first it performs a preliminary solve of the momentum
# equation. The preliminary velocity from that is not yet divergence free. Then in the pressure projection
# solve, we compute a pressure such that, if we were to substitute it back into the momentum equation
# the velocity becomes divergence free

# the velocity and pressure are stored together in a function z (which can be split into a velocity u and pressure p)
# the up_coupling basically tells that the "pressure" field used in the momentum equation should be the second part of 
# z, and "velocity" used in the continuity equation is the first part of z
up_coupling = [{'pressure': 1}, {'velocity': 0}]
up_timestepper = PressureProjectionTimeIntegrator([mom_eq, cty_eq], z, up_fields, up_coupling, dt, up_bcs,
                                                  solver_parameters=up_solver_parameters,
                                                  predictor_solver_parameters=mumps_solver_parameters,
                                                  picard_iterations=1, pressure_nullspace=pressure_nullspace)

# DIRK33 is a 3 stage Runge Kutta time stepping method
rho_timestepper = DIRK33(rho_eq, rho, rho_fields, dt, rho_bcs, solver_parameters=rho_solver_parameters)

# we use a limiter to ensure we don't get under and overshoots in the solution:
rho_limiter = VertexBasedLimiter(Q)

# these are used as temporary fields to store the u and v component of velocity when the limiter is applied:
u_comp = Function(Q)
v_comp = Function(Q)

t = 0.0
step = 0


output_dt = 0.5
output_step = output_dt/dt

while t < T - 0.5*dt:

    up_timestepper.advance(t)
    rho_timestepper.advance(t)

    rho_limiter.apply(rho)
    u_comp.interpolate(u[0])
    rho_limiter.apply(u_comp)
    v_comp.interpolate(u[1])
    rho_limiter.apply(v_comp)
    u_.interpolate(as_vector((u_comp, v_comp)))

    step += 1
    t += dt

    if step % output_step == 0:
        u_file.write(u_)
        p_file.write(p_)
        d_file.write(rho)
        print("t=", t)
