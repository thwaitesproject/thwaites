# Rayleigh-Taylor instability. see fluidity document for incomplete description.
# notebook has set up with derivation of non dimensional form of momentum equation.
# Made on 12th march.
# Originally a copy of "rayleigh_taylor_closed_iterative_29.11.
#
from thwaites import *
from math import pi
import sys
d=1
nx = 100 # number of points x direction


# create a rectangular mesh using inbuilt firedrake.
mesh = RectangleMesh(nx, nx*4, d, 4*d) # (nx, ny, lx, ly)

# check that mpi is working as expected.
import mpi4py
from mpi4py import MPI
print("You have Comm WORLD size = ",mesh.comm.size)
print("You have Comm WORLD rank = ", mesh.comm.rank)

sys.stdout.flush()

# Finite element space DG1-P2
V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
W = FunctionSpace(mesh, "CG", 2)  # pressure space
Z = MixedFunctionSpace([V,W])

Q = FunctionSpace(mesh, "DG", 1)  # density space

z = Function(Z)
u_, p_ = z.split()
rho = Function(Q)



x,y = SpatialCoordinate(mesh)

# convert x: 0 -> 1 to -0.5 -> 0.5
x = x - 0.5
# convert y: 0 ->4  to -2 -> 2
y = y - 2

# initialise zero veclocity
u_init = Constant((0.0, 0.0))
u_.assign(u_init)

# set the diffusivity to zero
kappa = Constant(0)  # no diffusion of density /temp  # grid peclet = 1

# Reynolds number from Guermond test case (2009)
Reynolds_number = Constant(500.0)
# non dimensional form of equations equivalent
mu = Constant(1/Reynolds_number)



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



folder = "/home/wscott/rayleigh_taylor_tests/"
# We declare the output filenames, and write out the initial conditions. ::
u_file = File(folder+"velocity.pvd")
u_file.write(u_)
p_file = File(folder+"pressure.pvd")
p_file.write(p_)
d_file = File(folder+"density.pvd")
d_file.write(rho)



# time period and time step
T = 2.5
dt = 0.01

u_test, p_test = TestFunctions(Z)


mom_eq = MomentumEquation(Z.sub(0), Z.sub(0))

cty_eq = ContinuityEquation(Z.sub(1), Z.sub(1))


rho_test = TestFunction(Q)
rho_eq = ScalarAdvectionDiffusionEquation(Q, Q)

u, p = split(z)

rho_mean = 0.5*(rho_min+rho_max)
mom_source = as_vector((0, -1.0))*(rho) # momentum source: the buoyancy term boussinesq approx

up_fields = {'viscosity': mu, 'source': mom_source}
rho_fields = {'diffusivity': kappa, 'velocity': u}


mumps_solver_parameters = {
    'snes_monitor': True,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij'
}
# weakly applied dirichlet bcs on top and bottom for density

# 1: top, 2: rhs, 3: bottom, 4: lhs

#1: plane x == 0
#2: plane x == Lx
#3: plane y == 0
#4: plane y == Ly

#rho_bcs = {1: {'q': rho_max}}
rho_bcs = {}
rho_solver_parameters = mumps_solver_parameters


no_normal_flow = {'un': 0.}
no_normal_no_slip_flow = {'u': as_vector((0,0))}

up_bcs = {1: no_normal_flow, 2: no_normal_flow, 3: no_normal_no_slip_flow, 4: no_normal_no_slip_flow}# no_normal_no_slip_flow}
#up_solver_parameters = mumps_solver_parameters


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
        'ksp_converged_reason': True,
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
        'schur_ksp_converged_reason': True,

        'schur_pc_type': 'gamg',
    },
}
pressure_nullspace = VectorSpaceBasis(constant=True)



up_coupling = [{'pressure': 1}, {'velocity': 0}]

up_timestepper = PressureProjectionTimeIntegrator([mom_eq, cty_eq], z, up_fields, up_coupling, dt, up_bcs,
                                                  solver_parameters=up_solver_parameters,
                                                  predictor_solver_parameters=mumps_solver_parameters,
                                                  picard_iterations=1, pressure_nullspace=pressure_nullspace)
rho_timestepper = DIRK33(rho_eq, rho, rho_fields, dt, rho_bcs, solver_parameters=rho_solver_parameters)



rho_limiter = VertexBasedLimiter(Q)
u_comp = Function(Q)
v_comp = Function(Q)

t = 0.0
step = 0


output_dt = 0.25
output_step = output_dt/dt

while t < T - 0.5*dt:
    if mesh.comm.rank == 0:
        print("t = ", t)
    sys.stdout.flush()

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
