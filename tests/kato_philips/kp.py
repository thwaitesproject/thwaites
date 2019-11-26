from thwaites import *
from math import pi
import numpy as np
from firedrake.petsc import PETSc
import os.path

output_dir = 'kato'

quad = True
Lx = 45.
H = 50.
nx = 3
nz = 40
mesh = PeriodicRectangleMesh(nx, nz, Lx, H, direction='x', quadrilateral=quad)
#h_mesh = PeriodicIntervalMesh(nx, Lx)
#v_mesh = IntervalMesh(nz, H)
#mesh = ExtrudedMesh(h_mesh, nz, layer_height=H)


if quad:
    V = VectorFunctionSpace(mesh, "DQ", 1)  # velocity space
    W = FunctionSpace(mesh, "DQ", 1)  # pressure space
    Q = FunctionSpace(mesh, "DQ", 1)  # density space
else:
    V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
    W = FunctionSpace(mesh, "CG", 2)  # pressure space
    Q = FunctionSpace(mesh, "DG", 1)  # density space
M = MixedFunctionSpace([V,W])

m = Function(M)
u_, p_ = m.split()

sal = Function(Q)
rho = Function(Q)



u_init = Constant((0.0, 0.0))
u_.assign(u_init)

# linearly vary viscosity/diffusivities over domain.
kappa_sal = Constant([[1., 0], [0, 1.4e-7]])
mu_visc = Constant([[1., 0],[0, 1.3e-6]])
rho0=1025
tau_wind = 1e-4
tke0 = Constant(1e-7)
psi0 = Constant(1.4639e-8)

Tref= 0.0
beta_temp = 3.87*10E-5 #5.0E-3
beta_sal = 0.7865/1027
g = 9.81
#rho_expr = -beta_temp*(temp-Tref)

## sal. initial conditions
buoyfreq0 = 0.01
rho_grad = -buoyfreq0**2 / g
salt_grad = rho_grad/beta_sal
x, z = SpatialCoordinate(mesh)
sal_init_expr = salt_grad*z
sal.interpolate(sal_init_expr)
rho_expr = beta_sal*sal



# We declare the output filenames, and write out the initial conditions. ::
u_file = File(os.path.join(output_dir, "velocity.pvd"))
u_file.write(u_)
p_file = File(os.path.join(output_dir, "pressure.pvd"))
p_file.write(p_)

t_file = File(os.path.join(output_dir, "salinity.pvd"))
t_file.write(sal)

# time period and time step
T = 4*3600.
dt = 60.

mom_eq = MomentumEquation(M.sub(0), M.sub(0))
cty_eq = ContinuityEquation(M.sub(1), M.sub(1))
sal_eq = ScalarAdvectionDiffusionEquation(Q, Q)



u, p = split(m)


#From Ben's thesis page 31

# momentum source: the buoyancy term boussinesq approx
mom_source = as_vector((0, -g*rho_expr))

up_fields = {'viscosity': mu_visc, 'source': mom_source}
sal_fields = {'diffusivity': kappa_sal, 'velocity': Constant([0.0, 0.0])}
rans_fields = {'velocity': u_, 'viscosity': mu_visc, 'diffusivity': mu_visc, 'density': rho_expr}

mumps_solver_parameters = {
    'snes_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij',
    'snes_max_it': 100,
    'snes_atol': 1e-16,
}
up_solver_parameters = mumps_solver_parameters
rans_solver_parameters = mumps_solver_parameters
sal_solver_parameters = mumps_solver_parameters

sal_bcs = {}


#up_bcs = {2: {'stress': as_vector((tau_wind, 0)), 'un': 0}, 1: {'un': 0}}
up_bcs = {2: {'stress': as_vector((tau_wind, 0))}, 1: {'un': 0}}

rans = RANSModel(rans_fields, mesh, bcs=up_bcs, options={'l_max': H})
rans._create_integrators(BackwardEuler, dt, up_bcs, rans_solver_parameters)
rans.initialize(rans_tke=tke0, rans_psi=psi0)

up_fields['rans_eddy_viscosity'] = rans.eddy_viscosity
sal_fields['rans_eddy_diffusivity'] = rans.eddy_diffusivity

rans_file = File(os.path.join(output_dir, "rans.pvd"))
rans_output_fields = (
    rans.tke, rans.psi,
    rans.fields.rans_eddy_viscosity,
    rans.production, rans.rate_of_strain, rans.rate_of_strain_p1dg, rans.rate_of_strain_vert,
    rans.eddy_viscosity,
    rans.fields.rans_mixing_length,
    rans.sqrt_tke, rans.gamma1,
    rans.grad_tke, rans.grad_psi,
    rans.C_mu, rans.C_mu_p,
    rans.fields.N2, rans.fields.N2_neg, rans.fields.N2_pos_over_k, rans.fields.N2_pos,
    rans.fields.M2,
    rans.u_tau, rans.y_plus, rans.u_plus)
rans_file.write(*rans_output_fields)


up_coupling = [{'pressure': 1}, {'velocity': 0}]

up_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], m, up_fields, up_coupling, dt, up_bcs,
                                                        solver_parameters=up_solver_parameters)
sal_timestepper = DIRK33(sal_eq, sal, sal_fields, dt, sal_bcs, solver_parameters=sal_solver_parameters)


# aux. vars to compute mixing depth
tke_p1 = Function(W)
mix_depth = Function(W)
mix_depths = []

t = 0.0
step = 0
trange = []

output_step = 5

limiter = VertexBasedLimiter(Q)
u_lim = Function(Q)

while t < T - 0.5*dt:


    uint0 = assemble(u_[0]*dx)
    up_timestepper.advance(t)
    uint1 = assemble(u_[0]*dx)
    print(uint1-uint0, tau_wind*Lx)

    sal_timestepper.advance(t)

    rans.advance(t)

    tke_p1.project(rans.tke)
    mix_depth.interpolate(conditional(tke_p1>rans.tke_min*1.5, z, H))
    mix_depths.append(H-mix_depth.dat.data.min())
    trange.append(t)

    step += 1
    t += dt

    if step % output_step == 0:
        print(t, uint1/Lx)
        u_file.write(u_)
        p_file.write(p_)
        t_file.write(sal)
        rans_file.write(*rans_output_fields)


        PETSc.Sys.Print("t=", t)

try:
    import matplotlib.pyplot as plt
    plt.plot(trange, mix_depths, label='Thwaites')
    plt.plot(trange, [1.05*sqrt(tau_wind)*buoyfreq0**(-0.5)*sqrt(t) for t in trange], label='theory')
    plt.legend()
    plt.savefig('mixing_depth.pdf')
except ImportError:
    pass
