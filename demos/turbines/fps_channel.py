# Demo for scalar advection-diffusion

from thwaites import *
from thwaites.rans import RANSModel
from math import pi

import numpy as np
import os.path

outputdir = 'turbine_rotate_test'
mesh = Mesh('channel_flow.msh')

# We set up a function space of discontinous bilinear elements for :math:`q`, and
# a vector-valued continuous function space for our velocity field. ::

V = VectorFunctionSpace(mesh, "DG", 1)  # velocity space
W = FunctionSpace(mesh, "CG", 2)  # pressure space
Z = MixedFunctionSpace([V,W])

z = Function(Z)
z_old = Function(Z)
u_, p_ = z.split()
u_old, p_old = z_old.split()
source = Function(V)

# We set up the initial velocity field using a simple analytic expression. ::

V_in = 20.0

x, y, zc = SpatialCoordinate(mesh)
u0 = V_in#*ln((zc+5)/0.1)
u_init = u0*Constant((1, 0.0, 0.0))
u_.project(u_init)
u_old.project(u_init)

# the viscosity
Re = 5000
H1 = 100
mu_visc = Constant(V_in*H1/Re)

tke_in = Constant(1.0e-3*0.02 * 0.5 * V_in**2)  # 2% turbulence - Bosch '98
eps_in = Constant(0.09 * tke_in**2 / mu_visc / 100)  # r_t=vu_t/vu=100 - Bosch '98
tke0 = Constant(tke_in)
psi0 = Constant(eps_in)

# time period and time step
T = 10.
dt = .1

mom_eq = MomentumEquation(Z.sub(0), Z.sub(0))
cty_eq = ContinuityEquation(Z.sub(1), Z.sub(1))

Xin = [(0,0,100),
     (800,300,100),
     (800,-500,100)]

N = [(-1,-0.4, 0),
     (-1,0, 0),
     (-1,0, 0)]

turbines = ActuatorDiscFactory(mesh, Xin, N, 50, 50)
turbines.update_forcing(u_)
source.assign(turbines.forcing)

up_fields = {'viscosity': mu_visc,
             'source': source}

mumps_solver_parameters = {
    'snes_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij',
}

gmres_solver_parameters = {
    'snes_monitor': None,
    'ksp_type': 'gmres',
    'pc_type': 'mg',
    'mat_type': 'aij',
}

pcg_solver_parameters = {
    'snes_monitor': None,
    'ksp_type': 'cg',
    'pc_type': 'ilu',
    'mat_type': 'aij',
}

no_normal_flow = {'un': 0.}
uv_in = Constant((V_in, 0, 0))
inflow_bc = {'u': uv_in, 'tke': tke_in, 'psi': eps_in}
outflow_bc = {}
wall_bc = {'un': 0., 'wall_law_drag': 0.}
no_slip_bc = {'u': Constant((0,0,0))}
# NOTE: in the current implementation of the momentum advection term,
# there's a subtle difference between not specifying a boundary, and
# specifying an empty {} boundary - is equivalent to specifying a 
# (0,0) 'u' value - it only adds a stabilisation on the outflow - need to check correctness
up_bcs = {1: inflow_bc, 2: outflow_bc, 3: no_normal_flow}
up_solver_parameters = mumps_solver_parameters

up_coupling = [{'pressure': 1}, {'velocity': 0}]

up_timestepper = PressureProjectionSteadyStateSolver([mom_eq, cty_eq], z, up_fields, up_coupling, dt, up_bcs, solver_parameters=pcg_solver_parameters, predictor_solver_parameters=gmres_solver_parameters)

rans_fields = {'velocity': u_, 'diffusivity': mu_visc, 'viscosity':mu_visc}
rans_solver_parameters = mumps_solver_parameters

rans = RANSModel(rans_fields, mesh, bcs=up_bcs, options={'l_max': 1000.})
rans._create_integrators(BackwardEuler, dt, up_bcs, rans_solver_parameters)
#rans._create_integrators(RelaxToSteadyState, dt, up_bcs, rans_solver_parameters)
rans.initialize(rans_tke=tke0, rans_psi=psi0)

up_fields['rans_eddy_viscosity'] = rans.eddy_viscosity


# output files for RANS
# we write initial states, so the indexing is in sync with the velocity/pressure files
rans_output_fields = (
    rans.tke, rans.psi,
    rans.fields.rans_eddy_viscosity,
    rans.production, rans.rate_of_strain,
    rans.eddy_viscosity,
    rans.fields.rans_mixing_length,
    rans.sqrt_tke, rans.gamma1,
    rans.grad_tke, rans.grad_psi,
    rans.u_tau, rans.y_plus, rans.u_plus)

def residual(dt, solution, solution_old):
    u_, p_ = solution.split()
    u_old, p_old = solution_old.split()

    r1 = dt*sqrt(assemble(dot(u_-u_old, u_-u_old)*dx))
    r2 = assemble((p_-p_old)**2*dx)

    return r1, r2

solution_old = Function(Z)
for K, th in enumerate(np.linspace(0*np.pi/10,np.pi/2*(2./9.), 5)):
    k = K

    t = 0.0
    step = 0

    u_.assign(u_init)
    rans.tke.assign(tke0)
    rans.psi.assign(psi0)
    solution_old.assign(z)
    
    N = [(-np.cos(th),np.sin(th), 0),
         (-1,0, 0),
         (-1,0, 0)]

    turbines = ActuatorDiscFactory(mesh, Xin, N, 50, 50)

    u_file = File(os.path.join(outputdir, f"velocity_{k}.pvd"))
    p_file = File(os.path.join(outputdir, f"pressure_{k}.pvd"))
    t_file = File(os.path.join(outputdir, f"turbine_{k}.pvd"))
    rans_file = File(os.path.join(outputdir, f"rans_{k}.pvd"))
    solution_old.assign(z)
    dt0 = 1.0e3
    while True:
        u_old, p_old = solution_old.split()
        turbines.update_forcing(u_old)
        t_file.write(source)
        source.assign(turbines.forcing)
        solution_old.assign(z)
        up_timestepper.advance(dt, t)
        if step>10:
            break
        rans.advance(dt0, t)

        step += 1
        dt *= 0.95

        u_file.write(u_)
        p_file.write(p_)
        rans_file.write(*rans_output_fields)
