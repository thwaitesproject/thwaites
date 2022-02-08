# MMS test for scalar advection diffusion of a tracer
# based on a fluidity one
from thwaites import *
from thwaites.equations import BaseEquation
from thwaites.meltrate_param import TwoEqMeltRateParam
from thwaites.utility import CombinedSurfaceMeasure
import numpy as np
from math import ceil

polynomial_order = 1
number_of_grids =4 
output_freq = 100
H1 = 100
H2 = 100
L = 100
depth = 1000
cells = [10, 20,40,80]  # i.e 10x10, 20x20, 40x40, 80x80
meshes = ["verification_unstructured_100m_square_res10m.msh",
        "verification_unstructured_100m_square_res5m.msh", 
        "verification_unstructured_100m_square_res2.5m.msh", 
        "verification_unstructured_100m_square_res1.25m.msh"]

def error(mesh_name, nx):
    dx = L/nx
    dz = H2/nx
    
    #mesh = SquareMesh(nx, nx, L)
    mesh = Mesh(mesh_name)
    mesh.coordinates.dat.data[:,1] -= depth
    
    V = FunctionSpace(mesh, "DG", polynomial_order)
    U = VectorFunctionSpace(mesh, "DG", polynomial_order)
    W = FunctionSpace(mesh, "CG", polynomial_order+1)
    M = MixedFunctionSpace([U, W])
    
    x, y = SpatialCoordinate(mesh)
    m = Function(M)
    vel_, p_ = m.split()  # function: y component of velocity, pressure
    vel, p = split(m)  # expression: y component of velocity, pressure
    vel_._name = "velocity"
    p_._name = "perturbation pressure"

    vel_init = zero(mesh.geometric_dimension())

    arg = - np.pi / H2 * (y + depth - H2)
    u_ana =  x / L * cos(arg)
    v_ana = (H2 / np.pi) * sin(arg) / L
    pint = x / L * sin(arg) * sin(-2 * np.pi / H2 * (y + depth - H2))
    p_ana = pint -  sin(-2 * np.pi / H2 * (y + depth - H2)) * sin(-2 * np.pi / H2 * (y + depth - H2))

    vel_ana = as_vector((u_ana,v_ana))
    vel_ana_f = Function(U, name='vel analytical').project(vel_ana)
   
    vel_.interpolate(vel_ana +0.1*vel_ana)
    File('vel.pvd').write(vel_)

    p_ana_f = Function(W, name='p analytical').project(p_ana)
    
    #p_.assign(p_ana_f)
    mu = Constant(1)
    u_source = 1.0*x*sin(3.14159265358979*(-H2 + depth + y)/H2)**2/L**2 + x*cos(3.14159265358979*(-H2 + depth + y)/H2)**2/L**2 + 9.86960440108936*mu*x*cos(3.14159265358979*(-H2 + depth + y)/H2)/(H2**2*L)
    v_source = 0.318309886183791*H2*sin(3.14159265358979*(-H2 + depth + y)/H2)*cos(3.14159265358979*(-H2 + depth + y)/H2)/L**2 - 3.14159265358979*mu*sin(3.14159265358979*(-H2 + depth + y)/H2)/(H2*L)
    vel_source = as_vector((u_source, v_source))
    
    # We declare the output filename, and write out the initial condition. ::
    vel_outfile = File("vel_gz_mms.pvd")
    vel_outfile.write(vel_, vel_ana_f)
    p_outfile = File("p_gz_mms.pvd")
    p_outfile.write(p_, p_ana_f)

    # a big timestep, which means BackwardEuler takes us to a steady state almost immediately
    # (needs to be smaller at polynomial_degree>0, 0.1/nx works for p=1 for 4 meshes)
    dt = Constant(1/nx)

    # Set up equations
    mom_eq = MomentumEquation(M.sub(0), M.sub(0))
    cty_eq = ContinuityEquation(M.sub(1), M.sub(1))

    vp_coupling = [{'pressure': 1}, {'velocity': 0}]
    vp_fields = {'viscosity': mu, 'source': vel_source}

    # boundary conditions, bottom and left are inflow
    # so Dirichlet, with others specifying a flux
    left_id, right_id, bottom_id, top_id = 1, 2, 3, 4
    one = Constant(1.0)
    n = FacetNormal(mesh)
    nonormal_noslip = as_vector((0,0))

    no_normal_flow = 0.0
    vp_bcs = {4: {'un': no_normal_flow}, 2: {'u': vel_ana},
          3: {'un': no_normal_flow }, 1: {'un': no_normal_flow}}
    
    mumps_solver_parameters = {
        'snes_monitor': None,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij',
        'snes_max_it': 100,
        'snes_rtol': 1e-8,
    }

#    vp_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
 #                                                       solver_parameters=mumps_solver_parameters)
    vp_timestepper = PressureProjectionTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
                                                          solver_parameters=mumps_solver_parameters,
                                                          predictor_solver_parameters=mumps_solver_parameters,
                                                          picard_iterations=1,
                                                          pressure_nullspace=VectorSpaceBasis(constant=True))
    vp_timestepper.initialize_pressure()
    vp_timestepper.dt_const.assign(1)
    vel_old, p_old = vp_timestepper.solution_old.split()
    u_prev = Function(V, name='u_old')
    v_prev = Function(V, name='v_old')
    p_prev = Function(W, name='p_old')

    u_diff_abs = Function(V, name='u_diff_abs')
    v_diff_abs = Function(V, name='v_diff_abs')
    p_diff_abs = Function(W, name='p_diff_abs')

    t = 0.0
    step = 0
    u_change = 1.0
    v_change = 1.0
    p_change = 1.0
    while (u_change>1e-6) and  (v_change>1e-6) and (p_change>1e-6): 

        u_prev.interpolate(vel_old[0])
        v_prev.interpolate(vel_old[1])
        p_prev.assign(p_old)
      
        
        vp_timestepper.advance(t)
        step += 1
        t += float(dt)
        # Inifity norm
#        u_diff_abs.interpolate(abs(vel_[0]-u_prev))
#        v_diff_abs.interpolate(abs(vel_[1]-v_prev))
#        p_diff_abs.interpolate(abs(p_-p_prev))
#        with u_diff_abs.dat.vec_ro as udiffabs:
#                u_change = udiffabs.max()[1]
#        with v_diff_abs.dat.vec_ro as vdiffabs:
#                v_change = vdiffabs.max()[1]
#        with p_diff_abs.dat.vec_ro as pdiffabs:
#                p_change = pdiffabs.max()[1]
        u_change = norm(vel_[0]-u_prev)
        v_change = norm(vel_[1]-v_prev)
        p_change = norm(p_- p_prev)
        if step % output_freq == 0:
            vel_outfile.write(vel_, vel_ana_f)
            p_outfile.write(p_, p_ana_f)
            print("t, u/v/p change: ", t, u_change, v_change, p_change)

    

    u_err = norm(vel_[0]-u_ana)
    v_err = norm(vel[1]-v_ana)
    p_err = norm(p-p_ana)
    return u_err, v_err, p_err

errors = np.array([error(meshes[i], cells[i]) for i in range(number_of_grids)]) #10*2**np.arange(number_of_grids)])


conv = np.log(errors[:-1]/errors[1:])/np.log(2)

print('U velocity errors:', errors[:,0])
print('V velocity errors:', errors[:,1])
print('Pressure errors:', errors[:,2])
print('U velocity convergence order:', conv[:,0])
print('V velocity convergence order:', conv[:,1])
print('Pressure convergence order:', conv[:,2])
assert all(conv[:,0]> polynomial_order+0.95)
assert all(conv[:,1]> polynomial_order+0.95)

