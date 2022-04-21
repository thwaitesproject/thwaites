# MMS test for scalar advection diffusion of a tracer
# based on a fluidity one
from thwaites import *
from thwaites.equations import BaseEquation
from thwaites.meltrate_param import TwoEqMeltRateParam
from thwaites.utility import CombinedSurfaceMeasure
import numpy as np
from math import ceil
from firedrake import dx
polynomial_order = 1
number_of_grids =4
output_freq = 100
H1 = 100
H2 = 100
horizontal_stretching = 1
L = 100 * horizontal_stretching
depth = 1000
cells = [10, 20, 40, 80]  # i.e 10x10, 20x20, 40x40, 80x80
meshes = ["verification_unstructured_100m_square_res10m.msh",
        "verification_unstructured_100m_square_res5m.msh", 
        "verification_unstructured_100m_square_res2.5m.msh",
        "verification_unstructured_100m_square_res1.25m.msh"]

def error(mesh_name, nx):
    dx_grid = L/nx
    dz = H2/nx
#    mesh = Mesh(mesh_name)
    mesh = SquareMesh(nx, nx, L)
    mesh.coordinates.dat.data[:,1] -= depth
    mesh.coordinates.dat.data[:,0] *= horizontal_stretching
    print(mesh.coordinates.dat.data[:,0].min())
    print(mesh.coordinates.dat.data[:,0].max())
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
    p_ana = cos(arg) * cos(np.pi * x / L) 
    vel_ana = as_vector((u_ana,v_ana))
    vel_ana_f = Function(U, name='vel analytical').project(vel_ana)
    vel_.interpolate(vel_init)

    p_ana_f = Function(W, name='p analytical').project(p_ana)
     
    pavg = assemble(p_*dx) / (L * H2)
    
    mu_h = Constant(1*horizontal_stretching)
    mu_v = Constant(1)
    mu = as_tensor([[mu_h, 0], [0, mu_v]])
    ramp = Constant(0.0)
    u_source =  -3.14159265358979*sin(3.14159265358979*x/L)*cos(3.14159265358979*(-H2 + depth + y)/H2)/L + 1.0*x*sin(3.14159265358979*(-H2 + depth + y)/H2)**2/L**2 + x*cos(3.14159265358979*(-H2 + depth + y)/H2)**2/L**2 + 9.86960440108936*mu_v*x*cos(3.14159265358979*(-H2 + depth + y)/H2)/(H2**2*L)
    v_source = 0.318309886183791*H2*sin(3.14159265358979*(-H2 + depth + y)/H2)*cos(3.14159265358979*(-H2 + depth + y)/H2)/L**2 - 3.14159265358979*sin(3.14159265358979*(-H2 + depth + y)/H2)*cos(3.14159265358979*x/L)/H2 - 3.14159265358979*mu_v*sin(3.14159265358979*(-H2 + depth + y)/H2)/(H2*L)
    vel_source = as_vector((u_source, v_source))
    
    # We declare the output filename, and write out the initial condition. ::
    vel_outfile = File("vel_gz_mms_L"+str(L)+"m_nx"+str(nx)+"_dt4overnx_100stepsdtLovernx_scaleMuh1_then0.1_cosp_initzero_pavg_whileor_2pic_pudiff1e-6_backeul_square.pvd")
    vel_outfile.write(vel_, vel_ana_f)
    p_outfile = File("p_gz_mms_L"+str(L)+"m_nx"+str(nx)+"_constdt4overnx_100stepsdtLovernx_scaleMuh1_then0.1_cosp_initzero_pavg_whileor_2pic_pudiff1e-6_back_eul_square.pvd")
    p_outfile.write(p_, p_ana_f)

    # a big timestep, which means BackwardEuler takes us to a steady state almost immediately
    # (needs to be smaller at polynomial_degree>0, 0.1/nx works for p=1 for 4 meshes)
    dt = Constant(4/nx)

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
    stress_tensor = mu_v*(nabla_grad(vel_ana)+ grad(vel_ana))
    no_normal_flow = 0.0
    vp_bcs = {4: {'un': no_normal_flow}, 2: {'u': vel_ana},
          3: {'un': no_normal_flow }, 1: {'un': no_normal_flow}}
    
    mumps_solver_parameters = {
        'snes_monitor': None,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'mat_mumps_icntl_14': 200,
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij',
        'snes_max_it': 100,
        'snes_rtol': 1e-8,
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
    vp_timestepper = PressureProjectionTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
                                                          solver_parameters=mumps_solver_parameters,
                                                         predictor_solver_parameters=mumps_solver_parameters,
                                                         picard_iterations=2, theta=1,
                                                         pressure_nullspace=VectorSpaceBasis(constant=True))
    vp_timestepper.initialize_pressure()
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
    while (u_change>1e-6) or (v_change>1e-6) or (p_change>1e-6): 

        u_prev.interpolate(vel_old[0])
        v_prev.interpolate(vel_old[1])
        p_prev.assign(p_old)
      
        
        vp_timestepper.advance(t)
        step += 1
        t += float(dt)
        
        u_change = norm(vel_[0]-u_prev)
        v_change = norm(vel_[1]-v_prev)
        p_change = norm(p_- p_prev)
        if step == 100:
            vp_timestepper.dt_const.assign(L/nx)
            dt = L /nx

        if step == 100 *int(nx/10):             
            mu_h.assign(0.1*horizontal_stretching)
            mu_v.assign(0.1)

        if step % output_freq == 0:
            vel_outfile.write(vel_, vel_ana_f)
            p_outfile.write(p_, p_ana_f)
            print("t, u/v/p change: ", t, u_change, v_change, p_change)

    pavg = assemble(p_*dx)/ (L*H2) #assemble(Constant(1.0, domain=mesh)*dx)
    p_.interpolate(p_ - pavg)
    
    u_err = norm(vel_[0]-u_ana)
    v_err = norm(vel_[1]-v_ana)
    p_err = norm(p_ - p_ana)
    return u_err, v_err, p_err

errors = np.array([error(m, c) for m, c in zip(meshes, cells)])


conv = np.log(errors[:-1]/errors[1:])/np.log(2)

print('U velocity errors:', errors[:,0])
print('V velocity errors:', errors[:,1])
print('Pressure errors:', errors[:,2])
print('U velocity convergence order:', conv[:,0])
print('V velocity convergence order:', conv[:,1])
print('Pressure convergence order:', conv[:,2])
assert all(conv[:,0]> polynomial_order+0.95)
assert all(conv[:,1]> polynomial_order+0.95)
assert all(conv[:,2]> polynomial_order+0.95)
