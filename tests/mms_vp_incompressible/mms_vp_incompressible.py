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
cells = [40,80]  # i.e 10x10, 20x20, 40x40, 80x80
meshes = ["verification_unstructured_100m_square_res10m.msh",
        "verification_unstructured_100m_square_res5m.msh", 
        "verification_unstructured_100m_square_res2.5m.msh", 
        "verification_unstructured_100m_square_res1.25m.msh"]

def error(mesh_name, nx):
    dx = L/nx
    dz = H2/nx
    #mesh1d = IntervalMesh(nx, 0.5)
    #layers = []
    #cell = 0
    #xr = 0
    #min_dz = 0.01*dz # if top cell is thinner than this, merge with cell below
    #tiny_dz = 1e-9*dz #  workaround for zero measure exterior facets (fd issue #1858)
    #for i in range(nx):
    #    xr += dx  # y of right-node (assumed to be the higher one)
    #    height = H1 + xr/L * (H2-H1)
    #    ncells = ceil((height-min_dz)/dz)
    #    layers.append([0, ncells])
    #    cell += ncells
    #mesh = ExtrudedMesh(mesh1d, layers, layer_height=dz)
    #y = mesh.coordinates.dat.data_ro[:,0]
    #z = mesh.coordinates.dat.data_ro[:,1]
    #height = np.maximum(H1 + y/L * (H2-H1), H1 + tiny_dz)
    #mesh.coordinates.dat.data[:,1] = np.minimum(height, z)
   # mesh = Mesh(mesh_name)
    mesh = SquareMesh(nx,nx,np.pi)
    #mesh.coordinates.dat.data[:,1] -= depth

    #ds = CombinedSurfaceMeasure(mesh, 5)
    #mesh.coordinates.dat.data[:,0] *= 100
    # function spaces DG0 for the scalar, and RT1 for the gradient
    
    V = FunctionSpace(mesh, "DG", polynomial_order)
    U = VectorFunctionSpace(mesh, "CG", polynomial_order+1)
    W = FunctionSpace(mesh, "CG", polynomial_order)
    M = MixedFunctionSpace([U, W])
    # set up prescribed velocity and diffusivity
    x, y = SpatialCoordinate(mesh)
    m = Function(M)
    vel_, p_ = m.split()  # function: y component of velocity, pressure
    vel, p = split(m)  # expression: y component of velocity, pressure
    vel_._name = "velocity"
    p_._name = "perturbation pressure"

    vel_init = zero(mesh.geometric_dimension())

    u_ana = sin(x)*cos(y)
    v_ana = -cos(x)*sin(y)
    p_ana = cos(x)*cos(y)
    vel_ana = as_vector((u_ana,v_ana))
    vel_ana_f = Function(U, name='vel analytical').project(vel_ana)
   
    vel_.assign(vel_init)
    File('vel.pvd').write(vel_)

    p_ana_f = Function(W, name='p analytical').project(p_ana)
    
    p_.assign(p_ana_f)
    mu = Constant(0.7)
    u_source = 2*mu*sin(x)*cos(y) + sin(x)*sin(y)**2*cos(x) + sin(x)*cos(x)*cos(y)**2 - sin(x)*cos(y)
    v_source = -2*mu*sin(y)*cos(x) + sin(x)**2*sin(y)*cos(y) + sin(y)*cos(x)**2*cos(y) - sin(y)*cos(x)
    vel_source = as_vector((u_source, v_source))
    # the diffusivity
    kappa = Constant(0.7)

    # We declare the output filename, and write out the initial condition. ::
    vel_outfile = File("vel_gz_mms.pvd")
    vel_outfile.write(vel_, vel_ana_f)
    p_outfile = File("p_gz_mms.pvd")
    p_outfile.write(p_, p_ana_f)

    # a big timestep, which means BackwardEuler takes us to a steady state almost immediately
    # (needs to be smaller at polynomial_degree>0, 0.1/nx works for p=1 for 4 meshes)
    dt = Constant(0.001)

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
    
    no_normal_flow = 0.0
    vp_bcs = {} #4: {'u': vel_ana}, 2: {'u': vel_ana},
#          3: {'u': vel_ana}, 1: {'u': vel_ana}}
    
    strong_bcs = [DirichletBC(M.sub(0),  vel_ana, 1),
                DirichletBC(M.sub(0),  vel_ana, 2),
                DirichletBC(M.sub(0),  vel_ana, 3),
                DirichletBC(M.sub(0),  vel_ana, 4)]
    mumps_solver_parameters = {
        'snes_monitor': None,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij',
        'snes_max_it': 100,
        'snes_rtol': 1e-8,
    }

    vp_timestepper = CrankNicolsonSaddlePointTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
                                                        solver_parameters=mumps_solver_parameters, strong_bcs=strong_bcs)
    #vp_timestepper = PressureProjectionTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
    #                                                      solver_parameters=mumps_solver_parameters,
    #                                                      predictor_solver_parameters=mumps_solver_parameters,
    #                                                      picard_iterations=1)
     #                                                     pressure_nullspace=VectorSpaceBasis(constant=True))
   # vp_timestepper.initialize_pressure()
    

    v_old, p_old = vp_timestepper.solution_old.split()
    vel_prev = Function(U, name='vel_old')
    p_prev = Function(W, name='p_old')

    t = 0.0
    step = 0
    vel_change = 1.0
    p_change = 1.0
    while (vel_change>1e-9) and (p_change>1e-9): 

        vel_prev.assign(v_old)
        p_prev.assign(p_old)
        vp_timestepper.advance(t)
        step += 1
        t += float(dt)

        vel_change = norm(vel_-vel_prev)
        p_change = norm(p_- p_prev)
        if step % output_freq == 0:
            vel_outfile.write(vel_, vel_ana_f)
            p_outfile.write(p_, p_ana_f)
            print("vel/p change =", vel_change, p_change)

    
    vel_err = norm(vel-vel_ana)
    p_err = norm(p-p_ana)
    return temp_err, sal_err, melt_err, integrated_melt_err, vel_err, p_err

errors = np.array([error(meshes[i], cells[i]) for i in range(2)]) #10*2**np.arange(number_of_grids)])
conv = np.log(errors[:-1]/errors[1:])/np.log(2)

print('Velocity errors:', errors[:,0])
print('Pressure errors:', conv[:,1])
print('Velocity convergence order:', conv[:,0])
print('Pressure convergence order:', conv[:,1])
assert all(conv[:,0]> polynomial_order+0.95)
assert all(conv[:,1]> polynomial_order+0.95)

