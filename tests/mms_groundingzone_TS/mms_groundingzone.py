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
horizontal_stretching = 1
L = 100 * horizontal_stretching
g = 9.81
T_ref = Constant(-1.0)
S_ref = Constant(34.2)
beta_temp = Constant(3.733E-5)
beta_sal = Constant(7.843E-4)
depth = 1000
cells = [10,20,40,80]  # i.e 10x10, 20x20, 40x40, 80x80
meshes = ["verification_unstructured_100m_square_res10m.msh",
        "verification_unstructured_100m_square_res5m.msh", 
        "verification_unstructured_100m_square_res2.5m.msh", 
        "verification_unstructured_100m_square_res1.25m.msh"]
print(depth)
def error(mesh_name, nx):
    dx = L/nx
    depth = 1000
    dz = H2/nx
    print(type(depth))
    mesh = Mesh(mesh_name)
    mesh.coordinates.dat.data[:,1] -= depth
    mesh.coordinates.dat.data[:,0] *= horizontal_stretching
    print(mesh.coordinates.dat.data[:,0].min())
    print(mesh.coordinates.dat.data[:,0].max())

    depth = Constant(depth)
    # function spaces DG0 for the scalar, and RT1 for the gradient
    
    V = FunctionSpace(mesh, "DG", polynomial_order)
    U = VectorFunctionSpace(mesh, "DG", polynomial_order)
    W = FunctionSpace(mesh, "CG", polynomial_order+1)
    M = MixedFunctionSpace([U, W])
    # set up prescribed velocity and diffusivity
    x, y = SpatialCoordinate(mesh)
    print(type(y))
    m = Function(M)
    vel_, p_ = m.split()  # function: y component of velocity, pressure
    vel, p = split(m)  # expression: y component of velocity, pressure
    vel_._name = "velocity"
    p_._name = "perturbation pressure"

    vel_init = zero(mesh.geometric_dimension())


    arg = - np.pi / H2 * (y + depth - H2)
    u_ana =  x / L * cos(arg)
    v_ana = (H2 / np.pi) * sin(arg) / L
    vel_ana = as_vector((u_ana,v_ana))
    vel_ana_f = Function(U, name='vel analytical').project(vel_ana)
    vel_.interpolate(vel_ana - 0.01*vel_ana)
    #vel_.interpolate(vel_ana)

    #p_ana = -cos(np.pi * x / L) + cos(arg) 
    p_ana = Constant(0) 
    p_ana_f = Function(W, name='p analytical').project(p_ana)
    p_.assign(0.0)
    mu_h = Constant(1*horizontal_stretching)
    mu_v = Constant(1)
    mu = as_tensor([[mu_h, 0], [0, mu_v]])
    u_source = 1.0*x*sin(3.14159265358979*(-H2 + depth + y)/H2)**2/L**2 + x*cos(3.14159265358979*(-H2 + depth + y)/H2)**2/L**2 + 9.86960440108936*mu_v*x*cos(3.14159265358979*(-H2 + depth + y)/H2)/(H2**2*L) 
    v_source =  0.318309886183791*H2*sin(3.14159265358979*(-H2 + depth + y)/H2)*cos(3.14159265358979*(-H2 + depth + y)/H2)/L**2 + g*(beta_sal*(-S_ref - 0.000144444444444565*y**2 - 0.281444444444675*y + 0.345*cos(12.5663706143592*x/L) - 102.50000000011) - beta_temp*(-T_ref - 0.00038888888888923*y**2 - 0.753888888889541*y + 0.1*sin(12.5663706143592*x/L) - 364.00000000031)) - 3.14159265358979*mu_v*sin(3.14159265358979*(-H2 + depth + y)/H2)/(H2*L) 
   
    vel_source = as_vector((u_source, v_source))
    
    # the diffusivity
    kappa = Constant(1)

    depths = [-900, -910, -1000]
    temperature = [-0.5, 0.0, 1]
    p_temperature = np.polyfit(depths, temperature, 2) 
    print("p = ", p_temperature)

    salinity = [33.8,34.0, 34.5]
    p_salt = np.polyfit(depths, salinity, 2) 
    print("p salt = ", p_salt)
    depth = 1000
    temp_ana =  0.1*sin(4*np.pi*x/L) + p_temperature[0]*pow(y,2) + p_temperature[1]*y + p_temperature[2]
    temp_ana_f = Function(V, name='temp analytical').project(temp_ana + 0.01*temp_ana)
    sal_ana  = 0.01* 34.5 * cos(4*np.pi*x/L) + p_salt[0]*pow(y,2) + p_salt[1]*y + p_salt[2]
    sal_ana_f = Function(V, name='sal analytical').project(sal_ana+0.01*sal_ana)
    print("type H2" , type(H2))
    print("type kappa" , type(kappa))
    print("type L" , type(L))

    temp_source =  -0.318309886183791*H2*(-0.000777777777778461*y - 0.753888888889541)*sin(3.14159265358979*(-H2 + depth + y)/H2)/L - kappa*(-0.000777777777778461 - 15.791367041743*sin(12.5663706143592*x/L)/L**2) + 1.25663706143592*x*cos(3.14159265358979*(-H2 + depth + y)/H2)*cos(12.5663706143592*x/L)/L**2

    sal_source = -0.318309886183791*H2*(-0.000288888888889131*y - 0.281444444444675)*sin(3.14159265358979*(-H2 + depth + y)/H2)/L - kappa*(-0.000288888888889131 - 54.4802162940133*cos(12.5663706143592*x/L)/L**2) - 4.33539786195391*x*sin(12.5663706143592*x/L)*cos(3.14159265358979*(-H2 + depth + y)/H2)/L**2 

    temp = Function(V, name='temperature').interpolate(temp_ana) #+ 0.01*temp_ana)
    sal = Function(V, name='salinity').interpolate(sal_ana) # + 0.01*sal_ana) # Initialising salinity with zero leads to nans, probably because messes up the melt rate with S<0.
    
    mom_source = as_vector((0.,-g))*(-beta_temp*(temp - T_ref) + beta_sal * (sal - S_ref)) 
    
    melt = Function(V, name='melt')
    
    # We declare the output filename, and write out the initial condition. ::
    vel_outfile = File("vel_gz_mms_TSmelt_L"+str(L)+"_nx"+str(nx)+"buoyancy_staticTS.pvd")
    vel_outfile.write(vel_, vel_ana_f)
    p_outfile = File("p_gz_mms_TSmelt_L"+str(L)+"_nx"+str(nx)+"buoyancy_staticTS.pvd")
    p_outfile.write(p_, p_ana_f)
    temp_outfile = File("temp_gz_mms_TSmelt_L"+str(L)+"_nx"+str(nx)+"buoyancy_staticTS.pvd")
    temp_outfile.write(temp, temp_ana_f)
    sal_outfile = File("sal_gz_mms_TSmelt_L"+str(L)+"_nx"+str(nx)+"buoyancy_staticTS.pvd")
    sal_outfile.write(sal, sal_ana_f)

    # a big timestep, which means BackwardEuler takes us to a steady state almost immediately
    # (needs to be smaller at polynomial_degree>0, 0.1/nx works for p=1 for 4 meshes)
    dt = Constant(L/nx)

    # Set up equations
    mom_eq = MomentumEquation(M.sub(0), M.sub(0))
    cty_eq = ContinuityEquation(M.sub(1), M.sub(1))
    temp_eq = ScalarAdvectionDiffusionEquation(V, V)
    sal_eq = ScalarAdvectionDiffusionEquation(V, V)

    vp_coupling = [{'pressure': 1}, {'velocity': 0}]
    vp_fields = {'viscosity': mu, 'source': vel_source + mom_source}
    temp_fields = {'velocity': vel, 'diffusivity': kappa, 'source': temp_source}
    sal_fields = {'velocity': vel, 'diffusivity': kappa, 'source': sal_source}

    # boundary conditions, bottom and left are inflow
    # so Dirichlet, with others specifying a flux
    left_id, right_id, bottom_id, top_id = 1, 2, 3, 4
    one = Constant(1.0)
    n = FacetNormal(mesh)
    
    # melting
    mp = ThreeEqMeltRateParam(sal, temp, 0, y)
    # T flux as calculated by melt rate using scipy. Used second solution from quadratic equation for Salinity at ice ocean boundary 
    T_flux_bc_sympy =   (-(-1.79109398044554e-9*y**2 - 3.47623225622948e-6*y - 0.001606428294557*(-1.19016244848201e-10*y**2 - 2.31898575538836e-7*y + 4.35093659764574e-6*(0.000759376404*y + 0.0832)*(-1.45888888889011e-7*y**2 - 0.000284258888889122*y + 0.00034845*cos(12.5663706143592*x/L) - 0.103525000000111) + (1.06954636059547e-6*y**2 + 0.00207547565184312*y - 0.000275011331425855*sin(12.5663706143592*x/L) - 1.38171291402741e-7*cos(12.5663706143592*x/L) + 1)**2 + 2.84265723271811e-7*cos(12.5663706143592*x/L) - 8.44557583634546e-5)**0.5 + 4.41785984126286e-7*sin(12.5663706143592*x/L) + 1.74446962272005e-7*cos(12.5663706143592*x/L) - 0.00165819079455705)/(0.00340227630891293*y**2 + 6.60218378571271*y + 3181.04612783564*(-1.19016244848201e-10*y**2 - 2.31898575538836e-7*y + 4.35093659764574e-6*(0.000759376404*y + 0.0832)*(-1.45888888889011e-7*y**2 - 0.000284258888889122*y + 0.00034845*cos(12.5663706143592*x/L) - 0.103525000000111) + (1.06954636059547e-6*y**2 + 0.00207547565184312*y - 0.000275011331425855*sin(12.5663706143592*x/L) - 1.38171291402741e-7*cos(12.5663706143592*x/L) + 1)**2 + 2.84265723271811e-7*cos(12.5663706143592*x/L) - 8.44557583634546e-5)**0.5 - 0.874823730943141*sin(12.5663706143592*x/L) - 0.00043952925149474*cos(12.5663706143592*x/L) + 3181.04612783564) - 0.0001)*(0.00019393845638852*y**2 + 0.376343134372203*y - 182.273943124982*(-1.19016244848201e-10*y**2 - 2.31898575538836e-7*y + 4.35093659764574e-6*(0.000759376404*y + 0.0832)*(-1.45888888889011e-7*y**2 - 0.000284258888889122*y + 0.00034845*cos(12.5663706143592*x/L) - 0.103525000000111) + (1.06954636059547e-6*y**2 + 0.00207547565184312*y - 0.000275011331425855*sin(12.5663706143592*x/L) - 1.38171291402741e-7*cos(12.5663706143592*x/L) + 1)**2 + 2.84265723271811e-7*cos(12.5663706143592*x/L) - 8.44557583634546e-5)**0.5 - 0.049872600216958*sin(12.5663706143592*x/L) + 2.51850261106486e-5*cos(12.5663706143592*x/L) + 181.809256875328)

    S_flux_bc_sympy =  (-(-1.79109398044554e-9*y**2 - 3.47623225622948e-6*y - 0.001606428294557*(-1.19016244848201e-10*y**2 - 2.31898575538836e-7*y + 4.35093659764574e-6*(0.000759376404*y + 0.0832)*(-1.45888888889011e-7*y**2 - 0.000284258888889122*y + 0.00034845*cos(12.5663706143592*x/L) - 0.103525000000111) + (1.06954636059547e-6*y**2 + 0.00207547565184312*y - 0.000275011331425855*sin(12.5663706143592*x/L) - 1.38171291402741e-7*cos(12.5663706143592*x/L) + 1)**2 + 2.84265723271811e-7*cos(12.5663706143592*x/L) - 8.44557583634546e-5)**0.5 + 4.41785984126286e-7*sin(12.5663706143592*x/L) + 1.74446962272005e-7*cos(12.5663706143592*x/L) - 0.00165819079455705)/(0.00340227630891293*y**2 + 6.60218378571271*y + 3181.04612783564*(-1.19016244848201e-10*y**2 - 2.31898575538836e-7*y + 4.35093659764574e-6*(0.000759376404*y + 0.0832)*(-1.45888888889011e-7*y**2 - 0.000284258888889122*y + 0.00034845*cos(12.5663706143592*x/L) - 0.103525000000111) + (1.06954636059547e-6*y**2 + 0.00207547565184312*y - 0.000275011331425855*sin(12.5663706143592*x/L) - 1.38171291402741e-7*cos(12.5663706143592*x/L) + 1)**2 + 2.84265723271811e-7*cos(12.5663706143592*x/L) - 8.44557583634546e-5)**0.5 - 0.874823730943141*sin(12.5663706143592*x/L) - 0.00043952925149474*cos(12.5663706143592*x/L) + 3181.04612783564) - 5.05e-7)*(0.00354672075335749*y**2 + 6.88362823015738*y + 3181.04612783564*(-1.19016244848201e-10*y**2 - 2.31898575538836e-7*y + 4.35093659764574e-6*(0.000759376404*y + 0.0832)*(-1.45888888889011e-7*y**2 - 0.000284258888889122*y + 0.00034845*cos(12.5663706143592*x/L) - 0.103525000000111) + (1.06954636059547e-6*y**2 + 0.00207547565184312*y - 0.000275011331425855*sin(12.5663706143592*x/L) - 1.38171291402741e-7*cos(12.5663706143592*x/L) + 1)**2 + 2.84265723271811e-7*cos(12.5663706143592*x/L) - 8.44557583634546e-5)**0.5 - 0.874823730943141*sin(12.5663706143592*x/L) - 0.345439529251495*cos(12.5663706143592*x/L) + 3283.54612783575)

    temp_melt_source =  kappa*dot(n, grad(temp_ana)) + T_flux_bc_sympy
    sal_melt_source =  kappa*dot(n, grad(sal_ana)) + S_flux_bc_sympy
    melt_ana =  (-1.79109398044554e-9*y**2 - 3.47623225622948e-6*y - 0.001606428294557*(-1.19016244848201e-10*y**2 - 2.31898575538836e-7*y + 4.35093659764574e-6*(0.000759376404*y + 0.0832)*(-1.45888888889011e-7*y**2 - 0.000284258888889122*y + 0.00034845*cos(12.5663706143592*x/L) - 0.103525000000111) + (1.06954636059547e-6*y**2 + 0.00207547565184312*y - 0.000275011331425855*sin(12.5663706143592*x/L) - 1.38171291402741e-7*cos(12.5663706143592*x/L) + 1)**2 + 2.84265723271811e-7*cos(12.5663706143592*x/L) - 8.44557583634546e-5)**0.5 + 4.41785984126286e-7*sin(12.5663706143592*x/L) + 1.74446962272005e-7*cos(12.5663706143592*x/L) - 0.00165819079455705)/(0.00340227630891293*y**2 + 6.60218378571271*y + 3181.04612783564*(-1.19016244848201e-10*y**2 - 2.31898575538836e-7*y + 4.35093659764574e-6*(0.000759376404*y + 0.0832)*(-1.45888888889011e-7*y**2 - 0.000284258888889122*y + 0.00034845*cos(12.5663706143592*x/L) - 0.103525000000111) + (1.06954636059547e-6*y**2 + 0.00207547565184312*y - 0.000275011331425855*sin(12.5663706143592*x/L) - 1.38171291402741e-7*cos(12.5663706143592*x/L) + 1)**2 + 2.84265723271811e-7*cos(12.5663706143592*x/L) - 8.44557583634546e-5)**0.5 - 0.874823730943141*sin(12.5663706143592*x/L) - 0.00043952925149474*cos(12.5663706143592*x/L) + 3181.04612783564)

    no_normal_flow = 0.0
    vp_bcs = {4: {'un': no_normal_flow}, 2: {'u': vel_ana},
          3: {'un': no_normal_flow }, 1: {'un': no_normal_flow}}
    
    temp_bcs = {
        left_id: {'q': temp_ana},
        bottom_id: {'flux':  kappa*dot(n, grad(temp_ana))},
        right_id:  {'q': temp_ana},
        top_id: {'flux': -mp.T_flux_bc + temp_melt_source},
    }
    sal_bcs = {
        left_id: {'q': sal_ana},
        bottom_id: {'flux':  kappa*dot(n, grad(sal_ana))},
        right_id:  {'q': sal_ana},
        top_id:  {'flux': -mp.S_flux_bc + sal_melt_source},
        } 
    mumps_solver_parameters = {
        'snes_monitor': None,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij',
        'snes_max_it': 100,
        'snes_rtol': 1e-8,
    }

    
    vp_timestepper = PressureProjectionTimeIntegrator([mom_eq, cty_eq], m, vp_fields, vp_coupling, dt, vp_bcs,
                                                          solver_parameters=mumps_solver_parameters,
                                                         predictor_solver_parameters=mumps_solver_parameters,
                                                         picard_iterations=2,
                                                          pressure_nullspace=VectorSpaceBasis(constant=True))
    
    temp_timestepper = BackwardEuler(temp_eq, temp, temp_fields, dt, temp_bcs, 
            solver_parameters={
                'snes_type': 'ksponly',
            })
    sal_timestepper = BackwardEuler(sal_eq, sal, sal_fields, dt, sal_bcs, 
            solver_parameters={
                'snes_type': 'ksponly',
            })

#    vp_timestepper.dt_const.assign(0.6*L/nx)
    v_old, p_old = vp_timestepper.solution_old.split()
    u_prev = Function(V, name='u_old')
    v_prev = Function(V, name='v_old')
    p_prev = Function(W, name='p_old')

    u_diff_abs = Function(V, name='u_diff_abs')
    v_diff_abs = Function(V, name='v_diff_abs')
    p_diff_abs = Function(W, name='p_diff_abs')
    temp_old = temp_timestepper.solution_old
    temp_prev = Function(V, name='temp_old')
    sal_old = sal_timestepper.solution_old
    sal_prev = Function(V, name='sal_old')

    t = 0.0
    step = 0
    temp_change = 1.0
    sal_change = 1.0
    u_change = 1.0
    v_change = 1.0
    p_change = 1.0
    #while (vel_change>1e-9) and (p_change>1e-9): #(sal_change > 1e-9) and (temp_change>1e-9) and (vel_change >1e-9:
    while (u_change> 1e-6) or (v_change>1e-6) or (p_change>1e-6): # or (sal_change>1e-6) or  (temp_change>1e-6): 

        u_prev.interpolate(v_old[0])
        v_prev.interpolate(v_old[1])
        p_prev.assign(p_old)
        vp_timestepper.advance(t)
        temp_prev.assign(temp_old)
        temp_timestepper.advance(t)
        sal_prev.assign(sal_old)
        sal_timestepper.advance(t)
        step += 1
        t += float(dt)

        u_change = norm(vel_[0]-u_prev)
        v_change = norm(vel_[1]-v_prev)
        p_change = norm(p_- p_prev)
        temp_change = norm(temp-temp_prev)
        sal_change = norm(sal-sal_prev)
        if step % output_freq == 0:
            temp_outfile.write(temp, temp_ana_f)
            sal_outfile.write(sal, sal_ana_f)
            vel_outfile.write(vel_, vel_ana_f)
            p_outfile.write(p_, p_ana_f)
            print("t, temp/sal change =", t, temp_change, sal_change)
            print("t, u/v/p change: ", t, u_change, v_change, p_change)

    
    u_err = norm(vel_[0]-u_ana)
    v_err = norm(vel_[1]-v_ana)
    p_err = norm(p_ - p_ana)
    
    temp_err = norm(temp-temp_ana)
    print('Temperature error at nx ={}: {}'.format(nx, temp_err))
    sal_err = norm(sal-sal_ana)
    print('Salinity error at nx ={}: {}'.format(nx, sal_err))

    melt.interpolate(mp.wb)
    melt_err = norm(melt - melt_ana)
    integrated_melt = assemble(melt * ds(4))
    integrated_melt_ana = assemble(melt_ana * ds(4))
    integrated_melt_err = abs(integrated_melt - integrated_melt_ana)
    return temp_err, sal_err, melt_err, integrated_melt_err, u_err, v_err, p_err

errors = np.array([error(meshes[i], cells[i]) for i in range(4)]) #10*2**np.arange(number_of_grids)])
conv = np.log(errors[:-1]/errors[1:])/np.log(2)

print('Temperature errors: ', errors[:,0])
print('Salinity errors: ', errors[:,1])
print('Melt errors:', errors[:,2])
print('Integrated melt errors:', errors[:,3])
print('U velocity errors:', errors[:,4])
print('V velocity errors:', errors[:,5])
print('Pressure errors:', errors[:,6])
print()
print('Temperature convergence order:', conv[:,0])
print('Salinity convergence order:', conv[:,1])
print('Melt convergence order:', conv[:,2])
print('Integrated melt convergence order:', conv[:,3])
print('U velocity convergence order:', conv[:,4])
print('V velocity convergence order:', conv[:,5])
print('Pressure convergence order:', conv[:,6])
assert all(conv[:,0]> polynomial_order+0.95)
assert all(conv[:,1]> polynomial_order+0.95)
