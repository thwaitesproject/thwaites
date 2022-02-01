# MMS test for scalar advection diffusion of a tracer
# based on a fluidity one
from thwaites import *
from thwaites.equations import BaseEquation
from thwaites.meltrate_param import TwoEqMeltRateParam
from thwaites.utility import CombinedSurfaceMeasure
import numpy as np
from math import ceil

polynomial_order = 1
number_of_grids =6 
output_freq = 100
H1 = 0.04
H2 = 0.4
L = 0.5

def error(nx):
    dx = L/nx
    dz = H2/nx
    mesh1d = IntervalMesh(nx, 0.5)
    layers = []
    cell = 0
    xr = 0
    min_dz = 0.01*dz # if top cell is thinner than this, merge with cell below
    tiny_dz = 1e-9*dz #  workaround for zero measure exterior facets (fd issue #1858)
    for i in range(nx):
        xr += dx  # y of right-node (assumed to be the higher one)
        height = H1 + xr/L * (H2-H1)
        ncells = ceil((height-min_dz)/dz)
        layers.append([0, ncells])
        cell += ncells
    mesh = ExtrudedMesh(mesh1d, layers, layer_height=dz)
    y = mesh.coordinates.dat.data_ro[:,0]
    z = mesh.coordinates.dat.data_ro[:,1]
    height = np.maximum(H1 + y/L * (H2-H1), H1 + tiny_dz)
    mesh.coordinates.dat.data[:,1] = np.minimum(height, z)

    mesh.coordinates.dat.data[:] += (0.1,-0.5)

    ds = CombinedSurfaceMeasure(mesh, 5)
    #mesh.coordinates.dat.data[:,0] *= 100
    # function spaces DG0 for the scalar, and RT1 for the gradient
    V = FunctionSpace(mesh, "Q", polynomial_order)
    W = VectorFunctionSpace(mesh, "Q", polynomial_order)

    # set up prescribed velocity and diffusivity
    x, y = SpatialCoordinate(mesh)
    velocity = as_vector((sin(5*(x*x+y*y)), cos(3*(x*x-y*y))))
    u = Function(W).interpolate(velocity)
    File('u.pvd').write(u)

    # the diffusivity
    kappa = Constant(0.7)

    temp_ana = sin(25*x*y)-2*y/sqrt(x)
    temp_ana_f = Function(V, name='temp analytical').project(temp_ana)
    sal_ana = 30+cos(25*x*y)-2*y/sqrt(x)
    sal_ana_f = Function(V, name='sal analytical').project(sal_ana)

    temp_source = -kappa*(-625*x**2*sin(25*x*y) - y*(625*y*sin(25*x*y) + 3/(2*x**(5/2)))) + (25*x*cos(25*x*y) - 2/sqrt(x))*cos(3*x**2 - 3*y**2) + (25*y*cos(25*x*y) + y/x**(3/2))*sin(5*x**2 + 5*y**2) # sympy 08.01.22
    sal_source = -kappa*(-625*x**2*cos(25*x*y) - y*(625*y*cos(25*x*y) + 3/(2*x**(5/2)))) + (-25*x*sin(25*x*y) - 2/sqrt(x))*cos(3*x**2 - 3*y**2) + (-25*y*sin(25*x*y) + y/x**(3/2))*sin(5*x**2 + 5*y**2) 
    
    temp = Function(V, name='temperature').assign(0.0)
    sal = Function(V, name='salinity').assign(10.0) # Initialising salinity with zero leads to nans, probably because messes up the melt rate with S<0.
    melt = Function(V, name='melt')
    
    # We declare the output filename, and write out the initial condition. ::
    temp_outfile = File("temp_advdif.pvd")
    temp_outfile.write(temp, temp_ana_f)
    sal_outfile = File("sal_advdif.pvd")
    sal_outfile.write(sal, sal_ana_f)

    # a big timestep, which means BackwardEuler takes us to a steady state almost immediately
    # (needs to be smaller at polynomial_degree>0, 0.1/nx works for p=1 for 4 meshes)
    dt = Constant(1.0/nx)

    temp_eq = ScalarAdvectionDiffusionEquation(V, V)
    sal_eq = ScalarAdvectionDiffusionEquation(V, V)

    temp_fields = {'velocity': u, 'diffusivity': kappa, 'source': temp_source}
    sal_fields = {'velocity': u, 'diffusivity': kappa, 'source': sal_source}

    # boundary conditions, bottom and left are inflow
    # so Dirichlet, with others specifying a flux
    left_id, right_id, bottom_id, top_id = 1, 2, "bottom", "top"
    one = Constant(1.0)
    n = FacetNormal(mesh)
    
    # melting
    mp = ThreeEqMeltRateParam(sal, temp, 0, y)
    # T flux as calculated by melt rate using scipy. Used second solution from quadratic equation for Salinity at ice ocean boundary 
    T_flux_bc_sympy =  (-(-3.34629218167539e-9*y - 8.83700642033444e-6*(0.143778882407223*(0.000759376404*y + 0.0832)*(0.00101*cos(25*x*y) + 0.0303 - 0.00202*y/sqrt(x)) +\
            (0.000378668071800354*y - 0.499927196057832*sin(25*x*y) - 7.28039421677275e-5*cos(25*x*y) - 0.198293394983272 + y/sqrt(x))**2 + 0.0271555175202523*cos(25*x*y) +\
            0.814665525607568 - 0.0543110350405046*y/sqrt(x))**0.5 + 4.41785984126286e-6*sin(25*x*y) + 5.05643368904362e-7*cos(25*x*y) + 1.69023200045771e-5 -\
            9.84700642033444e-6*y/sqrt(x))/(0.00662632115183246*y + 17.4990226145236*(0.143778882407223*(0.000759376404*y + 0.0832)*(0.00101*cos(25*x*y) + 0.0303 -\
            0.00202*y/sqrt(x)) + (0.000378668071800354*y - 0.499927196057832*sin(25*x*y) - 7.28039421677275e-5*cos(25*x*y) - 0.198293394983272 + y/sqrt(x))**2 +\
            0.0271555175202523*cos(25*x*y) + 0.814665525607568 - 0.0543110350405046*y/sqrt(x))**0.5 - 8.74823730943141*sin(25*x*y) - 0.00127399783041954*cos(25*x*y) -\
            3.46994060312294 + 17.4990226145236*y/sqrt(x)) - 0.0001)*(0.000379688202*y - 1.0026939958122*(0.143778882407223*(0.000759376404*y + 0.0832)*(0.00101*cos(25*x*y)+\
            0.0303 - 0.00202*y/sqrt(x)) + (0.000378668071800354*y - 0.499927196057832*sin(25*x*y) - 7.28039421677275e-5*cos(25*x*y) - 0.198293394983272 + y/sqrt(x))**2 +\
            0.0271555175202523*cos(25*x*y) + 0.814665525607568 - 0.0543110350405046*y/sqrt(x))**0.5 - 0.49872600216958*sin(25*x*y) + 7.30000756830394e-5*cos(25*x*y) +\
            0.282027596558944 + 0.997306004187795*y/sqrt(x))

    S_flux_bc_sympy = (-(-3.34629218167539e-9*y - 8.83700642033444e-6*(0.143778882407223*(0.000759376404*y + 0.0832)*(0.00101*cos(25*x*y) + 0.0303 - 0.00202*y/sqrt(x)) +\
            (0.000378668071800354*y - 0.499927196057832*sin(25*x*y) - 7.28039421677275e-5*cos(25*x*y) - 0.198293394983272 + y/sqrt(x))**2 + 0.0271555175202523*cos(25*x*y) +\
            0.814665525607568 - 0.0543110350405046*y/sqrt(x))**0.5 + 4.41785984126286e-6*sin(25*x*y) + 5.05643368904362e-7*cos(25*x*y) + 1.69023200045771e-5 - \
            9.84700642033444e-6*y/sqrt(x))/(0.00662632115183246*y + 17.4990226145236*(0.143778882407223*(0.000759376404*y + 0.0832)*(0.00101*cos(25*x*y) + 0.0303 - \
            0.00202*y/sqrt(x)) + (0.000378668071800354*y - 0.499927196057832*sin(25*x*y) - 7.28039421677275e-5*cos(25*x*y) - 0.198293394983272 + y/sqrt(x))**2 + \
            0.0271555175202523*cos(25*x*y) + 0.814665525607568 - 0.0543110350405046*y/sqrt(x))**0.5 - 8.74823730943141*sin(25*x*y) - 0.00127399783041954*cos(25*x*y) - \
            3.46994060312294 + 17.4990226145236*y/sqrt(x)) - 5.05e-7)*(0.00662632115183246*y + 17.4990226145236*(0.143778882407223*(0.000759376404*y + 0.0832)*(0.00101*cos(25*x*y) +\
            0.0303 - 0.00202*y/sqrt(x)) + (0.000378668071800354*y - 0.499927196057832*sin(25*x*y) - 7.28039421677275e-5*cos(25*x*y) - 0.198293394983272 + y/sqrt(x))**2 + \
            0.0271555175202523*cos(25*x*y) + 0.814665525607568 - 0.0543110350405046*y/sqrt(x))**0.5 - 8.74823730943141*sin(25*x*y) - 1.00127399783042*cos(25*x*y) - \
            33.4699406031229 + 19.4990226145236*y/sqrt(x))

    temp_melt_source =  kappa*dot(n, grad(temp_ana)) + T_flux_bc_sympy
    sal_melt_source =  kappa*dot(n, grad(sal_ana)) + S_flux_bc_sympy
    melt_ana = (-3.34629218167539e-9*y - 8.83700642033444e-6*(0.143778882407223*(0.000759376404*y + 0.0832)*(0.00101*cos(25*x*y) + 0.0303 - 0.00202*y/sqrt(x)) + (0.000378668071800354*y - 0.499927196057832*sin(25*x*y) - 7.28039421677275e-5*cos(25*x*y) - 0.198293394983272 + y/sqrt(x))**2 + 0.0271555175202523*cos(25*x*y) + 0.814665525607568 - 0.0543110350405046*y/sqrt(x))**0.5 + 4.41785984126286e-6*sin(25*x*y) + 5.05643368904362e-7*cos(25*x*y) + 1.69023200045771e-5 - 9.84700642033444e-6*y/sqrt(x))/(0.00662632115183246*y + 17.4990226145236*(0.143778882407223*(0.000759376404*y + 0.0832)*(0.00101*cos(25*x*y) + 0.0303 - 0.00202*y/sqrt(x)) + (0.000378668071800354*y - 0.499927196057832*sin(25*x*y) - 7.28039421677275e-5*cos(25*x*y) - 0.198293394983272 + y/sqrt(x))**2 + 0.0271555175202523*cos(25*x*y) + 0.814665525607568 - 0.0543110350405046*y/sqrt(x))**0.5 - 8.74823730943141*sin(25*x*y) - 0.00127399783041954*cos(25*x*y) - 3.46994060312294 + 17.4990226145236*y/sqrt(x))

    temp_bcs = {
        left_id: {'q': temp_ana},
        bottom_id: {'q': temp_ana},
        right_id:  {'flux':  kappa*dot(n, grad(temp_ana))},
        top_id: {'flux': -mp.T_flux_bc + temp_melt_source},
    }
    sal_bcs = {
        left_id: {'q': sal_ana},
        bottom_id: {'q': sal_ana},
        right_id:  {'flux':  kappa*dot(n, grad(sal_ana))},
        top_id:  {'flux': -mp.S_flux_bc + sal_melt_source},
    }

    temp_timestepper = BackwardEuler(temp_eq, temp, temp_fields, dt, temp_bcs, 
            solver_parameters={
                'snes_type': 'ksponly',
            })
    sal_timestepper = BackwardEuler(sal_eq, sal, sal_fields, dt, sal_bcs, 
            solver_parameters={
                'snes_type': 'ksponly',
            })

    temp_old = temp_timestepper.solution_old
    temp_prev = Function(V, name='temp_old')
    sal_old = sal_timestepper.solution_old
    sal_prev = Function(V, name='sal_old')

    t = 0.0
    step = 0
    temp_change = 1.0
    sal_change = 1.0
    while (sal_change > 1e-9) and (temp_change>1e-9):

        temp_prev.assign(temp_old)
        temp_timestepper.advance(t)
        sal_prev.assign(sal_old)
        sal_timestepper.advance(t)
        step += 1
        t += float(dt)

        temp_change = norm(temp-temp_prev)
        sal_change = norm(sal-sal_prev)
        if step % output_freq == 0:
            temp_outfile.write(temp, temp_ana_f)
            sal_outfile.write(sal, sal_ana_f)
            print("t, temp/sal change =", t, temp_change, sal_change)

    
    temp_err = norm(temp-temp_ana)
    print('Temperature error at nx ={}: {}'.format(nx, temp_err))
    sal_err = norm(sal-sal_ana)
    print('Salinity error at nx ={}: {}'.format(nx, sal_err))

    melt.interpolate(mp.wb)
    melt_err = norm(melt - melt_ana)
    integrated_melt = assemble(melt * ds('top'))
    integrated_melt_ana = assemble(melt_ana * ds('top'))
    integrated_melt_err = abs(integrated_melt - integrated_melt_ana)
    return temp_err, sal_err, melt_err, integrated_melt_err

errors = np.array([error(nx) for nx in 10*2**np.arange(number_of_grids)])
conv = np.log(errors[:-1]/errors[1:])/np.log(2)

print('Temperature errors: ', errors[:,0])
print('Salinity errors: ', errors[:,1])
print('Melt errors:', conv[:,2])
print('Integrated melt errors:', conv[:,3])
print('Temperature convergence order:', conv[:,0])
print('Salinity convergence order:', conv[:,1])
print('Melt convergence order:', conv[:,2])
print('Integrated melt convergence order:', conv[:,3])
assert all(conv[:,0]> polynomial_order+0.95)
assert all(conv[:,1]> polynomial_order+0.95)

