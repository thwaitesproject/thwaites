# MMS test for scalar advection diffusion of a tracer
# based on a fluidity one
from thwaites import *
from thwaites.equations import BaseEquation
import numpy as np

polynomial_order = 1
number_of_grids = 4
output_freq = 1

def error(nx):
    mesh1d = IntervalMesh(nx, 0.5)
    mesh = ExtrudedMesh(mesh1d, nx, layer_height=0.4/nx)

    mesh.coordinates.dat.data[:] += (0.1,-0.3)

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

    q_ana = sin(25*x*y)-2*y/sqrt(x)
    q_ana_f = Function(V, name='analytical').project(q_ana)

    adv = 1  # can be used to switch off advection term
    x1p5 = x*sqrt(x)
    x2 = x*x
    y2 = y*y
    xp5 = sqrt(x)
    x2p5 = x*x*sqrt(x)
    nu = kappa
    beta = 0 # ???
    source = adv*((25*y*cos(25*x*y) + y/x1p5)*sin(5*(y2 + x2)) + beta*(sin(25*x*y) - 2*y/xp5)*(10*x*cos(5*(y2 + x2)) + 6*y*sin(3*(x2 - y2))) + (25*x*cos(25*x*y) - 2/xp5)*cos(3*(x2 - y2))) - nu*(-625*y2*sin(25*x*y) - 625*x2*sin(25*x*y) - 3*y/(2*x2p5))

    q = Function(V, name='solution')

    # We declare the output filename, and write out the initial condition. ::
    outfile = File("advdif.pvd")
    outfile.write(q, q_ana_f)

    # a big timestep, which means BackwardEuler takes us to a steady state almost immediately
    # (needs to be smaller at polynomial_degree>0, 0.1/nx works for p=1 for 4 meshes)
    dt = Constant(1.0/nx)

    eq = ScalarAdvectionDiffusionEquation(V, V)

    fields = {'velocity': u, 'diffusivity': kappa, 'source': source}

    # boundary conditions, bottom and left are inflow
    # so Dirichlet, with others specifying a flux
    left_id, right_id, bottom_id, top_id = 1, 2, "bottom", "top"
    one = Constant(1.0)
    bcs = {
        left_id: {'q': q_ana},
        bottom_id: {'q': q_ana},
        right_id:  {'flux':  kappa*grad(q_ana)[0]},
        top_id: {'flux': kappa*grad(q_ana)[1]}
    }

    timestepper = BackwardEuler(eq, q, fields, dt, bcs, 
            solver_parameters={
                'snes_type': 'ksponly',
            })

    q_old = timestepper.solution_old
    q_prev = Function(V, name='q_old')

    t = 0.0
    step = 0
    change = 1.0
    while change>1e-9:

        q_prev.assign(q_old)
        timestepper.advance(t)

        step += 1
        t += float(dt)

        change = norm(q-q_prev)
        if step % output_freq == 0:
            outfile.write(q, q_ana_f)
            print("t, change =", t, change)


    err = norm(q-q_ana)
    print('Error at nx ={}: {}'.format(nx, err))
    return err

errors = np.array([error(nx) for nx in 10*2**np.arange(number_of_grids)])
conv = np.log(errors[:-1]/errors[1:])/np.log(2)

print('Errors: ', errors)
print('Convergence order:', conv)
assert all(conv> polynomial_order+0.95)
