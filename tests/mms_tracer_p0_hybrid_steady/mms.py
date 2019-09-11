# MMS test for scalar advection diffusion of a tracer
# based on a fluidity one
from thwaites import *
from thwaites.equations import BaseEquation
import numpy as np

quad = True
polynomial_order = 0
number_of_grids = 4
output_freq = 100

def error(nx):
    mesh = RectangleMesh(nx, nx, 0.5, 0.4, quadrilateral=quad)

    mesh.coordinates.dat.data[:] += (0.1,-0.3)

    # function spaces DG0 for the scalar, and RT1 for the gradient
    if quad:
        V = FunctionSpace(mesh, "DQ", polynomial_order)
        H = FunctionSpace(mesh, "RTCF", polynomial_order+1)
        W = VectorFunctionSpace(mesh, "CQ", polynomial_order+1)
    else:
        V = FunctionSpace(mesh, "DG", polynomial_order)
        H = FunctionSpace(mesh, "RT", polynomial_order+1)
        W = VectorFunctionSpace(mesh, "CG", polynomial_order+1)
    Z = V*H

    # set up prescribed velocity and diffusivity
    x, y = SpatialCoordinate(mesh)
    velocity = as_vector((sin(5*(x*x+y*y)), cos(3*(x*x-y*y))))
    u = Function(W).interpolate(velocity)
    File('u.pvd').write(u)

    # the diffusivity
    kappa = Constant(0.7)

    q_ana = sin(25*x*y)-2*y/sqrt(x)
    q_ana_dg0 = Function(V, name='analytical').project(q_ana)

    adv = 1  # can be used to switch off advection term
    x1p5 = x*sqrt(x)
    x2 = x*x
    y2 = y*y
    xp5 = sqrt(x)
    x2p5 = x*x*sqrt(x)
    nu = kappa
    beta = 0 # ???
    source = adv*((25*y*cos(25*x*y) + y/x1p5)*sin(5*(y2 + x2)) + beta*(sin(25*x*y) - 2*y/xp5)*(10*x*cos(5*(y2 + x2)) + 6*y*sin(3*(x2 - y2))) + (25*x*cos(25*x*y) - 2/xp5)*cos(3*(x2 - y2))) - nu*(-625*y2*sin(25*x*y) - 625*x2*sin(25*x*y) - 3*y/(2*x2p5))

    z = Function(Z)
    q, p = z.split()

    # We declare the output filename, and write out the initial condition. ::
    outfile = File("advdif.pvd")
    outfile.write(q, q_ana_dg0)

    outfile_grad = File("advdif_grad.pvd")
    outfile_grad.write(p)

    # a big timestep, which means BackwardEuler takes us to a steady state almost immediately
    # (needs to be smaller at polynomial_degree>0, 0.1/nx works for p=1 for 4 meshes)
    dt = Constant(1.0/nx)

    eq = HybridizedScalarEquation(ScalarAdvectionEquation, Z, Z)

    fields = {'velocity': u, 'diffusivity': kappa, 'source': source, 'dt': dt}

    # boundary conditions, bottom and left are inflow
    # so Dirichlet, with others specifying a flux
    left_id, right_id, bottom_id, top_id = range(1, 5)
    one = Constant(1.0)
    bcs = {
        left_id: {'q': q_ana},
        bottom_id: {'q': q_ana},
        right_id:  {'flux':  kappa*grad(q_ana)[0]},
        top_id: {'flux': kappa*grad(q_ana)[1]}
    }

    timestepper = BackwardEuler(eq, z, fields, dt, bcs)

    q_old, p_old = timestepper.solution_old.split()
    q_prev = Function(V, name='q_old')

    t = 0.0
    step = 0
    change = 1.0
    while change>1e-9:

        p_old.assign(0)
        p.assign(0)
        q_prev.assign(q_old)
        timestepper.advance(t)

        step += 1
        t += float(dt)

        change = norm(q-q_prev)
        if step % output_freq == 0:
            outfile.write(q, q_ana_dg0)
            outfile_grad.write(p)
            print("t, change =", t, change)


    err = norm(q-q_ana)
    print('Error at nx ={}: {}'.format(nx, err))
    return err

errors = np.array([error(nx) for nx in 10*2**np.arange(number_of_grids)])
conv = np.log(errors[:-1]/errors[1:])/np.log(2)

print('Errors: ', errors)
print('Convergence order:', conv)
assert all(conv> polynomial_order+0.95)
