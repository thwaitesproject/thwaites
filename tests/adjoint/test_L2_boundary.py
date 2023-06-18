from thwaites import *
from thwaites.adjoint_utility import RieszL2BoundaryRepresentation
from firedrake_adjoint import *
import numpy
import os.path


def test_L2_boundary():
    # tests RieszL2BoundaryRepresentation which converts a l2 adjoint gradient
    # into a L2-boundary Riesz representative

    # need mesh with non-uniform resolution to test mesh-dependence
    mesh = Mesh(os.path.join(os.path.dirname(__file__), "square.msh"))

    Q = FunctionSpace(mesh, "CG", 1)
    T = Function(Q)
    Tbc = Function(Q)
    Tbc.assign(1.)

    V = VectorFunctionSpace(mesh, "CG", 1)
    u = Function(V)
    u.interpolate(as_vector((1, 0)))

    mumps_solver_parameters = {
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        "mat_mumps_icntl_14": 200,
        'mat_type': 'aij',
    }

    fields = {'velocity': u, 'diffusivity': Constant(1e-4)}
    bcs = {1: {'q': Tbc}}
    dt = 0.01

    eq = ScalarAdvectionDiffusionEquation(Q, Q)

    timestepper = DIRK33(eq, T, fields, dt, bcs, solver_parameters=mumps_solver_parameters)

    #  f = File('test.pvd')

    t = 0.0
    while t < 1.2:
        timestepper.advance(t)
        t += dt
        #  f.write(T)

    # define non-uniform functional
    x, y = SpatialCoordinate(mesh)
    J = assemble(T*y**2*ds(2))

    rf = ReducedFunctional(J, Control(Tbc))
    grad_l2 = rf.derivative()
    grad_l2.rename("l2 derivative")
    grad_L2 = grad_l2._ad_convert_type(grad_l2, options={'riesz_representation': 'L2'})
    grad_L2.rename("L2 derivative")
    converter = RieszL2BoundaryRepresentation(Q, 1)
    grad_L2b = grad_l2._ad_convert_type(grad_l2, options={'riesz_representation': converter})
    grad_L2b.rename("L2 boundary derivative")
    File('grad.pvd').write(grad_l2, grad_L2, grad_L2b)

    yrange = numpy.linspace(0, 1, 100)
    gradvals = [grad_L2b.at([0, y]) for y in yrange]
    numpy.testing.assert_allclose(gradvals, yrange**2, atol=0.1)
