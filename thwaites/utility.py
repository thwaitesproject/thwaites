"""
A module with utitity functions for Thwaites
"""
from firedrake import outer


def is_continuous(ufl):
    elem = ufl.ufl_element()

    family = elem.family()
    if family == 'Lagrange':
        return True
    elif family == 'Discontinuous Lagrange' or family == 'DQ':
        return False
    else:
        raise NotImplemented('Unknown finite element family')


def normal_is_continuous(ufl):
    elem = ufl.ufl_element()

    family = elem.family()
    if family == 'Lagrange':
        return True
    elif family == 'Discontinuous Lagrange' or family == 'DQ':
        return False
    else:
        raise NotImplemented('Unknown finite element family')


def tensor_jump(v, n):
    r"""
    Jump term for vector functions based on the tensor product

    .. math::
        \text{jump}(\mathbf{u}, \mathbf{n}) = (\mathbf{u}^+ \mathbf{n}^+) +
        (\mathbf{u}^- \mathbf{n}^-)

    This is the discrete equivalent of grad(u) as opposed to the
    vectorial UFL jump operator :meth:`ufl.jump` which represents div(u).
    The equivalent of nabla_grad(u) is given by tensor_jump(n, u).
    """
    return outer(v('+'), n('+')) + outer(v('-'), n('-'))


# ice thickness shouldn't be used!!! unless the water depth is not relative to 1km!!!
def ice_thickness(x,x0,y0,x1,y1):
    m = (y1-y0)/(x1-x0)
    return y0 + m*x


def cavity_thickness(x,x0,y0,x1,y1):
    m = (y1-y0)/(x1-x0)
    return y0 + m*x


def get_top_boundary(cavity_length=5000., cavity_height=100., water_depth=1000., n=100):
    dx = cavity_length / float(n)
    shelf_boundary_points = []
    for i in range(n):
        x_i = i * dx
        y_i = cavity_thickness(x_i, 0.0, 2.0, cavity_length, cavity_height) - water_depth - 0.01
        shelf_boundary_points.append([x_i, y_i])

    return shelf_boundary_points
