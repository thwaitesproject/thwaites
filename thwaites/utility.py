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
