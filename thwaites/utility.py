"""
A module with utitity functions for Thwaites
"""
from firedrake import outer, Function, assemble, TestFunction, dx
import pyop2 as op2
import sys

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

class AttrDict(dict):
    """
    Dictionary that provides both self['key'] and self.key access to members.

    http://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute-in-python
    """
    def __init__(self, *args, **kwargs):
        if sys.version_info < (2, 7, 4):
            raise Exception('AttrDict requires python >= 2.7.4 to avoid memory leaks')
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class P1Average(object):
    """
    Takes a discontinuous field and computes a P1 field by averaging around
    nodes

    Source must be either a P0 or P1DG :class:`Function`.
    The averaging operation is both mass conservative and positivity preserving.
    """
    def __init__(self, p0, p1, p1dg):
        """
        :arg p0: P0 function space
        :arg p1: P1 function space
        :arg p1dg: P1DG function space
        """
        self.p0 = p0
        self.p1 = p1
        self.p1dg = p1dg
        self.vol_p1 = Function(self.p1, name='nodal volume p1')
        self.vol_p1dg = Function(self.p1dg, name='nodal volume p1dg')
        self.update_volumes()

    def update_volumes(self):
        """Computes nodal volume of the P1 and P1DG function function_spaces

        This must be called when the mesh geometry is updated"""
        assemble(TestFunction(self.p1)*dx, self.vol_p1)
        assemble(TestFunction(self.p1dg)*dx, self.vol_p1dg)

    def apply(self, source, solution):
        """
        Averages discontinuous :class:`Function` :attr:`source` on P1
        :class:`Function` :attr:`solution`
        """
        assert solution.function_space() == self.p1
        assert source.function_space() == self.p0 or source.function_space() == self.p1dg
        source_is_p0 = source.function_space() == self.p0

        source_str = 'source[c]' if source_is_p0 else 'source[%(func_dim)d*d + c]'
        solution.assign(0.0)
        fs_source = source.function_space()
        self.kernel = op2.Kernel("""
            void my_kernel(double *p1_average, double *source, double *vol_p1, double *vol_p1dg) {
                for ( int d = 0; d < %(nodes)d; d++ ) {
                    for ( int c = 0; c < %(func_dim)d; c++ ) {
                        p1_average[%(func_dim)d*d + c] += %(source_str)s * vol_p1dg[%(func_dim)d*d + c] / vol_p1[%(func_dim)d*d + c];
                    }
                }
            }""" % {'nodes': solution.cell_node_map().arity,
                    'func_dim': solution.function_space().value_size,
                    'source_str': source_str},
            'my_kernel')

        op2.par_loop(
            self.kernel, self.p1.mesh().cell_set,
            solution.dat(op2.INC, self.p1.cell_node_map()),
            source.dat(op2.READ, fs_source.cell_node_map()),
            self.vol_p1.dat(op2.READ, self.p1.cell_node_map()),
            self.vol_p1dg.dat(op2.READ, self.p1dg.cell_node_map()),
            iterate=op2.ALL)


