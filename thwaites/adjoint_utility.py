from pyadjoint import Block
from firedrake import DirichletBC, TestFunction, TrialFunction, utils, assemble, solve, dot, ds, Function
import numpy


class DiagnosticBlock(Block):
    def __init__(self, f, function):
        super().__init__()
        self.add_dependency(function)
        self.add_output(function.block_variable)
        self.f = f
        self.f_name = function.name()

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return block_variable.checkpoint

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        out = inputs[0]._ad_convert_type(adj_inputs[0], options={'riesz_representation': 'L2'})
        out.rename('adjoint_'+self.f_name)
        self.f.write(out)
        return 0


class DiagnosticConstantBlock(Block):
    def __init__(self, constant, name):
        super().__init__()
        self.add_dependency(constant)
        self.add_output(constant.block_variable)
        self.name = name

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return block_variable.checkpoint

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        out = inputs[0]._ad_convert_type(adj_inputs[0])
        print("Adjoint ", self.name, out.values()[0])
        return 0


class InteriorBC(DirichletBC):
    """DirichletBC applied to anywhere that is *not* on the specified boundary"""
    @utils.cached_property
    def nodes(self):
        return numpy.array(list(set(range(self._function_space.node_count)) - set(super().nodes)))


class RieszL2BoundaryRepresentation:
    """Callable that Converts l2-representatives to L2-boundary representatives"""
    def __init__(self, Q, bids):
        self.Q = Q
        v = TestFunction(Q)
        g = TrialFunction(Q)
        bc = InteriorBC(Q, 0, bids)
        self.M = assemble(dot(v, g)*ds(bids), bcs=bc)

    def __call__(self, value):
        ret = Function(self.Q)
        solve(self.M, ret, value)
        return ret
