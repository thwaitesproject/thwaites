from pyadjoint import Block


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
