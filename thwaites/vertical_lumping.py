from firedrake import *
from petsc4py import PETSc


class VerticallyLumpedPC(PCBase):
    """
    Preconditioner that implements vertical lumping approach
    that consists of a 2-level multigrid, where the fine level
    (original) problem is defined in a V_1 = V_hor x V_vert tensor
    product functionspace, and the coarse level in V_coarse = V_hor x R,
    i.e.  the vertically constant function space with the same
    horizontal function space.
    The operator at the coarse level is formed by P^T A P, where the
    prolongation operator P is formed by interpolation from V_coarse to V_1.
    This is the same approach as in https://doi.org/10.1016/j.ocemod.2010.08.001
    and similar to the vertical averaging in Marshall et al. (https://doi.org/10.1029/96JC02775)
    for the MIT model.
    Solver parameters for the coarse level can be set via the
    'lumped_mg_coarse_' prefix. Default is PETSc's LU so could be changed to
    mumps, or gamg to do algebraic multigrid on the horizontal mesh only.
    Default smoothing at the fine level (prefix 'lumped_mg_levels_') is
    Chebyshev+SOR.
    """

    def initialize(self, pc):
        options_prefix = pc.getOptionsPrefix()
        A, P = pc.getOperators()
        dm = pc.getDM()
        appctx = dm.getAppCtx()
        F = appctx[0].F
        V = F.arguments()[0].function_space()

        # create vertically constant version of functionspace
        mesh = V.mesh()
        hcell, vcell = mesh.ufl_cell().sub_cells()
        hele, _ = V.ufl_element().sub_elements()
        vele = FiniteElement("R", vcell, 0)
        ele = TensorProductElement(hele, vele)
        V_1layer = FunctionSpace(mesh, ele)

        # create interpolation matrix Prol from V_1layer to V
        v = TestFunction(V_1layer)
        interp = Interpolator(v, V)
        Prol = interp.callable().handle

        self.pc = PETSc.PC().create(comm=pc.comm)
        self.pc.setOptionsPrefix(options_prefix + 'lumped_')

        # hack: we actually want to call self.pc.setMGGalerkin()
        # but there appears to be no petsc4py interface
        options = PETSc.Options()
        options[options_prefix + 'lumped_pc_mg_galerkin'] = 'both'

        self.pc.setOperators(A, P)
        self.pc.setType("mg")
        self.pc.setMGLevels(2)
        self.pc.setMGInterpolation(1, Prol)
        self.pc.setFromOptions()
        self.pc.setUp()
        self.update(pc)

    def update(self, pc):
        return

    def apply(self, pc, X, Y):
        self.pc.apply(X, Y)

    def applyTranspose(self, pc, X, Y):
        raise NotImplementedError("applyTranspose not implemented for VerticallyLumpedPC")

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        viewer.pushASCIITab()
        viewer.printfASCII("Vertically lumped MG as in Kramer et al. doi:10.1016/j.ocemod.2010.08.001\n")
        self.pc.view(viewer)


class LaplacePC(AuxiliaryOperatorPC):
    """
    A preconditioner that uses the standard CG FEM Laplace operator as the preconditioner matrix to work with
    """
    _prefix = "laplace_"  # prefix for solver parameters

    def form(self, pc, test, trial):
        # returns the form from which the operator is assembled, and any bcs (None here)
        _, P = pc.getOperators()
        ctx = P.getPythonContext()
        dt = ctx.appctx['dt']
        dx = ctx.appctx['dx']
        return dt * dt * dot(grad(test), grad(trial))*dx, None
