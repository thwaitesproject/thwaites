from .time_stepper import TimeIntegratorBase
from .momentum_equation import PressureGradientTerm, DivergenceTerm
import firedrake
from pyop2.profiling import timed_stage

class BalancePressureSolver(TimeIntegrator):
    """
    Solver for balance pressure equation, div grad p_b = div B 

    e.g. see J.R. Maddison, D.P. Marshall, C.C. Pain, M.D. Piggott, Accurate representation of geostrophic and hydrostatic balance in unstructured mesh finite element ocean modelling, Ocean Modelling, Volume   9, Issues 3–4, 2011,
    """
    def __init__(self, equation, solution, fields, bcs=None, solver_parameters={}, strong_bcs=None):
        """
        :arg equation: the equation to solve
        :type equation: :class:`BaseEquation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation (any functions that are not part of the solution)
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :kwarg dict solver_parameters: PETSc solver options


        """

        super().__init__(equation, solution, fields, 0, solver_parameters) # note dt set to zero as the balance pressure solve is time independent


        self.solution_old = firedrake.Function(self.solution)
        self._initialized = False
        self.strong_bcs = strong_bcs

    def initialize(self, init_solution):
        self.solution_old.assign(init_solution)

        self.F = self.equation.residual(self.test, self.solution, self.solution_old, self.fields, bcs=self.bcs)  

        self.problem = firedrake.NonlinearVariationalProblem(self.F, self.solution, self.strong_bcs)
        self.solver = firedrake.NonlinearVariationalSolver(self.problem,
                                                           solver_parameters=self.solver_parameters,
                                                           options_prefix=self.name)
        self._initialized = True

    def advance(self, t, update_forcings=None):
        if not self._initialized:
            self.initialize(self.solution)
        self.solution_old.assign(self.solution)
        self.solver.solve()


