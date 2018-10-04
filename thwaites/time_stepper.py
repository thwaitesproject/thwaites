from abc import ABC, abstractmethod, abstractproperty
import firedrake
import operator
import numpy as np

"""
Timestepper code, this is mostly copied from Thetis. At the moment explicit RK methods only.
"""


class TimeIntegratorBase(ABC):
    """
    Abstract class that defines the API for all time integrators

    Both :class:`TimeIntegrator` and :class:`CoupledTimeIntegrator` inherit
    from this class.
    """

    @abstractmethod
    def advance(self, t, update_forcings=None):
        """
        Advances equations for one time step

        :arg t: simulation time
        :type t: float
        :arg update_forcings: user-defined function that takes the simulation
            time and updates any time-dependent boundary conditions
        """
        pass

    @abstractmethod
    def initialize(self, init_solution):
        """
        Initialize the time integrator

        :arg init_solution: initial solution
        """
        pass


class TimeIntegrator(TimeIntegratorBase):
    """
    Base class for all time integrator objects that march a single equation
    """
    def __init__(self, equation, solution, fields, dt, solver_parameters={}):
        """
        :arg equation: the equation to solve
        :type equation: :class:`BaseEquation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :kwarg dict solver_parameters: PETSc solver options
        """
        super(TimeIntegrator, self).__init__()

        self.equation = equation
        self.solution = solution
        self.fields = fields
        self.dt = dt
        self.dt_const = firedrake.Constant(dt)

        # unique identifier used in solver
        self.name = '-'.join([self.__class__.__name__,
                              self.equation.__class__.__name__])

        self.solver_parameters = {}
        self.solver_parameters.update(solver_parameters)


class RungeKuttaTimeIntegrator(TimeIntegrator):
    """Abstract base class for all Runge-Kutta time integrators"""

    @abstractmethod
    def get_final_solution(self):
        """
        Evaluates the final solution
        """
        pass

    @abstractmethod
    def solve_stage(self, i_stage, t, update_forcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at right state
        corresponding to each sub-step.
        """
        pass

    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if not self._initialized:
            self.initialize(self.solution)
        for i in range(self.n_stages):
            self.solve_stage(i, t, update_forcings)
        self.get_final_solution()


class ERKGeneric(RungeKuttaTimeIntegrator):
    """
    Generic explicit Runge-Kutta time integrator.

    Implements the Butcher form. All terms in the equation are treated explicitly.
    """
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None,
                 solver_parameters={}):
        """
        :arg equation: the equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :kwarg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        :kwarg dict solver_parameters: PETSc solver options
        """
        super(ERKGeneric, self).__init__(equation, solution, fields, dt, solver_parameters)
        self._initialized = False
        V = solution.function_space()
        self.solution_old = firedrake.Function(V, name='old solution')

        self.tendency = []
        for i in range(self.n_stages):
            k = firedrake.Function(V, name='tendency{:}'.format(i))
            self.tendency.append(k)

        # fully explicit evaluation
        trial = firedrake.TrialFunction(V)
        self.a_rk = self.equation.mass_term(trial)
        self.l_rk = self.dt_const*self.equation.residual(self.solution, self.solution, self.fields, bnd_conditions)

        self._nontrivial = self.l_rk != 0

        # construct expressions for stage solutions
        if self._nontrivial:
            self.sol_expressions = []
            for i_stage in range(self.n_stages):
                sol_expr = sum(map(operator.mul, self.tendency[:i_stage], self.a[i_stage][:i_stage]))
                self.sol_expressions.append(sol_expr)
            self.final_sol_expr = sum(map(operator.mul, self.tendency, self.b))

        self.update_solver()

    def update_solver(self):
        if self._nontrivial:
            self.solver = []
            for i in range(self.n_stages):
                prob = firedrake.LinearVariationalProblem(self.a_rk, self.l_rk, self.tendency[i])
                solver = firedrake.LinearVariationalSolver(prob, options_prefix=self.name + '_k{:}'.format(i),
                                                           solver_parameters=self.solver_parameters)
                self.solver.append(solver)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        self._initialized = True

    def update_solution(self, i_stage):
        """
        Computes the solution of the i-th stage

        Tendencies must have been evaluated first.

        """
        self.solution.assign(self.solution_old)
        if self._nontrivial and i_stage > 0:
            self.solution += self.sol_expressions[i_stage]

    def solve_tendency(self, i_stage, t, update_forcings=None):
        """
        Evaluates the tendency of i-th stage
        """
        if self._nontrivial:
            if update_forcings is not None:
                update_forcings(t + self.c[i_stage]*self.dt)
            self.solver[i_stage].solve()

    def get_final_solution(self):
        """Assign final solution to :attr:`self.solution`
        """
        self.solution.assign(self.solution_old)
        if self._nontrivial:
            self.solution += self.final_sol_expr
        self.solution_old.assign(self.solution)

    def solve_stage(self, i_stage, t, update_forcings=None):
        """Solve i-th stage and assign solution to :attr:`self.solution`."""
        self.update_solution(i_stage)
        self.solve_tendency(i_stage, t, update_forcings)


class AbstractRKScheme(ABC):
    """
    Abstract class for defining Runge-Kutta schemes.

    Derived classes must define the Butcher tableau (arrays :attr:`a`, :attr:`b`,
    :attr:`c`) and the CFL number (:attr:`cfl_coeff`).

    Currently only explicit or diagonally implicit schemes are supported.
    """

    @abstractproperty
    def a(self):
        """Runge-Kutta matrix :math:`a_{i,j}` of the Butcher tableau"""
        pass

    @abstractproperty
    def b(self):
        """weights :math:`b_{i}` of the Butcher tableau"""
        pass

    @abstractproperty
    def c(self):
        """nodes :math:`c_{i}` of the Butcher tableau"""
        pass

    @abstractproperty
    def cfl_coeff(self):
        """
        CFL number of the scheme

        Value 1.0 corresponds to Forward Euler time step.
        """
        pass

    def __init__(self):
        super(AbstractRKScheme, self).__init__()
        self.a = np.array(self.a)
        self.b = np.array(self.b)
        self.c = np.array(self.c)

        assert not np.triu(self.a, 1).any(), 'Butcher tableau must be lower diagonal'
        assert np.allclose(np.sum(self.a, axis=1), self.c), 'Inconsistent Butcher tableau: Row sum of a is not c'

        self.n_stages = len(self.b)
        self.butcher = np.vstack((self.a, self.b))

        self.is_implicit = np.diag(self.a).any()
        self.is_dirk = np.diag(self.a).all()


class ForwardEulerAbstract(AbstractRKScheme):
    """
    Forward Euler method
    """
    a = [[0]]
    b = [1.0]
    c = [0]
    cfl_coeff = 1.0


class ERKLSPUM2Abstract(AbstractRKScheme):
    """
    ERKLSPUM2, 3-stage, 2nd order Explicit Runge Kutta method

    From IMEX RK scheme (17) in Higureras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """
    a = [[0, 0, 0],
         [5.0/6.0, 0, 0],
         [11.0/24.0, 11.0/24.0, 0]]
    b = [24.0/55.0, 1.0/5.0, 4.0/11.0]
    c = [0, 5.0/6.0, 11.0/12.0]
    cfl_coeff = 1.2


class ERKLPUM2Abstract(AbstractRKScheme):
    """
    ERKLPUM2, 3-stage, 2nd order
    Explicit Runge Kutta method

    From IMEX RK scheme (20) in Higureras et al. (2014).

    Higueras et al (2014). Optimized strong stability preserving IMEX
    Runge-Kutta methods. Journal of Computational and Applied Mathematics
    272(2014) 116-140. http://dx.doi.org/10.1016/j.cam.2014.05.011
    """
    a = [[0, 0, 0],
         [1.0/2.0, 0, 0],
         [1.0/2.0, 1.0/2.0, 0]]
    b = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    c = [0, 1.0/2.0, 1.0]
    cfl_coeff = 2.0


class ERKMidpointAbstract(AbstractRKScheme):
    a = [[0.0, 0.0],
         [0.5, 0.0]]
    b = [0.0, 1.0]
    c = [0.0, 0.5]
    cfl_coeff = 1.0


class SSPRK33Abstract(AbstractRKScheme):
    r"""
    3rd order Strong Stability Preserving Runge-Kutta scheme, SSP(3,3).

    This scheme has Butcher tableau

    .. math::
        \begin{array}{c|ccc}
            0 &                 \\
            1 & 1               \\
          1/2 & 1/4 & 1/4 &     \\ \hline
              & 1/6 & 1/6 & 2/3
        \end{array}

    CFL coefficient is 1.0
    """
    a = [[0, 0, 0],
         [1.0, 0, 0],
         [0.25, 0.25, 0]]
    b = [1.0/6.0, 1.0/6.0, 2.0/3.0]
    c = [0, 1.0, 0.5]
    cfl_coeff = 1.0


class ERKLSPUM2(ERKGeneric, ERKLSPUM2Abstract):
    pass


class ERKLPUM2(ERKGeneric, ERKLPUM2Abstract):
    pass


class ERKMidpoint(ERKGeneric, ERKMidpointAbstract):
    pass


class ERKEuler(ERKGeneric, ForwardEulerAbstract):
    pass


class SSPRK33(ERKGeneric, SSPRK33Abstract):
    pass
