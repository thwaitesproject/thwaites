from .time_stepper import TimeIntegratorBase
import firedrake


class CoupledTimeIntegrator(TimeIntegratorBase):
    def __init__(self, equations, solution, fields, coupling, dt, bcs=None, solver_parameters={}):
        """
        :arg equations: the equations to solve
        :type equations: list of :class:`BaseEquation` objects
        :arg solution: :class:`Function` of MixedFunctionSpace where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation (any functions that are not part of the solution)
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg coupling: for each equation a map (dict) from field name to number of the (different) equation whose trial function
                       should be used for that field name (see example below)
        :type coulings: list of dicts
        :arg float dt: time step in seconds
        :kwarg dict solver_parameters: PETSc solver options


        Example for coupling:
        Suppose we couple three equations:
        0) a scalar adv. diff. equation for a density
        1) a momentum equation to solve for velocity
        2) a contintuity equation with associated trial function of pressure
        Then if equation 0) uses velocity of 1), equation 1) uses pressure of 2) and
        density of 0), and equation 2) is a continuity constraint on velocity of 1), we get:
            coupling = [{'velocity': 1}, {'pressure': 2, 'density': 0}, {'velocity': 1}]
        """

        super(CoupledTimeIntegrator, self).__init__()

        self.equations = equations
        self.solution = solution
        self.fields = fields
        self.coupling = coupling
        self.dt = dt
        self.dt_const = firedrake.Constant(dt)
        self.bcs = bcs

        # unique identifier used in solver
        self.name = '-'.join([self.__class__.__name__] +
                             [eq.__class__.__name__ for eq in self.equations])

        self.solver_parameters = {}
        self.solver_parameters.update(solver_parameters)


class SaddlePointTimeIntegrator(CoupledTimeIntegrator):
    """A CoupledTimeIntegrator for 2 equations,

    where the first equation is integrated in time, and the second equation
    acts as a constraint to the solution of the first."""
    def __init__(self, equations, solution, fields, coupling, dt, bcs=None, solver_parameters={}):
        super().__init__(equations, solution, fields, coupling, dt, bcs=bcs, solver_parameters=solver_parameters)
        assert(len(equations) == 2)
        assert(len(solution.function_space().split()) == 2)
        # FIXME:
        # assert(equations[0].test[0].ufl_element() == equations[1].trial_space.ufl_element())
        # assert(equations[1].test.ufl_element() == equations[0].trial_space.ufl_element())


class CrankNicolsonSaddlePointTimeIntegrator(SaddlePointTimeIntegrator):
    def __init__(self, equations, solution, fields, coupling, dt, bcs=None, solver_parameters={}, theta=1.0):
        super().__init__(equations, solution, fields, coupling, dt, bcs=bcs, solver_parameters=solver_parameters)
        self.theta = firedrake.Constant(theta)

        self.solution_old = firedrake.Function(self.solution)
        self._initialized = False

    def initialize(self, init_solution):
        self.solution_old.assign(init_solution)
        u, p = firedrake.split(self.solution)
        u_old, p_old = self.solution_old.split()
        u_theta = (1-self.theta)*u_old + self.theta*u
        p_theta = (1-self.theta)*p_old + self.theta*p
        z_theta = [u_theta, p_theta]
        self._fields = []
        for i, cpl in enumerate(self.coupling):
            cfields = self.fields.copy()
            for field_name, eqno in cpl.items():
                assert 0 <= eqno <= 2 and not eqno == i
                cfields[field_name] = z_theta[eqno]
            self._fields.append(cfields)

        self.F = self.equations[0].mass_term(u-u_old)
        self.F -= self.dt_const*self.equations[0].residual(u_theta, u_theta, self._fields[0], bcs=self.bcs)
        self.F -= self.dt_const*self.equations[1].residual(p_theta, p_theta, self._fields[1], bcs=self.bcs)

        self.problem = firedrake.NonlinearVariationalProblem(self.F, self.solution)
        self.solver = firedrake.NonlinearVariationalSolver(self.problem,
                                                           solver_parameters=self.solver_parameters,
                                                           options_prefix=self.name)
        self._initialized = True

    def advance(self, t, update_forcings=None):
        if not self._initialized:
            self.initialize(self.solution)
        self.solution_old.assign(self.solution)
        self.solver.solve()
