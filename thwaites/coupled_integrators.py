from .time_stepper import TimeIntegratorBase
from .momentum_equation import PressureGradientTerm, DivergenceTerm
import firedrake
from pyop2.profiling import timed_stage

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
        self.test = firedrake.TestFunctions(solution.function_space())
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


class CrankNicolsonSaddlePointTimeIntegrator(SaddlePointTimeIntegrator):
    def __init__(self, equations, solution, fields, coupling, dt, bcs=None,
                 solver_parameters={}, theta=1.0, strong_bcs=None):
        super().__init__(equations, solution, fields, coupling, dt, bcs=bcs, solver_parameters=solver_parameters)
        self.theta = firedrake.Constant(theta)

        self.solution_old = firedrake.Function(self.solution)
        self._initialized = False
        self.strong_bcs = strong_bcs

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

        self.F = self.equations[0].mass_term(self.test[0], u-u_old)
        #self.F -= self.dt_const*self.equations[0].residual(self.test[0], u_theta, u_theta, self._fields[0], bcs=self.bcs)
        self.F -= self.dt_const * self.equations[0].residual(self.test[0], u_theta, u_old, self._fields[0],
                                                             bcs=self.bcs)  # linearise using u_old
        self.F -= self.dt_const*self.equations[1].residual(self.test[1], p_theta, p_theta, self._fields[1], bcs=self.bcs)

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


class PressureProjectionTimeIntegrator(SaddlePointTimeIntegrator):
    def __init__(self, equations, solution, fields, coupling, dt, bcs=None, solver_parameters={}, theta=1.0,
                 predictor_solver_parameters={}, picard_iterations=1, pressure_nullspace=None):
        super().__init__(equations, solution, fields, coupling, dt, bcs=bcs, solver_parameters=solver_parameters)
        self.theta = firedrake.Constant(theta)
        self.theta_p = 1 # should not be used for now - maybe revisit with free surface terms
        self.predictor_solver_parameters = predictor_solver_parameters
        self.picard_iterations = picard_iterations
        self.pressure_nullspace = pressure_nullspace

        self.solution_old = firedrake.Function(self.solution)
        self.solution_lag = firedrake.Function(self.solution)
        self.u_test, self.p_test = firedrake.TestFunctions(self.solution.function_space())

        # the predictor space is the same as the first sub-space of the solution space, but indexed independently
        mesh = self.solution.function_space().mesh()
        self.u_space = firedrake.FunctionSpace(mesh, self.solution.split()[0].ufl_element())
        self.u_star_test = firedrake.TestFunction(self.u_space)
        self.u_star = firedrake.Function(self.u_space)

        self._initialized = False

    def initialize(self, init_solution):
        self.solution_old.assign(init_solution)
        u, p = firedrake.split(self.solution)
        u_old, p_old = self.solution_old.split()
        u_star_theta = (1-self.theta)*u_old + self.theta*self.u_star
        u_theta = (1-self.theta)*u_old + self.theta*u
        p_theta = (1-self.theta_p)*p_old + self.theta_p*p
        u_lag, p_lag = self.solution_lag.split()
        u_lag_theta = (1-self.theta)*u_old + self.theta*u_lag
        p_lag_theta = (1-self.theta_p)*p_old + self.theta_p*p_lag

        # setup predictor solve, this solves for u_start only using a fixed p_lag_theta for pressure
        self.fields_star = self.fields.copy()
        self.fields_star['pressure'] =  p_lag_theta

        self.Fstar = self.equations[0].mass_term(self.u_star_test, self.u_star-u_old)
        self.Fstar -= self.dt_const*self.equations[0].residual(self.u_star_test, u_star_theta, u_lag_theta, self.fields_star, bcs=self.bcs)
        self.predictor_problem = firedrake.NonlinearVariationalProblem(self.Fstar, self.u_star)
        self.predictor_solver = firedrake.NonlinearVariationalSolver(self.predictor_problem,
                                                                     solver_parameters=self.predictor_solver_parameters,
                                                                     options_prefix='predictor_momentum')

        # the correction solve, solving the coupled system:
        #   u1 = u* - dt*G ( p_theta - p_lag_theta)
        #   div(u1) = 0
        self.F = self.equations[0].mass_term(self.u_test, u-self.u_star)

        pg_term = [term for term in self.equations[0]._terms if isinstance(term, PressureGradientTerm)][0]
        pg_fields = self.fields.copy()
        # note that p_theta-p_lag_theta = theta_p*(p1-p_lag)
        pg_fields['pressure'] = self.theta * (p - p_lag)
        self.F -= self.dt_const*pg_term.residual(self.u_test, u_theta, u_lag_theta, pg_fields, bcs=self.bcs)

        div_term = [term for term in self.equations[1]._terms if isinstance(term, DivergenceTerm)][0]
        div_fields = self.fields.copy()
        div_fields['velocity'] = u
        self.F -= self.dt_const*div_term.residual(self.p_test, p_theta, p_lag_theta, div_fields, bcs=self.bcs)

        W = self.solution.function_space()
        mixed_nullspace = firedrake.MixedVectorSpaceBasis(W, [W.sub(0), self.pressure_nullspace])

        self.problem = firedrake.NonlinearVariationalProblem(self.F, self.solution)
        self.solver = firedrake.NonlinearVariationalSolver(self.problem,
                                                           solver_parameters=self.solver_parameters,
                                                           appctx = {'a': firedrake.derivative(self.F, self.solution),
                                                                     'schur_nullspace': self.pressure_nullspace,
                                                                     'dt': self.dt_const, 'dx': self.equations[1].dx},
                                                           nullspace = mixed_nullspace, transpose_nullspace = mixed_nullspace,
                                                           options_prefix=self.name)

        self._initialized = True

    def initialize_pressure(self):
        """Perform pseudo timestep to establish good initial pressure."""
        if not self._initialized:
            self.initialize(self.solution)
        u, p = self.solution.split()
        u_old, p_old = self.solution_old.split()
        # solve predictor step, but now fully explicit
        Fstar = self.equations[0].mass_term(self.u_star_test, self.u_star-u_old)
        Fstar -= self.dt_const*self.equations[0].residual(self.u_star_test, u_old, u_old, self.fields_star, bcs=self.bcs)
        # as an explicit solve this is now a trivial mass matrix solve, but let's just borrow the solver parameters of the normal predictor solve
        firedrake.solve(Fstar==0, self.u_star, solver_parameters=self.predictor_solver_parameters,
                options_prefix='predictor_momentum_initialise_pressure')
        # now do the usual pressure projection step
        # note that the combination of these two is equivalent with solving the fully coupled
        # system but with all momentum terms handled explicitly on the rhs
        self.solver.solve()
        # reset velocity to its initial value:
        u.assign(u_old)
        self.u_star.assign(u_old)

    def advance(self, t, update_forcings=None):
        if not self._initialized:
            self.initialize(self.solution)
        self.solution_old.assign(self.solution)
        for i in range(self.picard_iterations):
            self.picard_step()

    def picard_step(self):
        self.solution_lag.assign(self.solution)
        # solve for self.u_star
        with timed_stage("momentum_solve"):
            self.predictor_solver.solve()
        # pressure correction solve, solves for final solution (corrected velocity and pressure)
        with timed_stage("correction_solve"):
            self.solver.solve()

class CoupledEquationsTimeIntegrator(CoupledTimeIntegrator):
    def __init__(self, equations, solution, fields, dt, bcs=None, mass_terms=None, solver_parameters={}, strong_bcs=None):
        super().__init__(equations, solution, fields, {}, dt, bcs, solver_parameters)

        if mass_terms is None:
            self.mass_terms = [True]*len(equations)
        else:
            self.mass_terms = mass_terms

        self.strong_bcs = strong_bcs
        self.theta = 1.0
        self.solution_old = firedrake.Function(self.solution)
        self._initialized = False

    def initialize(self, init_solution):
        self.solution_old.assign(init_solution)
        z_theta = (1-self.theta)*self.solution_old + self.theta*self.solution

        self._fields = []
        for fields in self.fields:
            cfields = fields.copy()
            for field_name, field_expr in fields.items():
                if isinstance(field_expr, float):
                    continue
                cfields[field_name] = firedrake.replace(field_expr, {self.solution: z_theta})
            self._fields.append(cfields)

        F = 0
        for test, u, u_old, eq, mass_term, fields, bcs in zip(self.test, firedrake.split(self.solution), firedrake.split(self.solution_old), self.equations, self.mass_terms, self._fields, self.bcs):
            if mass_term:
                F += eq.mass_term(test, u-u_old)

            u_theta = (1-self.theta)*u_old + self.theta*u
            F -= self.dt_const * eq.residual(test, u_theta, u_theta, fields, bcs=bcs)

        self.problem = firedrake.NonlinearVariationalProblem(F, self.solution, bcs=self.strong_bcs)
        self.solver = firedrake.NonlinearVariationalSolver(self.problem,
                                                           solver_parameters=self.solver_parameters,
                                                           options_prefix=self.name)
        self._initialized = True

    def advance(self, t, update_forcings=None):
        if not self._initialized:
            self.initialize(self.solution)
        self.solution_old.assign(self.solution)
        self.solver.solve()
