"""Physical rotating momentum and independent monolithic theta reference."""
from firedrake import (
    Function, NonlinearVariationalProblem, NonlinearVariationalSolver,
    TestFunctions, split,
)
from thwaites.equations import BaseEquation
from thwaites.momentum_equation import CoriolisTerm, PressureGradientTerm
from free_surface_helpers import DIRECT_SOLVER_PARAMETERS


class RotatingMomentumEquation(BaseEquation):
    """Only physical pressure and Coriolis; no balancing force."""

    terms = [PressureGradientTerm, CoriolisTerm]


class CoupledThetaReference:
    """Independent monolithic weak theta equations, with direct LU.

    No predictor, pressure increment, lagged pressure or projection call.
    Only the physical equation residuals are shared with production.
    """

    def __init__(self, c, state, dt, theta):
        self.old = Function(c.W)
        self.state = state
        u, p = split(state)
        un, pn = split(self.old)
        v, q = TestFunctions(c.W)
        ut = (1-theta)*un+theta*u
        pt = (1-theta)*pn+theta*p
        F = c.momentum.mass_term(v, u-un)-dt*c.momentum.residual(
            v, ut, ut, {**c.fields, "pressure": pt}, c.bcs,
        )
        F -= dt*c.continuity.residual(q, p, p, {
            **c.fields, "velocity": ut, "old_pressure": pn,
            "dt": dt, "theta": theta,
        }, c.bcs)
        self.form = F
        self.solver = NonlinearVariationalSolver(
            NonlinearVariationalProblem(F, state),
            solver_parameters=DIRECT_SOLVER_PARAMETERS,
            options_prefix="rotating_coupled_reference",
        )

    def advance(self, time):
        self.old.assign(self.state)
        self.solver.solve()
