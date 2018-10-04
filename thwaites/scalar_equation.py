from .equations import BaseTerm, BaseEquation
from firedrake import dot, div, conditional
from .utility import is_continuous

"""
This module contains the scalar terms and equations (e.g. for temperature and salinity transport)
"""


class ScalarAdvectionTerm(BaseTerm):
    r"""
    Scalar advection term (non-conservative): u \dot \div(q)
    """
    def residual(self, trial, trial_lagged, fields, bcs):
        u = fields['velocity']
        phi = self.test
        n = self.n
        q = trial

        F = q*div(phi*u)*self.dx

        for id, bc in bcs.items():
            if 'q' in bc:
                F -= conditional(dot(u, n) < 0,
                                 phi*dot(u, n)*bc['q'],
                                 phi*dot(u, n)*q) * self.ds(id)

        if not (is_continuous(trial) and is_continuous(u)):
            # this is the same trick as in the DG_advection firedrake demo
            un = 0.5*(dot(u, n) + abs(dot(u, n)))
            F -= (phi('+') - phi('-'))*(un('+')*q('+') - un('-')*q('-'))*self.dS

        return F


class ScalarAdvectionEquation(BaseEquation):
    """
    Scalar equation with only an advection term.
    """

    terms = [ScalarAdvectionTerm]
