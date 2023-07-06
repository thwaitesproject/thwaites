from .equations import BaseTerm, BaseEquation
from firedrake import dot, grad, as_vector
r"""
This module contains the terms and equations for the balance pressure solve

NOTE: because this is a time independent solve the residual() method returns the residual as it would appear
on the LHS of the equation i.e:

\sum term.residual()  = 0

"""


class BalancePressurePoisson(BaseTerm):
    r"""
    Balance pressure laplacian term: div grad p_b
    """

    def residual(self, test, trial, trial_lagged, fields, bcs):
        xi = test  # assume P2 field
        p_b = trial

        F = dot(grad(xi), grad(p_b)) * self.dx

        for id, bc in bcs.items():
            # for vertical side walls with melting and slope add a pressure gradient
            F -= xi * bc['gradpb'] * self.ds(id)

        return F


class DivergenceGeostrophicPoisson(BaseTerm):
    r"""
    Divergence of buoyancy and coriolis terms: div B
    """

    def residual(self, test, trial, trial_lagged, fields, bcs):
        xi = test  # assume P2 field
        B = fields['buoyancy']

        if 'coriolis_frequency' in fields:
            assert 'velocity' in fields

            u = fields['velocity']
            f = fields['coriolis_frequency']

            # coriolis term vector f-plane on LHS
            if self.dim == 3:
                c = as_vector((-f*u[1], f*u[0], 0))
            elif self.dim == 2:
                c = as_vector((f*u, 0))  # u is the scalar solution to the 2.5d evolution equations

            B = B - c  # combined buoyancy and coriolis vector on the RHS

        F = dot(grad(xi), B) * self.dx  # Still on RHS
        return -F   # move to LHS


class BalancePressureEquation(BaseEquation):
    """
    Balance pressure poisson equation
    """

    terms = [BalancePressurePoisson, DivergenceGeostrophicPoisson]
