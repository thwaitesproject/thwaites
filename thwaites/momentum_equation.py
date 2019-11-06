from .equations import BaseTerm, BaseEquation
from firedrake import dot, inner, outer, transpose, div, grad, nabla_grad, conditional, Constant
from firedrake import CellDiameter, avg, Identity, zero
from .utility import is_continuous, normal_is_continuous, tensor_jump
from ufl import tensors, algebra
#import numpy as np
"""
This module contains the classes for the momentum equation and its terms.

NOTE: for all terms, the residual() method returns the residual as it would be on the RHS of the equation, i.e.:

  dq/dt = \sum term.residual()

This sign-convention is for compatibility with Thetis' timeintegrators. In general, however we like to think about
the terms as they are on the LHS. Therefore in the residual methods below we assemble in F as it would be on the LHS:

  dq/dt + F(q) = 0

and at the very end "return -F".
"""


class MomentumAdvectionTerm(BaseTerm):
    r"""
    Momentum advection term (non-conservative): u \dot \grad(u)
    """
    def residual(self, test, trial, trial_lagged, fields, bcs):
        u_adv = trial_lagged
        phi = test
        n = self.n
        u = trial

        F = -dot(u, div(outer(phi, u_adv)))*self.dx

        for id, bc in bcs.items():
            if 'u' in bc:
                u_in = bc['u']
            elif 'un' in bc:
                u_in = bc['un'] * n  # this implies u_t=0 on the inflow
            else:
                u_in = zero(self.dim)
            F += conditional(dot(u_adv, n) < 0,
                             dot(phi, u_in)*dot(u_adv, n),
                             dot(phi, u)*dot(u_adv, n)) * self.ds(id)

        if not (is_continuous(self.trial_space) and normal_is_continuous(u_adv)):
            # this is the same trick as in the DG_advection firedrake demo
            un = 0.5*(dot(u_adv, n) + abs(dot(u_adv, n)))
            F += dot(phi('+') - phi('-'), un('+')*u('+') - un('-')*u('-'))*self.dS

        return -F


class ViscosityTerm(BaseTerm):

    # hard-coded for now
    symmetric_stress = True

    r"""
    Viscosity term :math:`-\nabla \cdot (\mu \nabla u)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        -\int_\Omega \nabla \cdot (\mu \nabla u) \phi dx
        =& \int_\Omega \mu (\nabla \phi) \cdot (\nabla u) dx \\
        &- \int_{\mathcal{I}\cup\mathcal{I}_v} \text{jump}(\phi \textbf{n})
        \cdot \text{avg}(\mu \nabla u) dS
        - \int_{\mathcal{I}\cup\mathcal{I}_v} \text{jump}(u \textbf{n})
        \cdot \text{avg}(\mu  \nabla \phi) dS \\
        &+ \int_{\mathcal{I}\cup\mathcal{I}_v} \sigma \text{avg}(\mu) \text{jump}(u \textbf{n}) \cdot
            \text{jump}(\phi \textbf{n}) dS

    where :math:`\sigma` is a penalty parameter,
    see Epshteyn and Riviere (2007).

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029

    """
    def residual(self, test, trial, trial_lagged, fields, bcs):
        mu = fields['viscosity']
        if len(mu.ufl_shape) == 2:
            diff_tensor = mu
        else:
            diff_tensor = mu * Identity(self.dim)
        phi = test
        n = self.n
        cellsize = CellDiameter(self.mesh)
        u = trial
        u_lagged = trial_lagged


        grad_test = nabla_grad(phi)
        stress = dot(diff_tensor, nabla_grad(u))
        if self.symmetric_stress:
            stress += dot(diff_tensor, grad(u))

        F = 0
        F += inner(grad_test, stress)*self.dx

        # Interior Penalty method by
        # Epshteyn (2007) doi:10.1016/j.cam.2006.08.029
        # sigma = 3*k_max**2/k_min*p*(p+1)*cot(Theta)
        # k_max/k_min  - max/min diffusivity
        # p            - polynomial degree
        # Theta        - min angle of triangles
        # assuming k_max/k_min=2, Theta=pi/3
        # sigma = 6.93 = 3.5*p*(p+1)

        degree = self.trial_space.ufl_element().degree()
        alpha = fields.get('interior_penalty', 5.0)
        sigma = alpha*degree*(degree + 1)/cellsize
        if degree == 0:
            sigma = 1.5 / cellsize

        if not is_continuous(self.trial_space):
            u_tensor_jump = tensor_jump(n, u)
            if self.symmetric_stress:
                u_tensor_jump += transpose(u_tensor_jump)
            F += avg(sigma)*inner(tensor_jump(n, phi), dot(avg(diff_tensor), u_tensor_jump))*self.dS
            F += -inner(avg(dot(diff_tensor, nabla_grad(phi))), u_tensor_jump)*self.dS
            F += -inner(tensor_jump(n, phi), avg(stress))*self.dS

        for id, bc in bcs.items():
            if 'u' in bc or 'un' in bc:
                if 'u' in bc:
                    u_tensor_jump = outer(n,u-bc['u'])
                else:
                    u_tensor_jump = outer(n, n)*(dot(n, u)-bc['un'])
                if self.symmetric_stress:
                    u_tensor_jump += transpose(u_tensor_jump)
                # this corresponds to the same 3 terms as the dS integrals for DG above:
                F += sigma*inner(outer(n, phi), dot(diff_tensor, u_tensor_jump))*self.ds(id)
                F += -inner(dot(diff_tensor, grad(phi)), u_tensor_jump)*self.ds(id)
                if 'u' in bc:
                    F += -inner(outer(n,phi), stress) * self.ds(id)
                elif 'un' in bc:
                    # we only keep, the normal part of stress, the tangential
                    # part is assumed to be zero stress (i.e. free slip), or prescribed via 'stress'
                    F += -dot(n, phi)*dot(n, dot(stress, n)) * self.ds(id)
            if 'stress' in bc:  # a momentum flux, a.k.a. "force"
                # here we need only the third term, because we assume jump_u=0 (u_ext=u)
                # the provided stress = n.(mu.stress_tensor)
                #F += -phi*bc['stress']*self.ds(id)
                F += dot(-phi, bc['stress']) * self.ds(id)
            if 'drag' in bc:  # (bottom) drag of the form tau = -C_D u |u|
                C_D = bc['drag']
                unorm = pow(dot(u_lagged, u_lagged),0.5)

                F += dot(-phi, -C_D*unorm*u) * self.ds(id)


            # NOTE 1: unspecified boundaries are equivalent to free stress (i.e. free in all directions)
            # NOTE 2: 'un' can be combined with 'stress' provided the stress force is tangential (e.g. no-normal flow with wind)

            if 'u' in bc and 'stress' in bc:
                raise ValueError("Cannot apply both 'u' and 'stress' bc on same boundary")
            if 'u' in bc and 'drag' in bc:
                raise ValueError("Cannot apply both 'u' and 'drag' bc on same boundary")
            if 'u' in bc and 'un' in bc:
                raise ValueError("Cannot apply both 'u' and 'un' bc on same boundary")


        return -F


class PressureGradientTerm(BaseTerm):
    def residual(self, test, trial, trial_lagged, fields, bcs):
        phi = test
        n = self.n
        p = fields['pressure']

        # NOTE: we assume p is continuous

        F = dot(phi, grad(p))*self.dx

        # do nothing should be zero (normal) stress:
        F += -dot(phi, n)*p*self.ds

        # for those boundaries where the normal component of u is specified
        # we take it out again
        for id, bc in bcs.items():
            if 'u' in bc or 'un' in bc:
                F += dot(phi, n)*p*self.ds(id)

        return -F


class DivergenceTerm(BaseTerm):
    def residual(self, test, trial, trial_lagged, fields, bcs):
        psi = test
        n = self.n
        u = fields['velocity']

        # NOTE: we assume psi is continuous
        # assert is_continuous(psi)
        F = -dot(grad(psi), u)*self.dx

        # do nothing should be zero (normal) stress, which means no (Dirichlet condition)
        # should be imposed on the normal component
        F += psi*dot(n, u)*self.ds

        # for those boundaries where the normal component of u is specified
        # we take it out again and replace with the specified un
        for id, bc in bcs.items():
            if 'u' in bc:
                F += psi*dot(n, bc['u']-u)*self.ds(id)
            elif 'un' in bc:
                F += psi*(bc['un'] - dot(n, u))*self.ds(id)

        return -F


class MomentumSourceTerm(BaseTerm):
    def residual(self, test, trial, trial_lagged, fields, bcs):
        if 'source' not in fields:
            return 0

        phi = test
        source = fields['source']

        # NOTE, here source term F is already on the RHS
        F = dot(phi, source)*self.dx

        return F


class CoriolisTerm(BaseTerm):
    def residual(self, test, trial, trial_lagged, fields, bcs):
        if 'coriolis_frequency' not in fields:
            return 0
        phi = test
        f = fields['coriolis_frequency']

        F = (-f*trial[1]*test[0] + f*trial[0]*phi[1])*self.dx
        return -F


class MomentumEquation(BaseEquation):
    """
    Momentum equation with advection, viscosity, pressure gradient, source term, and coriolis.
    """

    terms = [MomentumAdvectionTerm, ViscosityTerm, PressureGradientTerm, MomentumSourceTerm, CoriolisTerm]


class ContinuityEquation(BaseEquation):
    """
    Continuity equation: div(u) = 0
    """

    terms = [DivergenceTerm]
