from .equations import BaseTerm, BaseEquation
from firedrake import dot, inner, outer, transpose, div, grad, nabla_grad, conditional, as_tensor, sign
from firedrake import avg, Identity, zero
from .utility import is_continuous, normal_is_continuous, tensor_jump, cell_edge_integral_ratio
from firedrake import FacetArea, CellVolume
r"""
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
            # s=0: u.n(-)<0  =>  flow goes from '+' to '-' => '+' is upwind
            # s=1: u.n(-)>0  =>  flow goes from '-' to '+' => '-' is upwind
            s = 0.5*(sign(dot(avg(u), n('-'))) + 1.0)
            u_up = u('-')*s + u('+')*(1-s)
            F += dot(u_up, (dot(u_adv('+'), n('+'))*phi('+') + dot(u_adv('-'), n('-'))*phi('-'))) * self.dS

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

        if 'background_viscosity' in fields:
            assert 'grid_resolution' in fields
            mu_background = fields['background_viscosity']
            grid_dx = fields['grid_resolution'][0]
            grid_dz = fields['grid_resolution'][1]
            mu_h = 0.5*abs(trial[0]) * grid_dx + mu_background
            mu_v = 0.5*abs(trial[1]) * grid_dz + mu_background
            print("use redx viscosity")
            diff_tensor = as_tensor([[mu_h, 0], [0, mu_v]])

        else:
            mu = fields['viscosity']
            if len(mu.ufl_shape) == 2:
                diff_tensor = mu
            else:
                diff_tensor = mu * Identity(self.dim)
        phi = test
        n = self.n
        u = trial
        u_lagged = trial_lagged

        grad_test = nabla_grad(phi)
        stress = dot(diff_tensor, nabla_grad(u))
        if self.symmetric_stress:
            stress += dot(diff_tensor, grad(u))

        F = 0
        F += inner(grad_test, stress)*self.dx

        # Interior Penalty method
        #
        # see https://www.researchgate.net/publication/260085826 for details
        # on the choice of sigma

        degree = self.trial_space.ufl_element().degree()
        if not isinstance(degree, int):
            degree = max(degree[0], degree[1])
        # safety factor: 1.0 is theoretical minimum
        alpha = fields.get('interior_penalty', 2.0)
        if degree == 0:
            # probably only works for orthog. quads and hexes
            sigma = 1.0
        else:
            nf = self.mesh.ufl_cell().num_facets()
            family = self.trial_space.ufl_element().family()
            if family in ['DQ', 'TensorProductElement', 'EnrichedElement']:
                degree_gradient = degree
            else:
                degree_gradient = degree - 1
            sigma = alpha * cell_edge_integral_ratio(self.mesh, degree_gradient) * nf
        # we use (3.23) + (3.20) from https://www.researchgate.net/publication/260085826
        # instead of maximum over two adjacent cells + and -, we just sum (which is 2*avg())
        # and the for internal facets we have an extra 0.5:
        # WEIRDNESS: avg(1/CellVolume(mesh)) crashes TSFC - whereas it works in scalar diffusion! - instead just writing out explicitly
        sigma *= FacetArea(self.mesh)*(1/CellVolume(self.mesh)('-') + 1/CellVolume(self.mesh)('+'))/2

        if not is_continuous(self.trial_space):
            u_tensor_jump = tensor_jump(n, u)
            if self.symmetric_stress:
                u_tensor_jump += transpose(u_tensor_jump)
            F += sigma*inner(tensor_jump(n, phi), dot(avg(diff_tensor), u_tensor_jump))*self.dS
            F += -inner(avg(dot(diff_tensor, nabla_grad(phi))), u_tensor_jump)*self.dS
            F += -inner(tensor_jump(n, phi), avg(stress))*self.dS

        for id, bc in bcs.items():
            if 'u' in bc or 'un' in bc:
                if 'u' in bc:
                    u_tensor_jump = outer(n, u-bc['u'])
                else:
                    u_tensor_jump = outer(n, n)*(dot(n, u)-bc['un'])
                if self.symmetric_stress:
                    u_tensor_jump += transpose(u_tensor_jump)
                # this corresponds to the same 3 terms as the dS integrals for DG above:
                F += 2*sigma*inner(outer(n, phi), dot(diff_tensor, u_tensor_jump))*self.ds(id)
                F += -inner(dot(diff_tensor, nabla_grad(phi)), u_tensor_jump)*self.ds(id)
                if 'u' in bc:
                    F += -inner(outer(n, phi), stress) * self.ds(id)
                elif 'un' in bc:
                    # we only keep, the normal part of stress, the tangential
                    # part is assumed to be zero stress (i.e. free slip), or prescribed via 'stress'
                    F += -dot(n, phi)*dot(n, dot(stress, n)) * self.ds(id)
            if 'stress' in bc:  # a momentum flux, a.k.a. "force"
                # here we need only the third term, because we assume jump_u=0 (u_ext=u)
                # the provided stress = n.(mu.stress_tensor)
                F += dot(-phi, bc['stress']) * self.ds(id)
            if 'drag' in bc:  # (bottom) drag of the form tau = -C_D u |u|
                C_D = bc['drag']
                if 'coriolis_frequency' in fields and self.dim == 2:
                    assert 'u_velocity' in fields
                    u_vel_component = fields['u_velocity']
                    unorm = pow(dot(u_lagged, u_lagged) + pow(u_vel_component, 2) + 1e-6, 0.5)
                else:
                    unorm = pow(dot(u_lagged, u_lagged) + 1e-6, 0.5)

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

class BalancePressureGradientTerm(BaseTerm):
    def residual(self, test, trial, trial_lagged, fields, bcs):
        phi = test
        if 'balance_pressure' in fields:
            p_b = fields['balance_pressure']
            # NOTE: we assume p is continuous
            F = dot(phi, grad(p_b))*self.dx
            print("hello pb")
            return -F
        else:
            return 0.0


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
        if 'coriolis_frequency' in fields:
            if self.dim == 3:
                phi = test
                f = fields['coriolis_frequency']
                F = (-f*trial[1]*test[0] + f*trial[0]*phi[1])*self.dx
                return -F
            elif self.dim == 2:
                assert 'u_velocity' in fields
                u = fields['u_velocity']
                f = fields['coriolis_frequency']
                F = f*u*test[0]*self.dx
                return -F
        else:
            return 0.0


class MomentumEquation(BaseEquation):
    """
    Momentum equation with advection, viscosity, pressure gradient, source term, and coriolis.
    """

    terms = [MomentumAdvectionTerm, ViscosityTerm, PressureGradientTerm, MomentumSourceTerm, CoriolisTerm, BalancePressureGradientTerm]


class ContinuityEquation(BaseEquation):
    """
    Continuity equation: div(u) = 0
    """

    terms = [DivergenceTerm]
