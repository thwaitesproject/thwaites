from .equations import BaseTerm, BaseEquation
from firedrake import dot, inner, div, grad, conditional, CellDiameter, as_matrix, avg, jump, Constant
from .utility import is_continuous, normal_is_continuous
from ufl import tensors, algebra
"""
This module contains the scalar terms and equations (e.g. for temperature and salinity transport)

NOTE: for all terms, the residual() method returns the residual as it would be on the RHS of the equation, i.e.:

  dq/dt = \sum term.residual()

This sign-convention is for compatibility with Thetis' timeintegrators. In general, however we like to think about
the terms as they are on the LHS. Therefore in the residual methods below we assemble in F as it would be on the LHS:

  dq/dt + F(q) = 0

and at the very end "return -F".
"""


class ScalarAdvectionTerm(BaseTerm):
    r"""
    Scalar advection term (non-conservative): u \dot \div(q)
    """
    def residual(self, test, trial, trial_lagged, fields, bcs):
        u = fields['velocity']
        phi = test
        n = self.n
        q = trial

        F = -q*div(phi*u)*self.dx

        for id, bc in bcs.items():
            if 'q' in bc:
                F += conditional(dot(u, n) < 0,
                                 phi*dot(u, n)*bc['q'],
                                 phi*dot(u, n)*q) * self.ds(id)

        if not (is_continuous(self.trial_space) and normal_is_continuous(u)):
            # this is the same trick as in the DG_advection firedrake demo
            un = 0.5*(dot(u, n) + abs(dot(u, n)))
            F += (phi('+') - phi('-'))*(un('+')*q('+') - un('-')*q('-'))*self.dS

        return -F


class ScalarDiffusionTerm(BaseTerm):
    r"""
    Diffusion term :math:`-\nabla \cdot (\kappa \nabla q)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        -\int_\Omega \nabla \cdot (\kappa \nabla q) \phi dx
        =& \int_\Omega \kappa (\nabla \phi) \cdot (\nabla q) dx \\
        &- \int_{\mathcal{I}\cup\mathcal{I}_v} \text{jump}(\phi \textbf{n})
        \cdot \text{avg}(\kappa \nabla q) dS
        - \int_{\mathcal{I}\cup\mathcal{I}_v} \text{jump}(q \textbf{n})
        \cdot \text{avg}(\kappa  \nabla \phi) dS \\
        &+ \int_{\mathcal{I}\cup\mathcal{I}_v} \sigma \text{avg}(\kappa) \text{jump}(q \textbf{n}) \cdot
            \text{jump}(\phi \textbf{n}) dS

    where :math:`\sigma` is a penalty parameter,
    see Epshteyn and Riviere (2007).

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029

    """
    def residual(self, test, trial, trial_lagged, fields, bcs):
        kappa = fields['diffusivity']
        if kappa.__class__ == Constant:
            diff_tensor = as_matrix([[kappa, 0, ],
                                     [0, kappa, ]])
        if kappa.__class__ == algebra.Sum:
            diff_tensor = as_matrix([[kappa, 0, ],
                                     [0, kappa, ]])
        elif kappa.__class__ == tensors.ListTensor:
            diff_tensor = kappa  # predefine matrix as above with different horizontal and vertical diffusivities
        else:
            raise Exception(str(kappa.__class__)+"is not a valid assigment. Should be Matrix or Constant.")

        phi = test
        n = self.n
        cellsize = CellDiameter(self.mesh)
        q = trial


        grad_test = grad(phi)
        diff_flux = dot(diff_tensor, grad(q))

        F = 0
        F += inner(grad_test, diff_flux)*self.dx

        # Interior Penalty method by
        # Epshteyn (2007) doi:10.1016/j.cam.2006.08.029
        # sigma = 3*k_max**2/k_min*p*(p+1)*cot(Theta)
        # k_max/k_min  - max/min diffusivity
        # p            - polynomial degree
        # Theta        - min angle of triangles
        # assuming k_max/k_min=2, Theta=pi/3
        # sigma = 6.93 = 3.5*p*(p+1)

        degree = phi.ufl_element().degree()
        sigma = 5.0*degree*(degree + 1)/cellsize
        if degree == 0:
            sigma = 1.5 / cellsize

        if not is_continuous(self.trial_space):
            F += avg(sigma)*inner(jump(phi, n), dot(avg(diff_tensor), jump(q, n)))*self.dS
            F += -inner(avg(dot(diff_tensor, grad(phi))), jump(q, n))*self.dS
            F += -inner(jump(phi, n), avg(dot(diff_tensor, grad(q))))*self.dS

        for id, bc in bcs.items():
            if 'q' in bc:
                jump_q = q-bc['q']
                # this corresponds to the same 3 terms as the dS integrals for DG above:
                F += sigma*phi*inner(n, dot(diff_tensor, n))*jump_q*self.ds(id)
                F += -inner(dot(diff_tensor, grad(phi)), n)*jump_q*self.ds(id)
                F += -inner(phi*n, dot(diff_tensor, grad(q))) * self.ds(id)
                if 'flux' in bc:
                    raise ValueError("Cannot apply both `q` and `flux` bc on same boundary")
            elif 'flux' in bc:
                # here we need only the third term, because we assume jump_q=0 (q_ext=q)
                # the provided flux = kappa dq/dn = dot(n, dot(diff_tensor, grad(q))
                F += -phi*bc['flux']*self.ds(id)

        return -F

class ScalarSourceTerm(BaseTerm):
    r"""
        Source term :math:`s_T`
    """

    def residual(self, trial, trial_lagged, fields, bcs):
        if 'source' not in fields:
            return 0

        phi = self.test
        source = fields['source']

        # NOTE, here source term F is already on the RHS
        F = dot(phi, source)*self.dx

        return F


class ScalarAbsorptionTerm(BaseTerm):
    r"""
            Absorption Term :math:`\alpha_T T`
        """

    def residual(self, trial, trial_lagged, fields, bcs):
        if 'absorption coefficient' not in fields:
            return 0

        phi = self.test
        alpha = fields['absorption coefficient']

        # NOTE, here absorption term F is already on the RHS
        # implement absorption term implicitly at current time step.
        F = -dot(phi, alpha*trial)*self.dx

        return F








class ScalarAdvectionEquation(BaseEquation):
    """
    Scalar equation with only an advection term.
    """

    terms = [ScalarAdvectionTerm]


class ScalarAdvectionDiffusionEquation(BaseEquation):
    """
    Scalar equation with advection and diffusion.
    """

    terms = [ScalarAdvectionTerm, ScalarDiffusionTerm]
