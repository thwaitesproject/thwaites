from .equations import BaseTerm, BaseEquation
from firedrake import dot, inner, div, grad, conditional, CellDiameter, as_matrix, avg, jump
from .utility import is_continuous, normal_is_continuous

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
    def residual(self, trial, trial_lagged, fields, bcs):
        u = fields['velocity']
        phi = self.test
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
    Horizontal diffusion term :math:`-\nabla_h \cdot (\mu_h \nabla_h T)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        -\int_\Omega \nabla_h \cdot (\mu_h \nabla_h T) \phi dx
        =& \int_\Omega \mu_h (\nabla_h \phi) \cdot (\nabla_h T) dx \\
        &- \int_{\mathcal{I}_h\cup\mathcal{I}_v} \text{jump}(\phi \textbf{n}_h)
        \cdot \text{avg}(\mu_h \nabla_h T) dS
        - \int_{\mathcal{I}_h\cup\mathcal{I}_v} \text{jump}(T \textbf{n}_h)
        \cdot \text{avg}(\mu_h  \nabla \phi) dS \\
        &+ \int_{\mathcal{I}_h\cup\mathcal{I}_v} \sigma \text{avg}(\mu_h) \text{jump}(T \textbf{n}_h) \cdot
            \text{jump}(\phi \textbf{n}_h) dS

    where :math:`\sigma` is a penalty parameter,
    see Epshteyn and Riviere (2007).

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029

    """
    def residual(self, trial, trial_lagged, fields, bcs):
        kappa = fields['diffusivity']
        phi = self.test
        n = self.n
        cellsize = CellDiameter(self.mesh)
        q = trial

        diff_tensor = as_matrix([[kappa, 0, ],
                                 [0, kappa, ]])
        grad_test = grad(self.test)
        diff_flux = dot(diff_tensor, grad(q))

        F = 0
        F += inner(grad_test, diff_flux)*self.dx

        if not is_continuous(self.trial_space):
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
            alpha = avg(sigma)

            F += alpha*inner(jump(phi, n), dot(avg(diff_tensor), jump(q, n)))*self.dS
            F += -inner(avg(dot(diff_tensor, grad(phi))), jump(q, n))*self.dS
            F += -inner(jump(phi, n), avg(dot(diff_tensor, grad(q))))*self.dS

        for id, bc in bcs.items():
            if 'q' in bc:
                # this corresponds to the second dS term above
                F -= -inner(dot(diff_tensor, grad(phi)), (q-bc['q'])*n)*self.ds(id)
            if 'flux' in bc:
                # this corresponds to the third dS term above,
                # the provided flux = kappa dq/dn = dot(n, dot(diff_tensor, grad(q))
                F += -phi*bc['flux']*self.ds(id)

        return -F


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
