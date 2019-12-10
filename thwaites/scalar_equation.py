from .equations import BaseTerm, BaseEquation
from firedrake import dot, inner, div, grad, CellDiameter, as_matrix, as_vector, Identity
from firedrake import avg, jump, Constant, split, FacetNormal, min_value, max_value, as_tensor
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

        # integration by parts leads to boundary term
        F += q*dot(n, u)*phi*self.ds

        # which is replaced at incoming Dirichlet 'q' boundaries:
        for id, bc in bcs.items():
            if 'q' in bc:
                # on incoming boundaries, dot(u,n)<0, replace q with bc['q']
                F += phi*min_value(dot(u, n),0)*(bc['q']-q) * self.ds(id)

        if not (is_continuous(self.trial_space) and normal_is_continuous(u)):
            # outgoing velocity dot(u,n)>0 (i.e. the upwind side of the face)
            un = max_value(dot(u,n), 0)
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
        if kappa.ufl_shape == ():
            if 'rans_eddy_diffusivity' in fields:
                kappa = kappa + fields['rans_eddy_diffusivity']

            diff_tensor = as_matrix([[kappa, 0, ],
                                     [0, kappa, ]])
        elif len(kappa.ufl_shape) == 2:
            if 'rans_eddy_diffusivity' in fields:
                diff_tensor = kappa + fields['rans_eddy_diffusivity']*Identity(kappa.ufl_shape[0])
            else:
                diff_tensor = kappa
        else:
            raise ValueError("Unknown shape of diffusivity")

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

        degree = self.trial_space.ufl_element().degree()
        alpha = fields.get('interior_penalty', 5.0)
        sigma = alpha*degree*(degree + 1)/cellsize
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
            elif 'wall_law_drag' in bc:
                F += phi*bc['wall_law_drag']*q*self.ds(id)

        return -F

class ScalarSourceTerm(BaseTerm):
    r"""
        Source term :math:`s_T`
    """

    def residual(self, test, trial, trial_lagged, fields, bcs):
        if 'source' not in fields:
            return 0
        phi = test
        source = fields['source']

        # NOTE, here source term F is already on the RHS
        F = dot(phi, source)*self.dx

        return F


class ScalarAbsorptionTerm(BaseTerm):
    r"""
            Absorption Term :math:`\alpha_T T`
        """

    def residual(self, test, trial, trial_lagged, fields, bcs):
        if 'absorption coefficient' not in fields:
            return 0

        phi = test
        alpha = fields['absorption coefficient']

        # NOTE, here absorption term F is already on the RHS
        # implement absorption term implicitly at current time step.
        F = -dot(phi, alpha*trial)*self.dx

        return F


class ScalarVelocity2halfDTerm(BaseTerm):
    r"""
            coriolis forcing for scalar advection diffusion equation for x component of velocity :math:`-fv`
        """

    def residual(self, test, trial, trial_lagged, fields, bcs):
        assert 'coriolis_frequency' in fields
        v = fields['velocity']
        f = fields['coriolis_frequency']
        F = -f * v[0] * test * self.dx
        return -F


class ScalarAdvectionEquation(BaseEquation):
    """
    Scalar equation with only an advection term.
    """

    terms = [ScalarAdvectionTerm, ScalarSourceTerm, ScalarAbsorptionTerm]


class ScalarAdvectionDiffusionEquation(BaseEquation):
    """
    Scalar equation with advection and diffusion.
    """

    terms = [ScalarAdvectionTerm, ScalarDiffusionTerm, ScalarSourceTerm, ScalarAbsorptionTerm]


class ScalarVelocity2halfDEquation(BaseEquation):
    """
    Scalar equation with advection, and diffusion (viscosity) for x compenent of velocity for 2.5D coriolis modelling.
    """

    terms = [ScalarAdvectionTerm, ScalarDiffusionTerm, ScalarSourceTerm, ScalarAbsorptionTerm,
             ScalarVelocity2halfDTerm]


class HybridizedScalarEquation(BaseEquation):
    def __init__(self, scalar_equation_class, test_space, trial_space):
        self.eq_q = scalar_equation_class(test_space.sub(0), trial_space.sub(0))
        super().__init__(test_space, trial_space)

    def residual(self, test, trial, trial_lagged=None, fields=None, bcs=None):
        if trial_lagged is not None:
            trial_lagged_q = trial_lagged[0]
        else:
            trial_lagged_q = None

        F = self.eq_q.residual(test[0], trial[0],
                             trial_lagged=trial_lagged_q,
                             fields=fields, bcs=bcs)

        if isinstance(trial, list):
            qtri, ptri = trial
        else:
            qtri, ptri = split(trial)
        qtest, ptest = split(test)

        n = FacetNormal(self.mesh)
        kappa = fields['diffusivity']

        dt = fields['dt']
        F += qtest*div(kappa*ptri)*self.dx
        F += -dot(div(ptest), trial[0])/dt*self.dx

        F += -qtest*dot(n, kappa*ptri)*self.ds + dot(n, ptest)*trial[0]/dt*self.ds

        for id, bc in bcs.items():
            if 'q' in bc:
                F += qtest*dot(n, kappa*ptri)*self.ds(id) - dot(n, ptest)*(trial[0]-bc['q'])/dt*self.ds(id)
            if 'flux' in bc:
                F += qtest*bc['flux']*self.ds(id)
            if 'wall_law_drag' in bc:
                F += qtest*bc['wall_law_drag']*qtri*self.ds(id)


        return F
