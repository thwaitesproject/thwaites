from __future__ import absolute_import
from .utility import *
from firedrake import Constant, Function, FunctionSpace, TensorFunctionSpace
from firedrake import TestFunction, TrialFunction, FacetNormal
from firedrake import dx, ds, dS, sym, grad, sqrt, max_value, min_value, DirichletBC
from firedrake import LinearVariationalProblem, LinearVariationalSolver
from firedrake import VertexBasedLimiter
from pyop2.profiling import timed_stage
from .equations import BaseTerm, BaseEquation
from .scalar_equation import *
from abc import abstractmethod
import numpy as np
import pyop2 as op2
import collections


class WallSolver(object):

    def __init__(self, bnd_marker, delta, viscosity):
        
        self.bnd_marker = bnd_marker
        self.delta = delta
        self.viscosity = viscosity

    def apply(self, solution, u_plus, y_plus, uv):

        if self.bnd_marker:
            bnd_set = solution.function_space().boundary_nodes(self.bnd_marker, "geometric")
        else:
            bnd_set = []

        muv = Function(solution.function_space())
        muv.project(sqrt(dot(uv, uv)))
        muv.interpolate(max_value(muv, 0.))
        #muv.dat.data[:] = np.sqrt(np.sum(uv.dat.data*uv.dat.data,axis=1))

        solution.project(sqrt(self.viscosity*muv/self.delta))
        
        self.kernel = op2.Kernel("""

            double f(double y_plus){ 
                 if (y_plus<20) y_plus=20;
                 return 1.0/0.4*log(y_plus) + 5.5;
            }

            void newton_loop(double *solution, double*y_plus, double* muv) {

                *y_plus = (*solution)*%(delta)f/%(viscosity)f;
                if (*y_plus<20.0) return;

                for ( int i = 0; i < 100; i++ ) {
                     *y_plus = (*solution)*%(delta)f/%(viscosity)f;
                     double fval = f(*y_plus);
                     *solution += (*muv-fval*(*solution))/(1.0/0.4+fval);
                }
            }""" % {'delta': self.delta,
                    'viscosity': self.viscosity},
            'newton_loop')

        op2.par_loop(
            self.kernel, solution.function_space().node_set(bnd_set),
            solution.dat(op2.INC),
            y_plus.dat(op2.INC),
            muv.dat(op2.READ),
            iterate=op2.ALL)
        
        u_plus.interpolate(conditional(muv>1e-16*solution, muv/solution, 1e-16))

class RateOfStrainSolver(object):
    """
    Computes vertical gradient in the weak sense.

    """
    # TODO add weak form of the problem
    def __init__(self, source, solution, solver_parameters=None):
        """
        :arg source: A :class:`Function` or expression to differentiate.
        :arg solution: A :class:`Function` where the solution will be stored.
            Must be in P0 space.
        :kwarg dict solver_parameters: PETSc solver options
        """
        if solver_parameters is None:
            solver_parameters = {}
        solver_parameters.setdefault('snes_type', 'ksponly')
        solver_parameters.setdefault('ksp_type', 'preonly')
        solver_parameters.setdefault('pc_type', 'bjacobi')
        solver_parameters.setdefault('sub_ksp_type', 'preonly')
        solver_parameters.setdefault('sub_pc_type', 'ilu')

        self.source = source
        self.solution = solution

        self.fs = self.solution.function_space()
        self.mesh = self.fs.mesh()

        # weak gradient evaluator
        test = TestFunction(self.fs)
        tri = TrialFunction(self.fs)
        normal = FacetNormal(self.mesh)
        a = inner(test, tri)*dx
        uv = self.source
        stress = sym(grad(uv))
        stress_jump = sym(tensor_jump(uv, normal))
        l = inner(test, stress)*dx
        l += -inner(avg(test), stress_jump)*dS
        #l += -inner(test, sym(outer(uv, normal)))*ds
        prob = LinearVariationalProblem(a, l, self.solution, constant_jacobian=True)
        self.weak_grad_solver = LinearVariationalSolver(prob, solver_parameters=solver_parameters)

    def solve(self):
        """Computes the gradient"""
        self.weak_grad_solver.solve()

class ProductionSolver(object):
    r"""
    Computes vertical shear frequency squared form the given horizontal
    velocity field.

    .. math::
        M^2 = \left(\frac{\partial u}{\partial z}\right)^2
            + \left(\frac{\partial v}{\partial z}\right)^2
    """
    def __init__(self, uv, production, rate_of_strain, eddy_viscosity, relaxation=1.0, minval=1e-12):
        """
        :arg uv: horizontal velocity field
        :type uv: :class:`Function`
        :arg m2: :math:`M^2` field
        :type m2: :class:`Function`
        :arg mu: field for x component of :math:`M^2`
        :type mu: :class:`Function`
        :arg mv: field for y component of :math:`M^2`
        :type mv: :class:`Function`
        :arg mu_tmp: temporary field
        :type mu_tmp: :class:`Function`
        :kwarg float relaxation: relaxation coefficient for mixing old and new values
            M2 = relaxation*M2_new + (1-relaxation)*M2_old
        :kwarg float minval: minimum value for :math:`M^2`
        """
        self.production = production
        self.rate_of_strain = rate_of_strain
        self.eddy_viscosity = eddy_viscosity
        self.minval = minval
        self.relaxation = relaxation

        self.var_solver = RateOfStrainSolver(uv, self.rate_of_strain)

    def solve(self, init_solve=False):
        """
        Computes buoyancy frequency

        :kwarg bool init_solve: Set to True if solving for the first time, skips
            relaxation
        """
        # TODO init_solve can be omitted with a boolean property
        with timed_stage('shear_freq_solv'):
            self.var_solver.solve()

        tau = self.rate_of_strain
        self.production.interpolate(2.0*self.eddy_viscosity*(tau[0,0]**2+tau[0,1]**2+tau[1,0]**2+tau[1,1]**2))

class RANSTKESourceTerm(BaseTerm):
    r"""
    Production and destruction terms of the TKE equation :eq:`turb_tke_eq`

    .. math::
        F_k = P + B - \varepsilon

    To ensure positivity we use Patankar-type time discretization: all source
    terms are treated explicitly and sink terms are treated implicitly.
    To this end the buoyancy production term :math:`B` is split in two:

    .. math::
        F_k = P + B_{source} + \frac{k^{n+1}}{k^n}(B_{sink} - \varepsilon)

    with :math:`B_{source} \ge 0` and :math:`B_{sink} < 0`.
    """
    def residual(self, test, trial, trial_lagged, fields, bcs):

        production = fields['production']
        
        f = production*test*self.dx
        
        return f

class RANSTKEDestructionTerm(BaseTerm):
    r"""
    Production and destruction terms of the TKE equation :eq:`turb_tke_eq`

    .. math::
        F_k = P + B - \varepsilon

    To ensure positivity we use Patankar-type time discretization: all source
    terms are treated explicitly and sink terms are treated implicitly.
    To this end the buoyancy production term :math:`B` is split in two:

    .. math::
        F_k = P + B_{source} + \frac{k^{n+1}}{k^n}(B_{sink} - \varepsilon)

    with :math:`B_{source} \ge 0` and :math:`B_{sink} < 0`.
    """
    def residual(self, test, trial, trial_lagged, fields, bnd_conditions=None):

        gamma = fields['gamma1']
        C_0 = fields['C_0']
        
        f = -inner(gamma*trial, C_0*test)*self.dx
        
        return f

class RANSPsiSourceTerm(BaseTerm):
    r"""
    Production and destruction terms of the TKE equation :eq:`turb_tke_eq`

    .. math::
        F_k = P + B - \varepsilon

    To ensure positivity we use Patankar-type time discretization: all source
    terms are treated explicitly and sink terms are treated implicitly.
    To this end the buoyancy production term :math:`B` is split in two:

    .. math::
        F_k = P + B_{source} + \frac{k^{n+1}}{k^n}(B_{sink} - \varepsilon)

    with :math:`B_{source} \ge 0` and :math:`B_{sink} < 0`.
    """
    def residual(self, test, trial, trial_lagged, fields, bcs):

        production = fields['production']
        gamma = fields['gamma2']
        C_1 = fields['C_1']
        
        f = C_1*gamma*production*test*self.dx
        
        return f

class RANSPsiDestructionTerm(BaseTerm):
    r"""
    Production and destruction terms of the TKE equation :eq:`turb_tke_eq`

    .. math::
        F_k = P + B - \varepsilon

    To ensure positivity we use Patankar-type time discretization: all source
    terms are treated explicitly and sink terms are treated implicitly.
    To this end the buoyancy production term :math:`B` is split in two:

    .. math::
        F_k = P + B_{source} + \frac{k^{n+1}}{k^n}(B_{sink} - \varepsilon)

    with :math:`B_{source} \ge 0` and :math:`B_{sink} < 0`.
    """
    def residual(self, test, trial, trial_lagged, fields, bcs):

        gamma = fields['gamma1']
        C_2  = fields['C_2']

        f = -C_2*inner(gamma*trial, test)*self.dx
        return f


class TurbulenceModel(object):
    """Base class for all vertical turbulence models"""

    @abstractmethod
    def initialize(self):
        """Initialize all turbulence fields"""
        pass

    @abstractmethod
    def preprocess(self, init_solve=False):
        """
        Computes all diagnostic variables that depend on the mean flow model
        variables.

        To be called before updating the turbulence PDEs.
        """
        pass

    @abstractmethod
    def postprocess(self):
        """
        Updates all diagnostic variables that depend on the turbulence state
        variables.

        To be called after updating the turbulence PDEs.
        """
        pass

class RANSModel(TurbulenceModel):

    def __init__(self, fields, mesh, options=AttrDict(), bcs=None):

        self.fields = AttrDict(fields)
        self.options = options
        self.timesteppers = AttrDict()

        self.nu_0 = Constant(options.get('nu_0', 1.0e-6))
        self.l_max = Constant(options.get('l_max', 1.0))

        self.delta = options.get('delta', 1e-3)

        self.bcs = bcs or {}
        self.closure_name = options.get('closure_name', 'k-epsilon')

        
        if self.closure_name == 'k-epsilon':
            
            self.n0 = Constant(3)
            self.n1 = Constant(2)
            self.n2 = Constant(2)

            self.C_mu = Constant(options.get('C_mu', 0.09))
            self.C_0 = Constant(options.get('C_0', 1.0))
            self.C_1 = Constant(options.get('C_1', 1.44))
            self.C_2 = Constant(options.get('C_2', 1.92))

            self.schmidt_tke = options.get('schmidt_tke',1.0)
            self.schmidt_psi = options.get('schmidt_psi', 1.3)
            
        elif self.closure_name == 'k-omega':
            
            self.n0 = Constant(1)
            self.n1 = Constant(2)
            self.n2 = Constant(0)

            self.C_mu = Constant(options.get('C_mu', 1.0))
            self.C_0 = Constant(options.get('C_0', 0.09))
            self.C_1 = Constant(options.get('C_1', 5.0/9.0))
            self.C_2 = Constant(options.get('C_2', 0.075))

            self.schmidt_tke = options.get('schmidt_tke',2.0)
            self.schmidt_psi = options.get('schmidt_psi', 2.0)
        

        self.P0_2d = FunctionSpace(mesh, "DG", 0)
        self.P0_2dT = TensorFunctionSpace(mesh, "DG", 0)
        self.P1_2d = FunctionSpace(mesh, "CG", 1)
        self.P1DG_2d = FunctionSpace(mesh, "DG", 1)
        self.RT1 = FunctionSpace(mesh, "RT", 1)
        self.Z = self.P0_2d * self.RT1

        self.fields.rans_mixing_length = Function(self.P0_2d, name='rans_mixing_length')
        self.gamma1 = Function(self.P0_2d, name='rans_linearization_1')
        if self.closure_name == 'k-omega':
            self.gamma2 = Function(self.P0_2d, name='rans_linearization_2')
        else:
            self.gamma2 = self.gamma1
        if not 'rans_eddy_viscosity' in self.fields:
            self.fields.rans_eddy_viscosity = Function(self.P1_2d, name='rans_eddy_viscosity')
        self.fields.z_tke = Function(self.Z, name='rans_tke_hybrid')
        self.fields.z_psi = Function(self.Z, name='rans_psi_hybrid')
        self.tke, self.grad_tke = self.fields.z_tke.split()
        self.psi, self.grad_psi = self.fields.z_psi.split()

        self.sqrt_tke = Function(self.P0_2d, name='sqrt_tke')
        self.production = Function(self.P0_2d, name='production')
        self.rate_of_strain = Function(self.P0_2dT, name='rate of strain')

        self.eq_rans_tke = HybridizedScalarEquation(RANSTKEEquation2D, self.Z, self.Z)
        self.eq_rans_psi = HybridizedScalarEquation(RANSPsiEquation2D, self.Z, self.Z)

        self.uv = self.fields['velocity']
        self.eddy_viscosity = Function(self.P0_2d, name='P0 eddy viscosity')
        self.u_tau = Function(self.P1_2d)
        self.u_plus = Function(self.u_tau.function_space(),
                               name='u plus')
        self.y_plus = Function(self.u_tau.function_space(),
                               name='y plus')

        self.walls = set()
        self.bcs_tke = collections.defaultdict(dict)
        self.bcs_psi = collections.defaultdict(dict)
        for bnd_marker, funcs in self.bcs.items():
            if 'wall_law' in funcs:
                self.walls.add(bnd_marker)
                funcs['wall_law_drag'] = self.u_tau/self.u_plus
                self.bcs_tke[bnd_marker]['flux'] = Constant(0.0)
                if self.closure_name == 'k-epsilon':
                    self.bcs_psi[bnd_marker]['wall_law_drag'] = 0.4*self.u_tau/self.schmidt_psi
                elif self.closure_name == 'k-omega':
                    assert False
                    self.bcs_psi[bnd_marker]['flux'] = -self.u_tau/self.delta**2/0.4/sqrt(self.C_0)
            if 'tke' in funcs:
                self.bcs_tke[bnd_marker]['q'] = funcs['tke']
            if 'psi' in funcs:
                self.bcs_psi[bnd_marker]['q'] = funcs['psi']
            if 'un' in funcs:
                self.bcs_tke[bnd_marker]['un'] = funcs['un']
                self.bcs_psi[bnd_marker]['un'] = funcs['un']
            if 'u' in funcs:
                self.bcs_tke[bnd_marker]['u'] = funcs['u']
                self.bcs_psi[bnd_marker]['u'] = funcs['u']

        self.wall_solver = WallSolver(self.walls, self.delta, 1.0e-6)

        self.production_solver = ProductionSolver(self.uv, self.production, self.rate_of_strain, self.eddy_viscosity)
        if self.walls:
            self.wall_viscosity_bc = DirichletBC(self.fields.rans_eddy_viscosity.function_space(), 0.4*self.y_plus*1e-6, self.walls)
            self.wall_production = Function(self.production.function_space(), name="wall_production")
            self.wall_production_bc = DirichletBC(self.production.function_space(), self.wall_production, self.walls)
        self.p1_averager = P1Average(self.P0_2d, self.P1_2d, self.P1DG_2d)

    def preprocess(self, init_solve=False):
        self.production_solver.var_solver.source.assign(self.uv)
        self.production_solver.solve()
        
        self.sqrt_tke.project(conditional(self.tke>0, sqrt(self.tke), Constant(0.0)))

        self.fields.rans_mixing_length.project(conditional(self.psi*self.l_max>self.C_mu*(self.sqrt_tke**self.n0),
                                                self.C_mu*self.sqrt_tke**self.n0/max_value(self.psi, 1e-6),
                                                self.l_max))
        #self.fields.rans_mixing_length.interpolate(self.C_mu*max_value(self.sqrt_tke,1e-16)**self.n0/max_value(self.psi, 1e-16))
        #self.fields.rans_mixing_length.interpolate(min_value(self.fields.rans_mixing_length, self.l_max))
        
        self.eddy_viscosity.project(conditional(self.nu_0>self.fields.rans_mixing_length*self.sqrt_tke,
                                               self.nu_0,
                                               self.fields.rans_mixing_length*self.sqrt_tke))
        #self.eddy_viscosity.project(self.C_mu*max_value(self.tke, 1e-16)**2/max_value(self.psi, 1e-16))
        #self.eddy_viscosity.interpolate(
        #       max_value(min_value(self.eddy_viscosity, self.l_max*self.sqrt_tke), self.nu_0))
        #self.limiter.apply(self.eddy_viscosity)



        if self.closure_name == 'k-epsilon':
            self.gamma1.project(self.C_mu*max_value(self.tke, 0.)/self.eddy_viscosity)
        elif self.closure_name == 'k-omega':
            self.gamma1.project(conditional(self.psi>0, self.psi, Constant(0.0)))
            # check with James:
            self.gamma2.project(self.C_mu*self.sqrt_tke**self.n2/self.eddy_viscosity)
        if self.walls:
            self.wall_solver.apply(self.u_tau, self.u_plus, self.y_plus, self.uv)
            self.wall_viscosity_bc.apply(self.fields.rans_eddy_viscosity)
            self.wall_production.interpolate(self.u_tau**4/self.fields.rans_eddy_viscosity)
            self.wall_production_bc.apply(self.production)
        self.grad_tke_old.assign(0.)
        self.grad_psi_old.assign(0.)
        self.grad_tke.assign(0.)
        self.grad_psi.assign(0.)

    def postprocess(self):

        #self.limiter.apply(self.tke)
        #self.limiter.apply(self.psi)
        self.p1_averager.apply(self.eddy_viscosity, self.fields.rans_eddy_viscosity)
        #self.tke.interpolate(max_value(self.tke, 0.))
        #self.psi.interpolate(max_value(self.psi, 0.))
        
    def _create_integrators(self, integrator, dt, bnd_conditions, solver_parameters):
        
        diffusivity = self.fields.diffusivity
        diffusivity_tke = diffusivity + self.eddy_viscosity/self.schmidt_tke
        diffusivity_psi = diffusivity + self.eddy_viscosity/self.schmidt_psi
        fields_tke = {
                  'velocity': self.uv,
                  'diffusivity': diffusivity_tke,
                  'production': self.production,
                  'eddy_viscosity': self.eddy_viscosity,
                  'gamma1': self.gamma1,
                  'gamma2': self.gamma2,
                  'C_0': self.C_0,
                  'dt': dt,
                  }
        fields_psi = {
                  'velocity': self.uv,
                  'diffusivity': diffusivity_psi,
                  'production': self.production,
                  'eddy_viscosity': self.eddy_viscosity,
                  'gamma1': self.gamma1,
                  'gamma2': self.gamma2,
                  'C_1': self.C_1,
                  'C_2': self.C_2,
                  'dt': dt,
                  }


        self.timesteppers.rans_tke = integrator(self.eq_rans_tke, self.fields.z_tke, fields_tke, dt,
                                                bnd_conditions  = self.bcs_tke,
                                                solver_parameters=solver_parameters)

        self.timesteppers.rans_psi = integrator(self.eq_rans_psi, self.fields.z_psi, fields_psi, dt,
                                                bnd_conditions  = self.bcs_psi,
                                                solver_parameters=solver_parameters)
        
        _, self.grad_tke_old = self.timesteppers.rans_tke.solution_old.split()
        _, self.grad_psi_old = self.timesteppers.rans_psi.solution_old.split()

    def initialize(self, rans_tke=Constant(0.0), rans_psi=Constant(0.0), **kwargs):
        self.tke.project(rans_tke)
        self.psi.project(rans_psi)
        self.wall_solver.apply(self.u_tau, self.u_plus, self.y_plus, self.uv)
        self.timesteppers.rans_tke.initialize(self.fields.z_tke)
        self.timesteppers.rans_psi.initialize(self.fields.z_psi)

    def advance(self, t, update_forcings=None):
        self.preprocess()
        self.timesteppers.rans_tke.advance(t, update_forcings=update_forcings)
        self.timesteppers.rans_psi.advance(t, update_forcings=update_forcings)
        self.postprocess()


class RANSTKEEquation2D(BaseEquation):
    """
    2D tracer advection-diffusion equation :eq:`tracer_eq` in conservative form
    """
    terms = [ScalarAdvectionTerm, RANSTKEDestructionTerm, RANSTKESourceTerm]


class RANSPsiEquation2D(BaseEquation):
    """
    2D tracer advection-diffusion equation :eq:`tracer_eq` in conservative form
    """
    terms = [ScalarAdvectionTerm, RANSPsiDestructionTerm, RANSPsiSourceTerm]
