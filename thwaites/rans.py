from __future__ import absolute_import
from .utility import *
from firedrake import Constant, Function, FunctionSpace, TensorFunctionSpace
from firedrake import TestFunction, TrialFunction, FacetNormal, DirichletBC
from firedrake import dx, ds, dS, dS_v, dS_h, ds_v, ds_t, ds_b, sym, grad, Dx, sqrt, max_value, min_value, conditional
from firedrake import LinearVariationalProblem, LinearVariationalSolver
from firedrake import VertexBasedLimiter, Identity
from pyop2.profiling import timed_stage
from .equations import BaseTerm, BaseEquation
from .scalar_equation import *
from abc import abstractmethod
import numpy as np
import pyop2 as op2
import collections
import thwaites.stability_functions as stability_functions

def ewrite_minmax(u):
    print(u.name(), u.dat.data.min(), u.dat.data.max())

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

        u_plus.interpolate(max_value(muv/solution, 1e-16))


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
        n = FacetNormal(self.mesh)
        a = inner(test, tri)*dx
        uv = self.source
        l = -inner(uv, div(sym(test)))*dx
        l += inner(avg(uv), jump(sym(test), n))*dS
        l += inner(uv, dot(n,sym(test)))*ds
        prob = LinearVariationalProblem(a, l, self.solution, constant_jacobian=True)
        self.weak_grad_solver = LinearVariationalSolver(prob, solver_parameters=solver_parameters)

    def solve(self):
        """Computes the gradient"""
        self.weak_grad_solver.solve()


class VerticalGradSolver(object):
    """
    Computes vertical gradient in the weak sense.

    """
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
        p = self.source
        vert_dim = self.mesh.geometric_dimension()-1
        l = -inner(p, Dx(test, vert_dim))*dx
        l += avg(p)*jump(test, normal[vert_dim])*dS
        l += p*test*normal[vert_dim]*ds
        prob = LinearVariationalProblem(a, l, self.solution, constant_jacobian=True)
        self.weak_grad_solver = LinearVariationalSolver(prob, solver_parameters=solver_parameters)

    def solve(self):
        """Computes the gradient"""
        self.weak_grad_solver.solve()


class BuoyFrequencySolver(object):
    r"""
    Computes buoyancy frequency squared form the given horizontal
    velocity field.

    .. math::
        N^2 = -\frac{g}{\rho_0}\frac{\partial \rho}{\partial z}
    """
    def __init__(self, rho, N2, N2_tmp, relaxation=1.0, minval=1e-12):
        """
        :arg rho: water density field
        :type rho: :class:`Function`
        :arg N2: :math:`N^2` field
        :type N2: :class:`Function`
        :arg N2_tmp: temporary field
        :type N2_tmp: :class:`Function`
        :kwarg float relaxation: relaxation coefficient for mixing old and new
            values N2 = relaxation*N2_new + (1-relaxation)*N2_old
        :kwarg float minval: minimum value for :math:`N^2`
        """
        self._no_op = False
        if rho is None:
            self._no_op = True

        if not self._no_op:

            self.N2 = N2
            self.N2_tmp = N2_tmp
            self.relaxation = relaxation

            g = 9.81 # FIXME
            rho0 = 1.0
            p = -g/rho0 * rho
            solver = VerticalGradSolver(p, self.N2_tmp)
            self.var_solver = solver

    def solve(self, init_solve=False):
        """
        Computes buoyancy frequency

        :kwarg bool init_solve: Set to True if solving for the first time, skips
            relaxation
        """
        with timed_stage('buoy_freq_solv'):
            if not self._no_op:
                self.var_solver.solve()
                gamma = self.relaxation if not init_solve else 1.0
                self.N2.assign(gamma*self.N2_tmp
                               + (1.0 - gamma)*self.N2)


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

        if 'N2_neg' in fields:
            N2 = fields['N2_neg']
            nu_T = fields['eddy_diffusivity']
            f += -nu_T*N2*test*self.dx

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

        if 'N2_pos_over_k' in fields:
            N2_pos_over_k = fields['N2_pos_over_k']
            nu_T = fields['eddy_diffusivity']
            f += -nu_T*N2_pos_over_k*trial*test*self.dx

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
        if 'N2_neg' in fields:
            N2_neg = fields['N2_neg']
            N2_pos = fields['N2_pos']
            C_3_min = fields['C_3_min']
            C_3_plus = fields['C_3_plus']
            nu_T = fields['eddy_diffusivity']

            f += -C_3_plus * gamma * nu_T * N2_neg * test * self.dx

            if C_3_min < 0:
                f += -C_3_min * gamma * nu_T * N2_pos * test * self.dx

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

        if 'N2_pos_over_k' in fields:
            N2_pos_over_k = fields['N2_pos_over_k']
            C_3_min = fields['C_3_min']
            nu_T = fields['eddy_diffusivity']

            if C_3_min > 0:
                f += -C_3_min * nu_T * N2_pos_over_k * trial * test * self.dx
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

            self.C_0 = Constant(options.get('C_0', 1.0))
            self.C_1 = Constant(options.get('C_1', 1.44))
            self.C_2 = Constant(options.get('C_2', 1.92))

            self.schmidt_tke = options.get('schmidt_tke',1.0)
            self.schmidt_psi = options.get('schmidt_psi', 1.3)
            
        elif self.closure_name == 'k-omega':
            
            self.n0 = Constant(1)
            self.n1 = Constant(2)
            self.n2 = Constant(0)

            self.C_0 = Constant(options.get('C_0', 0.09))
            self.C_1 = Constant(options.get('C_1', 5.0/9.0))
            self.C_2 = Constant(options.get('C_2', 0.075))

            self.schmidt_tke = options.get('schmidt_tke',2.0)
            self.schmidt_psi = options.get('schmidt_psi', 2.0)


        if mesh.ufl_cell().is_simplex():
            DG_family = 'DG'
            CG_family = 'CG'
            RT_family = 'RT'
        elif mesh.ufl_cell().cellname() in ['quadrilateral', 'interval * interval']:
            DG_family = 'DQ'
            CG_family = 'Q'
            RT_family = 'RTCF'
        else:
            raise ValueError("Unknown cell type in mesh")

        self.P0 = FunctionSpace(mesh, DG_family, 0)
        self.P0T = TensorFunctionSpace(mesh, DG_family, 0)
        self.P1DGT = TensorFunctionSpace(mesh, DG_family, 1)
        self.P1 = FunctionSpace(mesh, CG_family, 1)
        self.P1DG = FunctionSpace(mesh, DG_family, 1)
        self.RT1 = FunctionSpace(mesh, RT_family, 1)
        self.Z = self.P0 * self.RT1

        if 'density' in fields:
            self.C_mu = Function(self.P0, name="C_mu")
            self.C_mu_p = Function(self.P0, name="C_mu_p")
        elif self.closure_name == 'k-epsilon':
            self.C_mu = Constant(options.get('C_mu', 0.09))
        elif self.closure_name == 'k-omega':
            self.C_mu = Constant(options.get('C_mu', 1.0))

        self.fields.rans_mixing_length = Function(self.P0, name='rans_mixing_length')
        self.gamma1 = Function(self.P0, name='rans_linearization_1')
        if self.closure_name == 'k-omega':
            self.gamma2 = Function(self.P0, name='rans_linearization_2')
        else:
            self.gamma2 = self.gamma1
        if not 'rans_eddy_viscosity' in self.fields:
            self.fields.rans_eddy_viscosity = Function(self.P1, name='rans_eddy_viscosity')
        self.fields.z_tke = Function(self.Z, name='rans_tke_hybrid')
        self.fields.z_psi = Function(self.Z, name='rans_psi_hybrid')
        self.tke, self.grad_tke = self.fields.z_tke.split()
        self.psi, self.grad_psi = self.fields.z_psi.split()

        self.sqrt_tke = Function(self.P0, name='sqrt_tke')
        self.production = Function(self.P0, name='production')
        self.rate_of_strain = Function(self.P0T, name='rate of strain')
        self.rate_of_strain_p1dg = Function(self.P1DGT, name='rate of strain p1dg')
        self.rate_of_strain_vert = Function(self.P0, name='rate of strain vertical')

        self.eq_rans_tke = HybridizedScalarEquation(RANSTKEEquation2D, self.Z, self.Z)
        self.eq_rans_psi = HybridizedScalarEquation(RANSPsiEquation2D, self.Z, self.Z)

        self.uv = self.fields['velocity']
        self.eddy_viscosity = Function(self.P0, name='P0 eddy viscosity')
        self.u_tau = Function(self.P1, name='u_tau')
        self.u_plus = Function(self.u_tau.function_space(),
                               name='u plus')
        self.y_plus = Function(self.u_tau.function_space(),
                               name='y plus')
        # buoyancy terms:
        if 'density' in self.fields:
            self.Ri_st = options.get('Ri_st', 0.25)  # steady state Richardson number
            self.stability_function = options.get('stability_function', None)
            if self.stability_function is None:
                stab_args = {'lim_alpha_shear': True,
                             'lim_alpha_buoy': True,
                             'smooth_alpha_buoy_lim': False}
                self.stability_function = stability_functions.StabilityFunctionCanutoA(**stab_args)
            self.C_3_plus = Constant(options.get('C_3_plus', 1.0))
            self.C_3_min = self.stability_function.compute_c3_minus(float(self.C_1), float(self.C_2), self.Ri_st)
            self.cmu0 = self.stability_function.compute_cmu0()
            self.galperin_clim = self.stability_function.compute_length_clim(self.cmu0, self.Ri_st)
            rho = self.fields['density']
            self.fields.M2 = Function(self.P0, name='shear frequency')
            self.fields.N2 = Function(self.P0, name='buoyancy frequency')
            N2_tmp = Function(self.P0, name='temp. buoyancy frequency')
            self.fields.N2_pos_over_k = Function(self.P0, name='positive buoyancy frequency over k')
            self.fields.N2_pos = Function(self.P0, name='positive buoyancy frequency')
            self.fields.N2_neg = Function(self.P0, name='negative buoyancy frequency')
            self.eddy_diffusivity = Function(self.P0, name='P0 eddy diffusivity')

            self.buoyancy_frequency_solver = BuoyFrequencySolver(self.fields['density'], self.fields.N2, N2_tmp)


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
            if 'stress' in funcs:
                self.bcs_tke[bnd_marker]['flux'] = Constant(0.0)
                if self.closure_name == 'k-epsilon':
                    ## PSI flux from fluidity:
                    #n = as_vector((0, -1))  # FIXME
                    #t = funcs['stress'] - dot(n, funcs['stress'])*n
                    #u_tau = dot(t,t)**0.25
                    #y = 18500/9.81*u_tau**2 # Charnock formula from fluidity source
                    #self.psi_flux_expr = self.C_mu*self.tke**2/y
                    z_s=2.5*0.5+0.05
                    self.psi_flux_expr = -(-1*self.eddy_viscosity*(self.cmu0)**3 * self.tke**1.5 * 0.4**-1 * z_s**-2)
                    self.psi_flux = Function(self.P0, name='psi flux')
                    self.bcs_psi[bnd_marker]['flux'] = self.psi_flux
                elif self.closure_name == 'k-omega':
                    assert False
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

        self.rate_of_strain_solver = RateOfStrainSolver(self.uv, self.rate_of_strain)
        self.rate_of_strain_solver_p1dg = RateOfStrainSolver(self.uv, self.rate_of_strain_p1dg)
        self.rate_of_strain_solver_vert = VerticalGradSolver(self.uv[0], self.rate_of_strain_vert)
        if self.walls:
            self.wall_viscosity_bc = DirichletBC(self.fields.rans_eddy_viscosity.function_space(), 0.4*self.y_plus*1e-6, self.walls)
            self.wall_production = Function(self.production.function_space(), name="wall_production")
            self.wall_production_bc = DirichletBC(self.production.function_space(), self.wall_production, self.walls)
        self.p1_averager = P1Average(self.P0, self.P1, self.P1DG)

        #self.limiter = VertexBasedLimiter(self.P0)

    def _create_integrators(self, integrator, dt, bnd_conditions, solver_parameters):
        
        viscosity = self.fields.viscosity
        if viscosity.ufl_shape == ():
            diffusivity_tke = (viscosity + self.eddy_viscosity)/self.schmidt_tke
            diffusivity_psi = (viscosity + self.eddy_viscosity)/self.schmidt_psi
        elif len(viscosity.ufl_shape) == 2:
            diffusivity_tke = (viscosity + self.eddy_viscosity*Identity(viscosity.ufl_shape[0]))/self.schmidt_tke
            diffusivity_psi = (viscosity + self.eddy_viscosity*Identity(viscosity.ufl_shape[0]))/self.schmidt_psi
        else:
            raise ValueError("Unknown shape of viscosity")
        fields_tke = {
            'velocity': self.uv,
            'diffusivity': diffusivity_tke,
            'production': self.production,
            'eddy_viscosity': self.eddy_viscosity,
            'eddy_diffusivity': self.eddy_diffusivity,
            'gamma1': self.gamma1,
            'gamma2': self.gamma2,
            'N2_neg': self.fields.N2_neg,
            'N2_pos_over_k': self.fields.N2_pos_over_k,
            'C_0': self.C_0,
            'dt': dt,
        }
        fields_psi = {
            'velocity': self.uv,
            'diffusivity': diffusivity_psi,
            'production': self.production,
            'eddy_viscosity': self.eddy_viscosity,
            'eddy_diffusivity': self.eddy_diffusivity,
            'gamma1': self.gamma1,
            'gamma2': self.gamma2,
            'N2_neg': self.fields.N2_neg,
            'N2_pos': self.fields.N2_pos,
            'N2_pos_over_k': self.fields.N2_pos_over_k,
            'C_1': self.C_1,
            'C_2': self.C_2,
            'C_3_min': self.C_3_min,
            'C_3_plus': self.C_3_plus,
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


    def preprocess(self, init_solve=False):
        self.rate_of_strain_solver.solve()
        self.rate_of_strain_solver_p1dg.solve()
        self.rate_of_strain_solver_vert.solve()
        #self.fields.M2.interpolate(inner(self.rate_of_strain, self.rate_of_strain))
        self.fields.M2.interpolate(self.rate_of_strain_vert**2)

        if 'density' in self.fields:
            self.buoyancy_frequency_solver.solve()
            self.fields.N2_pos_over_k.interpolate(max_value(self.fields.N2, 0)/self.tke)
            self.fields.N2_pos.interpolate(max_value(self.fields.N2, 0))
            self.fields.N2_neg.interpolate(min_value(self.fields.N2, 0))

        if 'viscosity' in self.fields:
            mu = self.fields['viscosity']
            if mu.ufl_shape == ():
                mu = mu + self.eddy_viscosity
            elif len(mu.ufl_shape) == 2:
                mu = mu[1,1] + self.eddy_viscosity
        else:
            mu = self.eddy_viscosity
        self.production.interpolate(mu*self.fields.M2)
        if self.closure_name == 'k-epsilon':
            #self.gamma1.project(self.C_mu*max_value(self.tke, 0.)/self.eddy_viscosity)
            self.gamma1.project(self.psi/self.tke)
        elif self.closure_name == 'k-omega':
            self.gamma1.project(conditional(self.psi>0, self.psi, Constant(0.0)))
            # check with James:
            self.gamma2.project(self.C_mu*self.sqrt_tke**self.n2/self.eddy_viscosity)

        # hack needed to make hybrized time integration work:
        self.grad_tke_old.assign(0.)
        self.grad_psi_old.assign(0.)
        self.grad_tke.assign(0.)
        self.grad_psi.assign(0.)

    def postprocess(self):

        self.tke.interpolate(max_value(self.tke, 1e-6))
        val = (sqrt(2)*self.galperin_clim * (self.cmu0)**-3 * self.tke**-1 * (self.fields.N2_pos + 1e-12)**(-0.5))**-1
        self.psi.interpolate(max_value(self.psi,max_value(val, 1e-14)))

        self.sqrt_tke.project(conditional(self.tke>0, sqrt(self.tke), Constant(0.0)))
        if 'density' in self.fields:
            cmu_dat, cmu_p_dat = self.stability_function.evaluate(self.fields.M2.dat.data, self.fields.N2.dat.data, self.tke.dat.data, self.psi.dat.data)
            self.C_mu.dat.data[:] = cmu_dat
            self.C_mu_p.dat.data[:] = cmu_p_dat
            self.psi_flux.interpolate(self.psi_flux_expr)

        #self.limiter.apply(self.tke)
        l_min = 1e-12
        self.fields.rans_mixing_length.interpolate(conditional(self.cmu0**3*self.tke**1.5>l_min*self.psi, self.cmu0**3*self.tke**1.5/self.psi, l_min))
        #self.eddy_viscosity.project(conditional(self.nu_0>self.fields.rans_mixing_length*self.sqrt_tke,
        #                                       self.nu_0,
        #                                       self.fields.rans_mixing_length*self.sqrt_tke))
        self.eddy_viscosity.interpolate(max_value(self.C_mu*self.fields.rans_mixing_length*self.sqrt_tke/self.cmu0**3, 1e-8))
        self.eddy_diffusivity.interpolate(self.C_mu_p/self.C_mu*self.eddy_viscosity)

        if self.walls:
            self.wall_solver.apply(self.u_tau, self.u_plus, self.y_plus, self.uv)
            self.wall_viscosity_bc.apply(self.fields.rans_eddy_viscosity)
            self.wall_production.interpolate(self.u_tau**4/self.fields.rans_eddy_viscosity)
            self.wall_production_bc.apply(self.production)

        #self.limiter.apply(self.psi)
        self.p1_averager.apply(self.eddy_viscosity, self.fields.rans_eddy_viscosity)
        #self.tke.interpolate(max_value(self.tke, 0.))
        #self.psi.interpolate(max_value(self.psi, 0.))
        
    def initialize(self, rans_tke=Constant(0.0), rans_psi=Constant(0.0), **kwargs):
        self.tke.project(rans_tke)
        self.psi.project(rans_psi)
        self.wall_solver.apply(self.u_tau, self.u_plus, self.y_plus, self.uv)
        self.timesteppers.rans_tke.initialize(self.fields.z_tke)
        self.timesteppers.rans_psi.initialize(self.fields.z_psi)
        self.preprocess()
        self.postprocess()

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
