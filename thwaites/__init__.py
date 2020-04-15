from firedrake import *
from .scalar_equation import ScalarAdvectionEquation, ScalarAdvectionDiffusionEquation, HybridizedScalarEquation
from .momentum_equation import MomentumEquation, ContinuityEquation
from .time_stepper import ERKEuler, ERKLSPUM2, ERKLPUM2, ERKMidpoint, SSPRK33, \
    BackwardEuler, DIRK22, DIRK23, DIRK33, DIRK43, DIRKLSPUM2, DIRKLPUM2, RelaxToSteadyState
from .coupled_integrators import CrankNicolsonSaddlePointTimeIntegrator, PressureProjectionTimeIntegrator, PressureProjectionSteadyStateSolver
from .assembledschur import AssembledSchurPC
from .meltrate_param import MeltRateParam
from .rans import RANSModel
from .actuators import ActuatorDiscFactory

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
