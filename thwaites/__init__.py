from firedrake import *
from .scalar_equation import ScalarAdvectionEquation, ScalarAdvectionDiffusionEquation
from .time_stepper import ERKEuler, ERKLSPUM2, ERKLPUM2, ERKMidpoint, SSPRK33, \
    BackwardEuler, DIRK22, DIRK23, DIRK33, DIRK43, DIRKLSPUM2, DIRKLPUM2

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
