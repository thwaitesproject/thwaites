from firedrake import *
from .scalar_equation import ScalarAdvectionEquation
from .time_stepper import ERKEuler, ERKLSPUM2, ERKLPUM2, ERKMidpoint, SSPRK33

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
