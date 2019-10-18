from abc import ABC, abstractmethod
import firedrake


class BaseEquation(ABC):
    """An equation class that can produce the UFL for the registered terms."""

    """This should be a list of BaseTerm sub-classes that form the terms of the equation."""
    terms = []

    def __init__(self, test_space, trial_space):
        """
        :arg test_space: the test functionspace
        :arg trial_space: The trial functionspace
        test and trial space are only used to determine the the discretisation that's used (ufl_element)
        not what test and trial functions are actually used (these are provided seperately in residual())
        """
        self.test_space = test_space
        self.trial_space = trial_space
        self.mesh = trial_space.mesh()

        # use default quadrature for now
        self.dx = firedrake.dx(domain=self.mesh)
        self.ds = firedrake.ds(domain=self.mesh)
        self.dS = firedrake.dS(domain=self.mesh)

        # self._terms stores the actual instances of the BaseTerm-classes in self.terms
        self._terms = []
        for TermClass in self.terms:
            self._terms.append(TermClass(test_space, trial_space, self.dx, self.ds, self.dS))

    def mass_term(self, test, trial):
        """Return the UFL for the mass term \int test * trial * dx typically used in the time term."""
        return firedrake.inner(test, trial) * self.dx

    def residual(self, test, trial, trial_lagged=None, fields=None, bcs=None):
        """Return the UFL for all terms (except the time derivative)."""
        if trial_lagged is None:
            trial_lagged = trial
        if fields is None:
            fields = {}
        if bcs is None:
            bcs = {}
        F = 0
        for term in self._terms:
            F += term.residual(test, trial, trial_lagged, fields, bcs)

        return F


class BaseTerm(ABC):
    """A term in an equation, that can produce the UFL expression for its contribution to the FEM residual."""
    def __init__(self, test_space, trial_space, dx, ds, dS):
        self.test_space = test_space
        self.trial_space = trial_space
        self.dx = dx
        self.ds = ds
        self.dS = dS
        self.mesh = test_space.mesh()
        self.dim = self.mesh.geometric_dimension()
        self.n = firedrake.FacetNormal(self.mesh)

    @abstractmethod
    def residual(self, test, trial, trial_lagged, fields):
        """Return the UFL for this term"""
        pass
