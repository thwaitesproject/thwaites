from firedrake import *
import numpy as np
from numbers import Number

"""
Code to add actuators to Thwaites, currently Actuator Disc model only
"""

class ActuatorDiscFactory:
    """Class to store and build the source term"""

    def __init__(self, mesh, centres,
                 planes, radii, widths,
                 C_t=4./3.0, a=0.5):

        """ Factory for source terms to apply actuator disk function for turbines.

        Parameters
        ==========

        mesh: firedrake.mesh for calculations
        centres: collection of vectors specifying the location of the centres of the turbines
        planes: collection of vectors specifying the normal vectors of the turbine (when viewed as a disc).
        radii: radiuses of the turbines (arraylike or single value
        widths: lengthscale for spreading kernel
        C_t: Thrus coefficient
        a: Induction factor
        """
        

        self.mesh = mesh
        self.X = SpatialCoordinate(mesh)
        self.centres = [Constant(c) for c in centres]
        self.planes = [Constant(np.asarray(p)/np.linalg.norm(p))
                        for p in planes]
        self.radii = radii
        self.widths = widths
        self.C_t = C_t
        self.a = a

        self.S = FunctionSpace(mesh, "DG", 1)
        self.V = VectorFunctionSpace(mesh, "DG", 1)
        self.T = TensorFunctionSpace(mesh, "DG", 1)
        self.forcing = Function(self.V, name='disc_forcing')
        self.scalar_drag_coefficient = Function(self.S,
                                                          name='disc_coefficient')
        self.drag_coefficient = Function(self.T,
                                                   name='disc_tensor')

        self.calculate_kernels()

    def get_radius(self, key):
        if isinstance(self.radii, Number):
            return self.radii
        else:
            return self.radii[key]

    def get_width(self, key):
        if isinstance(self.widths, Number):
            return self.widths
        else:
            return self.widths[key]

    def calculate_kernels(self):
        """Calculate the spreading kernel for each turbine."""

        self.kernels = []
        
        for k, (X0, n) in enumerate(zip(self.centres, self.planes)):
            self.forcing = Function(self.V)
            r_vec = (self.X-X0-dot(self.X-X0, n)*n )

            radius = self.get_radius(k)
            width = self.get_width(k)
            r2 = dot(r_vec, r_vec)/radius**2

            D = exp(-r2)/0.888
            Tk = exp(-(dot(self.X-X0, n)/width)**2)/width

            F = 0.5*self.C_t*self.a*(1-self.a)

            kernel = Function(self.S)
            kernel.project(F*Tk*D)
            self.kernels.append(kernel)

        
    def update_forcing(self, vel):
        """Update the forcing/source term to apply in the model."""

        tmp = Function(self.V)

        tmp.assign(0*vel)

        self.forcing.assign(0*vel)
        
        for kernel, n in zip(self.kernels, self.planes):
            v = dot(vel, n)
            tmp.project(-sign(v)*kernel*v**2*n)

            self.forcing += tmp
            
        return self.forcing
