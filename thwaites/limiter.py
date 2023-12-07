"""
Slope limiters for discontinuous fields
"""
from __future__ import absolute_import
from firedrake import FunctionSpace, TestFunction, assemble
from firedrake import Function, dS_v, ds_v, conditional, avg
from pyop2.profiling import timed_region, timed_function, timed_stage  # NOQA
from pyop2 import op2
import gadopt


class SqueezedDQ1Filter:
    """
    Filter that acts on squashed quads (wedges?)

    In those cells in an extruded mesh that have a vertical facet that is not an external
    boundary facet or an interior facet (i.e. they are not in dS_v or ds_v), the DQ1 values
    of nodes on that facet are averaged, and all set to that average value. In 2D, assuming
    that facet has been squeezed to length 0, this results in a DQ1 solution in the resulting
    triangle that is linear.
    """
    def __init__(self, dq1_space):
        v = TestFunction(dq1_space)
        # nodes on squeezed facets are not included in dS_v or ds_v
        # and all other nodes are (for DQ1), so this step gives nonzero for these other nodes
        self.marker = assemble(avg(v)*dS_v + v*ds_v)
        # flip this: 1 for squeezed nodes, and 0 for all others
        self.marker.assign(conditional(self.marker > 1e-12, 0, 1))

        self.P0 = FunctionSpace(dq1_space.mesh(), "DG", 0)
        self.u0 = Function(self.P0, name='averaged squeezed values')
        self.u1 = Function(dq1_space, name='aux. squeezed values')

    def apply(self, u):
        self.u1.interpolate(self.marker*u)
        self.u0.interpolate(2*self.u1)
        self.u1.interpolate(self.u0)
        u.assign(self.marker*self.u1 + (1-self.marker)*u)


class VertexBasedP1DGLimiter(gadopt.VertexBasedP1DGLimiter):
    """
    Vertex based limiter for P1DG tracer fields, see Kuzmin (2010)

    .. note::
        Currently only scalar fields are supported

    Kuzmin (2010). A vertex-based hierarchical slope limiter
    for p-adaptive discontinuous Galerkin methods. Journal of Computational
    and Applied Mathematics, 233(12):3077-3085.
    http://dx.doi.org/10.1016/j.cam.2009.05.028
    """
    def __init__(self, p1dg_space, squeezed_triangles=False):
        """
        :arg p1dg_space: P1DG function space
        :arg squeezed_triangles: whether to deal with quads with squeezed vertical edges
        """

        super().__init__(p1dg_space)
        self.squeezed_triangles = squeezed_triangles
        if squeezed_triangles:
            self.squeezed_filter = SqueezedDQ1Filter(p1dg_space)

    def compute_bounds(self, field):
        """
        Re-compute min/max values of all neighbouring centroids

        :arg field: :class:`Function` to limit
        """
        super().compute_bounds(field)
        if self.squeezed_triangles:
            code = """
                void my_kernel(double *qmax, double *qmin, double *marker) {
                    float min_val, max_val;
                    for (int i=0; i<%(nnodes)d; i++) {
                        if (marker[i] > 0) {
                            max_val = qmax[i];
                            min_val = qmin[i];
                            break;
                        }
                    }
                    for (int i=i+1; i<%(nnodes)d; i++) {
                        if (marker[i] > 0) {
                            max_val = fmax(qmax[i], max_val);
                            min_val = fmin(qmin[i], min_val);
                        }
                    }
                    for (int i=0; i<%(nnodes)d; i++) {
                        if (marker[i] > 0) {
                            qmax[i] = max_val;
                            qmin[i] = min_val;
                        }
                    }
                }"""
            cnode_map = self.min_field.function_space().cell_node_map()
            kernel = op2.Kernel(code % {'nnodes': cnode_map.shape[1]}, 'my_kernel')

            marker = self.squeezed_filter.marker

            # NOTE: for multiple squeezed triangle on top (e.g. ice front!) this currently only
            # works at the top, under the assumption that cells are iterated
            # over in each column bottom to top:
            op2.par_loop(kernel, self.mesh.cell_set,
                         self.max_field.dat(op2.MAX, self.max_field.function_space().cell_node_map()),
                         self.min_field.dat(op2.MIN, self.min_field.function_space().cell_node_map()),
                         marker.dat(op2.READ, marker.function_space().cell_node_map()))

    def apply(self, field):
        """
        Applies the limiter on the given field (in place)

        :arg field: :class:`Function` to limit
        """
        with timed_stage('limiter'):
            if self.squeezed_triangles:
                self.squeezed_filter.apply(field)
        super().apply(field)
