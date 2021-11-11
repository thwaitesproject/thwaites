"""
Slope limiters for discontinuous fields
"""
from __future__ import absolute_import
from firedrake import VertexBasedLimiter, FunctionSpace, TrialFunction, LinearSolver, TestFunction, dx, assemble
from firedrake import Function, dS_v, ds_v, conditional, avg
from firedrake import TensorProductElement
import numpy as np
import ufl
from pyop2.profiling import timed_region, timed_function, timed_stage  # NOQA
from pyop2 import op2


def assert_function_space(fs, family, degree):
    """
    Checks the family and degree of function space.

    Raises AssertionError if function space differs.
    If the function space lies on an extruded mesh, checks both spaces of the
    outer product.

    :arg fs: function space
    :arg string family: name of element family
    :arg int degree: polynomial degree of the function space
    """
    fam_list = family
    if not isinstance(family, list):
        fam_list = [family]
    ufl_elem = fs.ufl_element()
    if isinstance(ufl_elem, ufl.VectorElement):
        ufl_elem = ufl_elem.sub_elements()[0]

    if ufl_elem.family() == 'TensorProductElement':
        # extruded mesh
        A, B = ufl_elem.sub_elements()
        assert A.family() in fam_list,\
            'horizontal space must be one of {0:s}'.format(fam_list)
        assert B.family() in fam_list,\
            'vertical space must be {0:s}'.format(fam_list)
        assert A.degree() == degree,\
            'degree of horizontal space must be {0:d}'.format(degree)
        assert B.degree() == degree,\
            'degree of vertical space must be {0:d}'.format(degree)
    else:
        # assume 2D mesh
        assert ufl_elem.family() in fam_list,\
            'function space must be one of {0:s}'.format(fam_list)
        assert ufl_elem.degree() == degree,\
            'degree of function space must be {0:d}'.format(degree)


def get_extruded_base_element(ufl_element):
    """
    Return UFL TensorProductElement of an extruded UFL element.

    In case of a non-extruded mesh, returns the element itself.
    """
    if isinstance(ufl_element, ufl.HDivElement):
        ufl_element = ufl_element._element
    if isinstance(ufl_element, ufl.MixedElement):
        ufl_element = ufl_element.sub_elements()[0]
    if isinstance(ufl_element, ufl.VectorElement):
        ufl_element = ufl_element.sub_elements()[0]  # take the first component
    if isinstance(ufl_element, ufl.EnrichedElement):
        ufl_element = ufl_element._elements[0]
    return ufl_element


def get_facet_mask(function_space, facet='bottom'):
    """
    Returns the top/bottom nodes of extruded 3D elements.

    :arg function_space: Firedrake :class:`FunctionSpace` object
    :kwarg str facet: 'top' or 'bottom'

    .. note::
        The definition of top/bottom depends on the direction of the extrusion.
        Here we assume that the mesh has been extruded upwards (along positive
        z axis).
    """
    from tsfc.finatinterface import create_element as create_finat_element

    # get base element
    elem = get_extruded_base_element(function_space.ufl_element())
    assert isinstance(elem, TensorProductElement), \
        f'function space must be defined on an extruded 3D mesh: {elem}'
    # figure out number of nodes in sub elements
    h_elt, v_elt = elem.sub_elements()
    nb_nodes_h = create_finat_element(h_elt).space_dimension()
    nb_nodes_v = create_finat_element(v_elt).space_dimension()
    # compute top/bottom facet indices
    # extruded dimension is the inner loop in index
    # on interval elements, the end points are the first two dofs
    offset = 0 if facet == 'bottom' else 1
    indices = np.arange(nb_nodes_h)*nb_nodes_v + offset
    return indices


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


class VertexBasedP1DGLimiter(VertexBasedLimiter):
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

        assert_function_space(p1dg_space, ['Discontinuous Lagrange', 'DQ'], 1)
        self.is_vector = p1dg_space.value_size > 1
        if self.is_vector:
            p1dg_scalar_space = FunctionSpace(p1dg_space.mesh(), 'DG', 1)
            super(VertexBasedP1DGLimiter, self).__init__(p1dg_scalar_space)
        else:
            super(VertexBasedP1DGLimiter, self).__init__(p1dg_space)
        self.mesh = self.P0.mesh()
        self.dim = self.mesh.geometric_dimension()
        self.extruded = hasattr(self.mesh.ufl_cell(), 'sub_cells')
        assert not self.extruded or len(p1dg_space.ufl_element().sub_elements()) > 0, \
            "Extruded mesh requires extruded function space"
        assert not self.extruded or all(e.variant() == 'equispaced' for e in p1dg_space.ufl_element().sub_elements()), \
            "Extruded function space must be equivariant"

        self.squeezed_triangles = squeezed_triangles
        if squeezed_triangles:
            self.squeezed_filter = SqueezedDQ1Filter(p1dg_space)

    def _construct_centroid_solver(self):
        """
        Constructs a linear problem for computing the centroids

        :return: LinearSolver instance
        """
        u = TrialFunction(self.P0)
        v = TestFunction(self.P0)
        self.a_form = u * v * dx
        a = assemble(self.a_form)
        return LinearSolver(a, solver_parameters={'ksp_type': 'preonly',
                                                  'pc_type': 'bjacobi',
                                                  'sub_pc_type': 'ilu'})

    def _update_centroids(self, field):
        """
        Update centroid values
        """
        b = assemble(TestFunction(self.P0) * field * dx)
        self.centroid_solver.solve(self.centroids, b)

    def compute_bounds(self, field):
        """
        Re-compute min/max values of all neighbouring centroids

        :arg field: :class:`Function` to limit
        """
        # Call general-purpose bound computation.
        super(VertexBasedP1DGLimiter, self).compute_bounds(field)

        # Add the average of lateral boundary facets to min/max fields
        # NOTE this just computes the arithmetic mean of nodal values on the facet,
        # which in general is not equivalent to the mean of the field over the bnd facet.
        # This is OK for P1DG triangles, but not exact for the extruded case (quad facets)
        from finat.finiteelementbase import entity_support_dofs

        if self.extruded:
            entity_dim = (self.dim-2, 1)  # get vertical facets
        else:
            entity_dim = self.dim-1
        boundary_dofs = entity_support_dofs(self.P1DG.finat_element, entity_dim)
        local_facet_nodes = np.array([boundary_dofs[e] for e in sorted(boundary_dofs.keys())])
        n_bnd_nodes = local_facet_nodes.shape[1]
        local_facet_idx = op2.Global(local_facet_nodes.shape, local_facet_nodes, dtype=np.int32, name='local_facet_idx')
        code = """
            void my_kernel(double *qmax, double *qmin, double *field, unsigned int *facet, unsigned int *local_facet_idx)
            {
                double face_mean = 0.0;
                for (int i = 0; i < %(nnodes)d; i++) {
                    unsigned int idx = local_facet_idx[facet[0]*%(nnodes)d + i];
                    face_mean += field[idx];
                }
                face_mean /= %(nnodes)d;
                for (int i = 0; i < %(nnodes)d; i++) {
                    unsigned int idx = local_facet_idx[facet[0]*%(nnodes)d + i];
                    qmax[idx] = fmax(qmax[idx], face_mean);
                    qmin[idx] = fmin(qmin[idx], face_mean);
                }
            }"""
        bnd_kernel = op2.Kernel(code % {'nnodes': n_bnd_nodes}, 'my_kernel')
        op2.par_loop(bnd_kernel,
                     self.P1DG.mesh().exterior_facets.set,
                     self.max_field.dat(op2.MAX, self.max_field.exterior_facet_node_map()),
                     self.min_field.dat(op2.MIN, self.min_field.exterior_facet_node_map()),
                     field.dat(op2.READ, field.exterior_facet_node_map()),
                     self.P1DG.mesh().exterior_facets.local_facet_dat(op2.READ),
                     local_facet_idx(op2.READ))
        if self.extruded:
            # Add nodal values from surface/bottom boundaries
            # NOTE calling firedrake par_loop with measure=ds_t raises an error
            bottom_nodes = get_facet_mask(self.P1CG, 'bottom')
            top_nodes = get_facet_mask(self.P1CG, 'top')
            bottom_idx = op2.Global(len(bottom_nodes), bottom_nodes, dtype=np.int32, name='node_idx')
            top_idx = op2.Global(len(top_nodes), top_nodes, dtype=np.int32, name='node_idx')
            code = """
                void my_kernel(double *qmax, double *qmin, double *field, int *idx) {
                    double face_mean = 0;
                    for (int i=0; i<%(nnodes)d; i++) {
                        face_mean += field[idx[i]];
                    }
                    face_mean /= %(nnodes)d;
                    for (int i=0; i<%(nnodes)d; i++) {
                        qmax[idx[i]] = fmax(qmax[idx[i]], face_mean);
                        qmin[idx[i]] = fmin(qmin[idx[i]], face_mean);
                    }
                }"""
            kernel = op2.Kernel(code % {'nnodes': len(bottom_nodes)}, 'my_kernel')

            op2.par_loop(kernel, self.mesh.cell_set,
                         self.max_field.dat(op2.MAX, self.max_field.function_space().cell_node_map()),
                         self.min_field.dat(op2.MIN, self.min_field.function_space().cell_node_map()),
                         field.dat(op2.READ, field.function_space().cell_node_map()),
                         bottom_idx(op2.READ),
                         iterate=op2.ON_BOTTOM)

            op2.par_loop(kernel, self.mesh.cell_set,
                         self.max_field.dat(op2.MAX, self.max_field.function_space().cell_node_map()),
                         self.min_field.dat(op2.MIN, self.min_field.function_space().cell_node_map()),
                         field.dat(op2.READ, field.function_space().cell_node_map()),
                         top_idx(op2.READ),
                         iterate=op2.ON_TOP)
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

            if self.is_vector:
                tmp_func = self.P1DG.get_work_function()
                fs = field.function_space()
                for i in range(fs.value_size):
                    tmp_func.dat.data_with_halos[:] = field.dat.data_with_halos[:, i]
                    super(VertexBasedP1DGLimiter, self).apply(tmp_func)
                    field.dat.data_with_halos[:, i] = tmp_func.dat.data_with_halos[:]
                self.P1DG.restore_work_function(tmp_func)
            else:
                super(VertexBasedP1DGLimiter, self).apply(field)
