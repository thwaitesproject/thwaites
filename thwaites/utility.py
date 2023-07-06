"""
A module with utitity functions for Thwaites
"""
from firedrake import outer, ds_v, ds_t, ds_b, CellDiameter, CellVolume
from firedrake import sqrt, exp, Function, FiniteElement, TensorProductElement, FunctionSpace, VectorFunctionSpace
from firedrake import RW, READ, dx, par_loop, ExtrudedMesh
import ufl
import numpy as np


class CombinedSurfaceMeasure(ufl.Measure):
    """
    A surface measure that combines ds_v, the integral over vertical boundary facets, and ds_t and ds_b,
    the integral over horizontal top and bottom facets. The vertical boundary facets are identified with
    the same surface ids as ds_v. The top and bottom surfaces are identified via the "top" and "bottom" ids."""
    def __init__(self, domain, degree):
        self.ds_v = ds_v(domain=domain, degree=degree)
        self.ds_t = ds_t(domain=domain, degree=degree)
        self.ds_b = ds_b(domain=domain, degree=degree)

    def __call__(self, subdomain_id, **kwargs):
        if subdomain_id == 'top':
            return self.ds_t(**kwargs)
        elif subdomain_id == 'bottom':
            return self.ds_b(**kwargs)
        else:
            return self.ds_v(subdomain_id, **kwargs)

    def __rmul__(self, other):
        """This is to handle terms to be integrated over all surfaces in the form of other*ds.
        Here the CombinedSurfaceMeasure ds is not called, instead we just split it up as below."""
        return other*self.ds_v + other*self.ds_t + other*self.ds_b


def _get_element(ufl_or_element):
    if isinstance(ufl_or_element, ufl.FiniteElementBase):
        return ufl_or_element
    else:
        return ufl_or_element.ufl_element()


def is_continuous(expr):
    elem = _get_element(expr)

    family = elem.family()
    if family == 'Lagrange':
        return True
    elif family == 'Discontinuous Lagrange' or family == 'DQ':
        return False
    elif isinstance(elem, ufl.HCurlElement) or isinstance(elem, ufl.HDivElement):
        return False
    elif family == 'TensorProductElement':
        elem_h, elem_v = elem.sub_elements()
        return is_continuous(elem_h) and is_continuous(elem_v)
    elif family == 'EnrichedElement':
        return all(is_continuous(e) for e in elem._elements)
    else:
        raise NotImplementedError("Unknown finite element family")


def normal_is_continuous(expr):
    elem = _get_element(expr)

    family = elem.family()
    if family == 'Lagrange':
        return True
    elif family == 'Discontinuous Lagrange' or family == 'DQ':
        return False
    elif isinstance(elem, ufl.HCurlElement):
        return False
    elif isinstance(elem, ufl.HDivElement):
        return True
    elif family == 'TensorProductElement':
        elem_h, elem_v = elem.sub_elements()
        return normal_is_continuous(elem_h) and normal_is_continuous(elem_v)
    elif family == 'EnrichedElement':
        return all(normal_is_continuous(e) for e in elem._elements)
    else:
        raise NotImplementedError("Unknown finite element family")


def cell_size(mesh):
    if hasattr(mesh.ufl_cell(), 'sub_cells'):
        return sqrt(CellVolume(mesh))
    else:
        return CellDiameter(mesh)


def cell_edge_integral_ratio(mesh, p):
    r"""
    Ratio C such that \int_f u^2 <= C Area(f)/Volume(e) \int_e u^2
    for facets f, elements e and polynomials u of degree p.

    See eqn. (3.7) ad table 3.1 from Hillewaert's thesis: https://www.researchgate.net/publication/260085826
    and its appendix C for derivation."""
    cell_type = mesh.ufl_cell().cellname()
    if cell_type == "triangle":
        return (p+1)*(p+2)/2.
    elif cell_type == "quadrilateral" or cell_type == "interval * interval":
        return (p+1)**2
    elif cell_type == "triangle * interval":
        return (p+1)**2
    elif cell_type == "quadrilateral * interval":
        # if e is a wedge and f is a triangle: (p+1)**2
        # if e is a wedge and f is a quad: (p+1)*(p+2)/2
        # here we just return the largest of the the two (for p>=0)
        return (p+1)**2
    elif cell_type == "tetrahedron":
        return (p+1)*(p+3)/3
    else:
        raise NotImplementedError("Unknown cell type in mesh: {}".format(cell_type))


def tensor_jump(v, n):
    r"""
    Jump term for vector functions based on the tensor product

    .. math::
        \text{jump}(\mathbf{u}, \mathbf{n}) = (\mathbf{u}^+ \mathbf{n}^+) +
        (\mathbf{u}^- \mathbf{n}^-)

    This is the discrete equivalent of grad(u) as opposed to the
    vectorial UFL jump operator :meth:`ufl.jump` which represents div(u).
    The equivalent of nabla_grad(u) is given by tensor_jump(n, u).
    """
    return outer(v('+'), n('+')) + outer(v('-'), n('-'))


# ice thickness shouldn't be used!!! unless the water depth is not relative to 1km!!!
def ice_thickness(x, x0, y0, x1, y1):
    m = (y1-y0)/(x1-x0)
    return y0 + m*x


def cavity_thickness(x, x0, y0, x1, y1):
    m = (y1-y0)/(x1-x0)
    return y0 + m*x


def get_top_boundary(cavity_length=5000., cavity_height=100., water_depth=1000., n=100):
    dx = cavity_length / float(n)
    shelf_boundary_points = []
    for i in range(n):
        x_i = i * dx
        y_i = cavity_thickness(x_i, 0.0, 2.0, cavity_length, cavity_height) - water_depth - 0.01
        shelf_boundary_points.append([x_i, y_i])

    return shelf_boundary_points


def get_top_surface(cavity_xlength=5000., cavity_ylength=5000., cavity_height=100., water_depth=1000., dx=500.0, dy=500.):
    nx = round(cavity_xlength / (0.5*dx)) + 1
    ny = round(cavity_ylength / (0.5*dy)) + 1
    shelf_boundary_points = []
    for i in range(nx):
        x_i = i * dx * 0.5
        for j in range(ny):
            y_i = j * dy * 0.5
            z_i = cavity_thickness(y_i, 0.0, 2.0, cavity_ylength, cavity_height) - water_depth - 0.01
            shelf_boundary_points.append([x_i, y_i, z_i])

    return shelf_boundary_points


def offset_backward_step_approx(x, k=1.0, x0=0.0):
    return 1.0 / (1.0 + exp(2.0*k*(x-x0)))


def extend_function_to_3d(func, mesh_extruded):
    """
    Returns a 3D view of a 2D :class:`Function` on the extruded domain.
    The 3D function resides in V x R function space, where V is the function
    space of the source function. The 3D function shares the data of the 2D
    function.
    """
    fs = func.function_space()
#    assert fs.mesh().geometric_dimension() == 2, 'Function must be in 2D space'
    ufl_elem = fs.ufl_element()
    family = ufl_elem.family()
    degree = ufl_elem.degree()
    name = func.name()
    if isinstance(ufl_elem, ufl.VectorElement):
        # vector function space
        fs_extended = get_functionspace(mesh_extruded, family, degree, 'R', 0, dim=2, vector=True)
    else:
        fs_extended = get_functionspace(mesh_extruded, family, degree, 'R', 0)
    func_extended = Function(fs_extended, name=name, val=func.dat._data)
    func_extended.source = func
    return func_extended


class ExtrudedFunction(Function):
    """
    A 2D :class:`Function` that provides a 3D view on the extruded domain.
    The 3D function can be accessed as `ExtrudedFunction.view_3d`.
    The 3D function resides in V x R function space, where V is the function
    space of the source function. The 3D function shares the data of the 2D
    function."""
    def __init__(self, *args, mesh_3d=None, **kwargs):
        """
        Create a 2D :class:`Function` with a 3D view on extruded mesh.
        :arg mesh_3d: Extruded 3D mesh where the function will be extended to.
        """
        # create the 2d function
        super().__init__(*args, **kwargs)
        print(*args)
        if mesh_3d is not None:
            self.view_3d = extend_function_to_3d(self, mesh_3d)


def get_functionspace(mesh, h_family, h_degree, v_family=None, v_degree=None,
                      vector=False, hdiv=False, variant=None, v_variant=None,
                      **kwargs):
    cell_dim = mesh.cell_dimension()
    print(cell_dim)
    assert cell_dim in [2, (2, 1), (1, 1)], 'Unsupported cell dimension'
    hdiv_families = [
        'RT', 'RTF', 'RTCF', 'RAVIART-THOMAS',
        'BDM', 'BDMF', 'BDMCF', 'BREZZI-DOUGLAS-MARINI',
    ]
    if variant is None:
        if h_family.upper() in hdiv_families:
            if h_family in ['RTCF', 'BDMCF']:
                variant = 'equispaced'
            else:
                variant = 'integral'
        else:
            print("var = equi")
            variant = 'equispaced'
    if v_variant is None:
        v_variant = 'equispaced'
    if cell_dim == (2, 1) or (1, 1):
        if v_family is None:
            v_family = h_family
        if v_degree is None:
            v_degree = h_degree
        h_cell, v_cell = mesh.ufl_cell().sub_cells()
        h_elt = FiniteElement(h_family, h_cell, h_degree, variant=variant)
        v_elt = FiniteElement(v_family, v_cell, v_degree, variant=v_variant)
        elt = TensorProductElement(h_elt, v_elt)
        if hdiv:
            elt = ufl.HDiv(elt)
    else:
        elt = FiniteElement(h_family, mesh.ufl_cell(), h_degree, variant=variant)

    constructor = VectorFunctionSpace if vector else FunctionSpace
    return constructor(mesh, elt, **kwargs)


class FrazilRisingVelocity:
    epsilon = 0.0625  # aspect ratio of frazil disk = 1/16
    rho_ice = 920.0  # density of ice / kg /m^3
    rho_seawater = 1030.0  # density of seawater / kg/m^2
    R = -1.0 * (rho_ice - rho_seawater) / rho_seawater
    g = 9.81  # gravitational constant / m/s^2
    nu = 1.95e-6  # kinematic viscosity of seawater / m^2/s

    def __init__(self, w_i, r=7.5e-4, picard_steps=10):
        self.r = r
        self.w_i = w_i
        assert self.w_i > 0  # Initial velocity guess needs to be greater than zero.
        self.w_i_old = self.w_i
        self.picard_steps = picard_steps

    def calculate_rising_velocity(self, Cd):
        return pow(4 * self.R * self.g * self.r * self.epsilon / Cd, 0.5)

    def calculate_drag_coefficient(self, Re_disk):
        log10_Re_disk = np.log10(Re_disk)
        log10_Cd = 1.386 - 0.892*log10_Re_disk + 0.111*pow(log10_Re_disk, 2)
        return pow(10, log10_Cd)

    def calculate_Re_disk(self, w_i):
        return w_i * 2 * self.r / self.nu

    def picard_step(self):
        self.w_i_old = (self.w_i)
        Re_disk_update = self.calculate_Re_disk(self.w_i)
        Cd_update = self.calculate_drag_coefficient(Re_disk_update)
        self.w_i = self.calculate_rising_velocity(Cd_update)

    def frazil_rising_velocity(self):
        step = 0
        print(step)
        w_i_change = 1
        print(w_i_change)
        while w_i_change > 1e-6:
            print("hello world")
            self.picard_step()
            w_i_change = abs(self.w_i - self.w_i_old)
            step += 1
            print("Step ", step, ": wi old = ", self.w_i_old, " -> wi new = ", self.w_i)
            print("Step ", step, ": change in frazil ice rising velocity =", w_i_change)
            if step > self.picard_steps:
                break
        return self.w_i


def extruded_cavity_mesh(base_mesh, ocean_thickness, dz, layers):
    P0dg = FunctionSpace(base_mesh, "DG", 0)
    P0dg_cells = Function(P0dg)
    tmp = ocean_thickness.copy(deepcopy=True)
    P0dg_cells.assign(np.finfo(0.).min)
    domain = "{[i]: 0 <= i < bathy.dofs}"
    instructions = "bathy_max[0] = fmax(bathy[i], bathy_max[0])"
    keys = {'bathy_max': (P0dg_cells, RW), 'bathy': (tmp, READ)}
    par_loop((domain, instructions), dx, keys)

    P0dg_cells /= dz

    P0dg_cells_array = P0dg_cells.dat.data_ro_with_halos[:]

    for i in P0dg_cells_array:
        layers.append([0, i])

    mesh = ExtrudedMesh(base_mesh, layers, layer_height=dz)
    return mesh
