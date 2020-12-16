import pytest
from thwaites import VertexBasedP1DGLimiter
from firedrake import *
from math import ceil

_triangular_mesh_2d = UnitSquareMesh(10, 10)
_interval_mesh = UnitIntervalMesh(10)
_extruded_mesh_2d = ExtrudedMesh(_interval_mesh, 10)
_box_mesh_3d = UnitCubeMesh(10, 10, 10)
_extruded_mesh_3d = ExtrudedMesh(_triangular_mesh_2d, 10)

def _create_extruded_cavity_mesh():
    #  Generate mesh
    L = 1
    shelf_length = 0.6
    H1 = 0.2
    H2 = 0.8
    H3 = 1.
    dy = 0.1
    ny = round(L/dy)
    dz = 0.1
    layers = []
    cell = 0
    yr = 0
    min_dz = 0.5*dz # if top cell is thinner than this, merge with cell below
    tiny_dz = 0.01*dz # workaround zero measure facet issue (fd issue #1858)

    # create mesh
    mesh1d = IntervalMesh(ny, L)
    layers = []
    cell = 0
    yr = 0
    min_dz = 0.5*dz # if top cell is thinner than this, merge with cell below
    tiny_dz = 0.01*dz # workaround zero measure facet issue (fd issue #1858)

    for i in range(ny):
        yr += dy  # y of right-node (assumed to be the higher one)
        if yr <= shelf_length:
            height = H1 + yr/shelf_length * (H2-H1)
        else:
            height = H3
        ncells = ceil((height-min_dz)/dz)
        layers.append([0, ncells])

    mesh = ExtrudedMesh(mesh1d, layers, layer_height=dz)

    nlayers_column1 = layers[0][1]
    # top-left corner node should be at least tiny_dz above height of node below
    min_height_corner_node = (nlayers_column1-1)*dz + tiny_dz

    # move top nodes to correct position:
    cfs = mesh.coordinates.function_space()
    x, y = SpatialCoordinate(mesh)
    bc = DirichletBC(cfs, as_vector((x, Max(conditional(x>shelf_length, H3, H1+x/shelf_length * (H2-H1)), min_height_corner_node))), "top")
    bc.apply(mesh.coordinates)

    return mesh

_extruded_variable_layered_mesh_2d = _create_extruded_cavity_mesh()

@pytest.mark.parametrize("mesh", [
    _triangular_mesh_2d,
    _extruded_mesh_2d,
    _extruded_variable_layered_mesh_2d,
    _box_mesh_3d,
    _extruded_mesh_3d,
    ], ids=[
        'triangular mesh 2D',
        'extruded mesh 2D',
        'extruded variable layer mesh 2D',
        'tetrahedral mesh 3D',
        'extruded mesh 3D'
    ])
def test_limit_linear_slope(mesh):
    if mesh.is_piecewise_linear_simplex_domain():
        V = FunctionSpace(mesh, "DG", 1)
    else:
        h_cell, v_cell = mesh.ufl_cell().sub_cells()
        h_ele = FiniteElement("DG", h_cell, 1, variant='equispaced')
        v_ele = FiniteElement("DG", v_cell, 1, variant='equispaced')
        mix_ele = TensorProductElement(h_ele, v_ele)
        V = FunctionSpace(mesh, mix_ele)

    limiter = VertexBasedP1DGLimiter(V, time_dependent_mesh=False)
    # uncomment to compare with standard limiter:
    # limiter = VertexBasedLimiter(V)

    u_before = Function(V)
    u_after = Function(V)

    print()
    xyz = SpatialCoordinate(mesh)
    for i in range(mesh.geometric_dimension()):
        u_before.interpolate(xyz[i])
        u_after.assign(u_before)
        limiter.apply(u_after)
        e = errornorm(u_before, u_after)
        print('Dimension {}: {}'.format(i, e))
        try:
            assert e < 1e-12
        except:
            File('tmp.pvd').write(u_before, u_after)
            raise
