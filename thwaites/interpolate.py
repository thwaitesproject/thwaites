#This code coems from Icepack c thanks to Daniel Shapero
# It reads in netcdf / geotiff files and overoloads 
# firedrake interpolate so that it can access this data...
r"""Functions for interpolating gridded remote sensing data sets to finite
element spaces"""

import numpy as np
import ufl
import firedrake
import rasterio
from scipy.interpolate import RegularGridInterpolator


def _sample(dataset, X, method):

    if True:
        
        xres = dataset.res[0]
        bounds = dataset.bounds
        print("bounds", bounds)
        print(type(bounds))
        print("dataset", dataset)
        xmin = max(X[:].min() - 2 * xres, bounds.left)
        print("X[:].min() - 2 * xres",X[:].min() - 2 * xres)
        print("or")
        print("bounds.left", bounds.left)
        xmax = min(X[:].max() + 2 * xres, bounds.right)


        print("X[:, 0].max() + 2 * xres", X[:].max() + 2 * xres)
        print("or")
        print("bounds.right", bounds.right)

        print(xmin)
        print(xmax)
        ymin = 0.
        ymax = 80000.


  #  xres = dataset.res[0]
 ##   bounds = dataset.bounds
#    print("bounds", bounds)
 #   print(type(bounds))
#    print("dataset", dataset)
#    xmin = max(X[:, 0].min() - 2 * xres, bounds.left)
##    print("X[:, 0].min() - 2 * xres",X[:, 0].min() - 2 * xres)
#    print("or")
#    print("bounds.left", bounds.left)
#    xmax = min(X[:, 0].max() + 2 * xres, bounds.right)
#
#
 #   print("X[:, 0].max() + 2 * xres", X[:, 0].max() + 2 * xres)
 #   print("or")
 #   print("bounds.right", bounds.right)
#
#    ymin = max(X[:, 1].min() - 2 * xres, bounds.bottom)
#    print("X[:, 1].min() - 2 * xres", X[:, 1].min() - 2 * xres)
##    print("or")
#    print("bounds.bottom", bounds.bottom)
#
#    ymax = min(X[:, 1].max() + 2 * xres, bounds.top)
#    print("X[:, 1].max() + 2 * xres", X[:, 1].max() + 2 * xres)
#    print("or")
#    print("bounds.bottom", bounds.top)
#
 #   print("xres", xres)
#    print("xmin", xmin)
#    print("xmax", xmax)
#    print("ymin", ymin)
#    print("ymax", ymax)
#    ymin = 40000.
#    ymax = 40000.

    window = rasterio.windows.from_bounds(
        left=xmin,
        right=xmax,
        bottom=ymin,
        top=ymax,
        width=dataset.width,
        height=dataset.height,
        transform=dataset.transform
    )
    window = window.round_lengths(op='ceil').round_offsets(op='floor')
    transform = rasterio.windows.transform(window, dataset.transform)

    upper_left = transform * (0, 0)
    lower_right = transform * (window.width - 1, window.height - 1)
    xs = np.linspace(upper_left[0], lower_right[0], window.width)
    print("xs", np.shape(xs))
    ys = np.linspace(lower_right[1], upper_left[1], window.height)

    print("ys", ys)
    data = np.flipud(dataset.read(indexes=1, window=window, masked=True)).T

    print("data",data)
    interpolator = RegularGridInterpolator((xs, ys), data, method=method, bounds_error=False, fill_value=None)
    X2 = []
    for i in X:
        X2.append([i, 41000])
    X2 = np.array(X2)
    print(X2)

    int_out = interpolator(X2, method=method)
    print(int_out)
    return interpolator(X2, method=method)


def interpolate(f, Q, method='linear'):
    r"""Interpolate an expression or a gridded data set to a function space

    Parameters
    ----------
    f : rasterio dataset or tuple of rasterio datasets
        The gridded data set for scalar fields or the tuple of gridded data
        sets for each component
    Q : firedrake.FunctionSpace
        The function space where the result will live

    Returns
    -------
    firedrake.Function
        A finite element function defined on `Q` with the same nodal values
        as the data `f`
    """
    if isinstance(f, (ufl.core.expr.Expr, firedrake.Function)):
        return firedrake.interpolate(f, Q)

    mesh = Q.mesh()
    element = Q.ufl_element()
    if len(element.sub_elements()) > 0:
        element = element.sub_elements()[0]

    V = firedrake.VectorFunctionSpace(mesh, element)
    X = firedrake.interpolate(mesh.coordinates, V).dat.data_ro

    q = firedrake.Function(Q)

    if isinstance(f, rasterio.DatasetReader):
        q.dat.data[:] = _sample(f, X, method)
    elif (isinstance(f, tuple) and
          all(isinstance(fi, rasterio.DatasetReader) for fi in f)):
        for i, fi in enumerate(f):
            q.dat.data[:, i] = _sample(fi, X, method)
    else:
        raise ValueError('Argument must be a rasterio data set or a tuple of '
                         'data sets!')

    return q
