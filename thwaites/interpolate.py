# This code coems from Icepack c thanks to Daniel Shapero
# It reads in netcdf / geotiff files and overoloads
# firedrake interpolate so that it can access this data...
r"""Functions for interpolating gridded remote sensing data sets to finite
element spaces"""

import numpy as np
import ufl
import firedrake
import rasterio
from scipy.interpolate import RegularGridInterpolator


def _sample(dataset, X, method, y_transect):
    xres = dataset.res[0]
    bounds = dataset.bounds
    if y_transect is None:
        print(y_transect)
        # 2d interpolation
        xmin = max(X[:, 0].min() - 2 * xres, bounds.left)
        xmax = min(X[:, 0].max() + 2 * xres, bounds.right)

        ymin = max(X[:, 1].min() - 2 * xres, bounds.bottom)
        ymax = min(X[:, 1].max() + 2 * xres, bounds.top)
    else:
        # 1d interpolation
        xmin = max(X[:].min() - 2 * xres, bounds.left)
        xmax = min(X[:].max() + 2 * xres, bounds.right)

        ymin = bounds.bottom
        ymax = bounds.top

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
    ys = np.linspace(lower_right[1], upper_left[1], window.height)

    data = np.flipud(dataset.read(indexes=1, window=window, masked=True)).T
    interpolator = RegularGridInterpolator((xs, ys), data, method=method, bounds_error=False, fill_value=None)

    if y_transect is None:
        return interpolator(X, method=method)
    else:
        assert isinstance(y_transect, (int, float))
        # y_transect should be the fixed y value that the transect passes through
        X_1d = []
        for i in X:
            X_1d.append([i, y_transect])
        X_1d = np.array(X_1d)

        return interpolator(X_1d, method=method)


def interpolate(f, Q, method='linear', y_transect=None):
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
        q.dat.data[:] = _sample(f, X, method, y_transect)
    elif (isinstance(f, tuple)
          and all(isinstance(fi, rasterio.DatasetReader) for fi in f)):
        for i, fi in enumerate(f):
            q.dat.data[:, i] = _sample(fi, X, method, y_transect)
    else:
        raise ValueError('Argument must be a rasterio data set or a tuple of '
                         'data sets!')

    return q
