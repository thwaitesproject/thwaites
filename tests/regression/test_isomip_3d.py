import numpy as np
from pathlib import Path

base = Path(__file__).parent.resolve()


def test_isomip_3d():
    '''Checks that integrated melt within cavity is consistent for first 5 timesteps'''

    expected_melt = np.loadtxt(base / "expected_isomip_3d_melt_test.log")
    melt = np.loadtxt(base / "isomip_3d_melt_test.log")

    # check that norm(q) is the same as previously run
    assert np.allclose(expected_melt, melt, rtol=1e-6, atol=1e-16)
