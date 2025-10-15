import numpy as np
from pathlib import Path

base = Path(__file__).parent.resolve()


def test_geostrophic_spinup_3d():
    '''Checks that max/min limits of velocity components is consistent 
    after 4 days, at which point the simulation is in geostrophic balance'''

    expected_vel_lims = np.loadtxt(base / "expected_geostrophic_spinup_3d_test.log")
    vel_lims = np.loadtxt(base / "geostrophic_spinup_3d_test.log")

    assert np.allclose(expected_vel_lims, vel_lims, rtol=1e-6, atol=1e-16)
