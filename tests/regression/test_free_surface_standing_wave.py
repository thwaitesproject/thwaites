"""Fixed-mesh verification of the combined pressure/free-surface equation.

The test evolves half a period of a small-amplitude, two-dimensional standing
wave in a closed rectangular tank.

For a tank with depth H and horizontal wavenumber k, linear wave theory gives

    omega**2 = g*k*tanh(k*H).

At half a period the free-surface elevation should have reversed sign and the
velocity should be zero again.  Backward Euler pressure projection damps the
wave slightly, so the assertions allow a modest time-discretisation error.
"""

import numpy as np

from firedrake.petsc import PETSc

from thwaites import (
    Constant,
    ContinuityEquation,
    Function,
    FunctionSpace,
    MixedFunctionSpace,
    PressureProjectionTimeIntegrator,
    RectangleMesh,
    SpatialCoordinate,
    VectorFunctionSpace,
    as_vector,
    assemble,
    cos,
    cosh,
    inner,
    sin,
    sinh,
)
from thwaites.equations import BaseEquation
from thwaites.momentum_equation import PressureGradientTerm


class LinearFreeSurfaceMomentumEquation(BaseEquation):
    """Linear, inviscid momentum equation used to isolate the surface mode."""

    terms = [PressureGradientTerm]


DIRECT_SOLVER_PARAMETERS = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_type": "aij",
}


def run_standing_wave(nx=16, nz=2, steps_per_period=128):
    """Run half a standing-wave period and return dimensionless diagnostics."""

    length = 10_000.0
    depth = 100.0
    amplitude = 0.01
    gravity = 9.81
    wavenumber = np.pi / length
    frequency = np.sqrt(gravity*wavenumber*np.tanh(wavenumber*depth))
    period = 2*np.pi/frequency
    dt = period/steps_per_period

    # RectangleMesh boundary ids are 1/2 for the end walls, 3 for the bottom,
    # and 4 for the top.  Coordinates are shifted to -H <= z <= 0, but never
    # changed after mesh construction.
    mesh = RectangleMesh(nx, nz, length, depth)
    mesh.coordinates.dat.data[:, 1] -= depth
    initial_coordinates = mesh.coordinates.dat.data_ro.copy()

    velocity_space = VectorFunctionSpace(mesh, "DG", 1)
    pressure_space = FunctionSpace(mesh, "CG", 2)
    mixed_space = MixedFunctionSpace([velocity_space, pressure_space])

    solution = Function(mixed_space, name="velocity_pressure")
    velocity, pressure = solution.subfunctions
    velocity.rename("velocity")
    pressure.rename("kinematic_pressure")

    x, z = SpatialCoordinate(mesh)
    vertical_structure = (
        cosh(wavenumber*(z+depth))/np.cosh(wavenumber*depth)
    )
    pressure_initial = (
        amplitude*gravity*cos(wavenumber*x)*vertical_structure
    )
    velocity.assign(0.0)
    pressure.interpolate(pressure_initial)

    momentum_equation = LinearFreeSurfaceMomentumEquation(
        mixed_space.sub(0), mixed_space.sub(0)
    )
    continuity_equation = ContinuityEquation(
        mixed_space.sub(1), mixed_space.sub(1)
    )

    coupling = [{"pressure": 1}, {"velocity": 0}]
    fields = {"gravity": Constant(gravity)}
    boundary_conditions = {
        1: {"un": 0.0},
        2: {"un": 0.0},
        3: {"un": 0.0},
        4: {"free_surface": None},
    }

    timestepper = PressureProjectionTimeIntegrator(
        [momentum_equation, continuity_equation],
        solution,
        fields,
        coupling,
        dt,
        boundary_conditions,
        solver_parameters=DIRECT_SOLVER_PARAMETERS,
        predictor_solver_parameters=DIRECT_SOLVER_PARAMETERS,
        picard_iterations=1,
    )

    top = 4
    surface_measure = momentum_equation.ds(top)
    volume_measure = momentum_equation.dx
    surface_mode = cos(wavenumber*x)
    eta_scale = amplitude*np.sqrt(length/2)

    # Use the exact quarter-period velocity as a non-zero error scale.
    horizontal_velocity_scale = (
        amplitude*gravity*wavenumber/frequency
        * sin(wavenumber*x)*vertical_structure
    )
    vertical_velocity_scale = (
        -amplitude*gravity*wavenumber/frequency*cos(wavenumber*x)
        * sinh(wavenumber*(z+depth))/np.cosh(wavenumber*depth)
    )
    quarter_period_velocity = as_vector(
        (horizontal_velocity_scale, vertical_velocity_scale)
    )
    velocity_scale = np.sqrt(assemble(
        inner(quarter_period_velocity, quarter_period_velocity)*volume_measure
    ))

    number_of_steps = steps_per_period//2
    time = 0.0
    for _ in range(number_of_steps):
        timestepper.advance(time)
        time += dt

    eta_numerical = pressure/gravity
    eta_exact = -amplitude*cos(wavenumber*x)
    eta_error = np.sqrt(assemble(
        (eta_numerical-eta_exact)**2*surface_measure
    ))/eta_scale
    velocity_error = np.sqrt(assemble(
        inner(velocity, velocity)*volume_measure
    ))/velocity_scale
    mode_amplitude = (
        2.0/length*assemble(eta_numerical*surface_mode*surface_measure)
    )
    mode_amplitude_error = abs(mode_amplitude+amplitude)/amplitude
    mean_surface_elevation = (
        assemble(eta_numerical*surface_measure)/length
    )
    coordinate_change = np.max(np.abs(
        mesh.coordinates.dat.data_ro-initial_coordinates
    ))

    diagnostics = {
        "period": period,
        "dt": dt,
        "surface_l2_error": float(eta_error),
        "velocity_error": float(velocity_error),
        "mode_amplitude_error": float(mode_amplitude_error),
        "mean_surface_elevation": float(mean_surface_elevation),
        "coordinate_change": float(coordinate_change),
    }
    PETSc.Sys.Print("fixed-mesh standing-wave diagnostics:", diagnostics)
    return diagnostics


def test_fixed_mesh_standing_wave():
    diagnostics = run_standing_wave()

    assert diagnostics["coordinate_change"] == 0.0
    assert abs(diagnostics["mean_surface_elevation"]) < 1.0e-10
    assert diagnostics["surface_l2_error"] < 0.15
    assert diagnostics["mode_amplitude_error"] < 0.15
    assert diagnostics["velocity_error"] < 0.15


if __name__ == "__main__":
    test_fixed_mesh_standing_wave()
