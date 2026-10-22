"""Standing-wave and pressure-projection verification on a fixed mesh.

For a tank with depth H and horizontal wavenumber k, linear wave theory gives

    omega**2 = g*k*tanh(k*H).

The backward-Euler test ends at half a period, where the free-surface
elevation should have reversed sign and the velocity should be zero again.
The Crank-Nicolson test ends at three eighths of a period, where both
elevation and velocity are non-zero and therefore exhibit the expected
second-order error.

Several-period runs also measure kinetic plus surface potential energy at
every step, separately from the accumulated standing-wave phase error.
The rigid-lid balance test checks the existing pressure convention.
"""
import numpy as np
import pytest

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
    inner,
)
from thwaites.equations import BaseEquation
from thwaites.momentum_equation import MomentumSourceTerm, PressureGradientTerm

from free_surface_helpers import (
    DIRECT_SOLVER_PARAMETERS,
    SOLVER_CASES,
    run_standing_wave,
)


class LinearForcedMomentumEquation(BaseEquation):
    """Pressure and a body force for the existing fixed-boundary balance."""

    terms = [PressureGradientTerm, MomentumSourceTerm]


@pytest.mark.parametrize("solver_parameters", SOLVER_CASES)
@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_fixed_mesh_standing_wave_energy(theta, solver_parameters):
    """Track damping and phase separately over five complete wave periods."""
    steps_per_period = 64
    periods = 5
    diagnostics = run_standing_wave(
        steps_per_period=steps_per_period,
        period_fraction=periods,
        theta=theta,
        solver_parameters=solver_parameters,
        track_energy=True,
    )
    relative_energy = diagnostics["relative_energy_drift"] + 1.0
    assert len(relative_energy) == periods*steps_per_period + 1
    assert diagnostics["coordinate_change"] == 0.0
    assert abs(diagnostics["mean_surface_elevation"]) < 1.0e-10

    if theta == 0.5:
        # Measured drift is below 4e-14 for both solver configurations.
        assert diagnostics["max_relative_energy_drift"] < 1.0e-10
    else:
        # Permit only roundoff/linear-solver noise in the per-step test.
        assert np.all(np.diff(relative_energy) <= 1.0e-12)
        assert relative_energy[-1] < 0.1

    # For the corresponding theta-method oscillator, |R|^2 is the energy
    # ratio per step, and arg(R) is its phase advance. Comparing both histories
    # distinguishes amplitude loss from the expected accumulated phase lag.
    omega_dt = 2*np.pi/steps_per_period
    amplification = (
        (1+(1-theta)*1j*omega_dt)/(1-theta*1j*omega_dt)
    )
    step_numbers = np.arange(len(relative_energy))
    expected_energy = abs(amplification)**(2*step_numbers)
    expected_phase_error = step_numbers*(np.angle(amplification)-omega_dt)
    energy_error = np.max(np.abs(relative_energy-expected_energy))
    phase_error = np.max(np.abs(
        diagnostics["phase_error"]-expected_phase_error
    ))
    PETSc.Sys.Print("theta-oscillator comparison:", {
        "max_relative_energy_error": float(energy_error),
        "max_phase_error": float(phase_error),
        "expected_final_phase_error": float(expected_phase_error[-1]),
    })
    # The spatially discrete mode differs slightly from the analytical one:
    # observed discrepancies are 8e-7 in energy and 3.3e-5 rad in phase.
    assert energy_error < 1.0e-5, energy_error
    assert phase_error < 1.0e-4, phase_error


@pytest.mark.parametrize("solver_parameters", SOLVER_CASES)
def test_fixed_mesh_standing_wave(solver_parameters):
    steps = [64*2**i for i in range(4)]
    all_diagnostics = []
    for step_count in steps:
        diagnostics = run_standing_wave(
            steps_per_period=step_count,
            solver_parameters=solver_parameters,
        )
        all_diagnostics.append(diagnostics)

        assert diagnostics["coordinate_change"] == 0.0
        assert abs(diagnostics["mean_surface_elevation"]) < 1.0e-10
        assert diagnostics["surface_l2_error"] < 0.15
        assert diagnostics["mode_amplitude_error"] < 0.15
        assert diagnostics["velocity_error"] < 0.15

    surface_errors = np.array([
        diagnostics["surface_l2_error"] for diagnostics in all_diagnostics
    ])
    mode_errors = np.array([
        diagnostics["mode_amplitude_error"]
        for diagnostics in all_diagnostics
    ])
    surface_orders = np.log2(surface_errors[:-1]/surface_errors[1:])
    mode_orders = np.log2(mode_errors[:-1]/mode_errors[1:])

    PETSc.Sys.Print("surface convergence orders:", surface_orders)
    PETSc.Sys.Print("mode-amplitude convergence orders:", mode_orders)

    # Backward Euler should converge at first order as dt is halved.
    assert np.all(np.abs(surface_orders-1.0) < 0.1), surface_orders
    assert np.all(np.abs(mode_orders-1.0) < 0.1), mode_orders


def test_fixed_mesh_standing_wave_picard_invariance():
    one_picard = run_standing_wave(
        picard_iterations=1,
    )
    two_picard = run_standing_wave(
        picard_iterations=2,
    )

    for diagnostic in [
        "surface_l2_error",
        "mode_amplitude_error",
        "velocity_error",
        "mean_surface_elevation",
    ]:
        np.testing.assert_allclose(
            two_picard[diagnostic],
            one_picard[diagnostic],
            rtol=1.0e-8,
            atol=1.0e-12,
        )


def test_fixed_mesh_standing_wave_crank_nicolson():
    # Use a phase at which both pressure and velocity are non-zero; evaluating
    # either variable at one of its extrema would give a superconvergent error
    # and obscure the expected second-order temporal convergence.
    steps = [8*2**i for i in range(4)]
    all_diagnostics = []
    for step_count in steps:
        diagnostics = run_standing_wave(
            nx=32,
            nz=4,
            steps_per_period=step_count,
            theta=0.5,
            period_fraction=3/8,
        )
        all_diagnostics.append(diagnostics)

        assert diagnostics["coordinate_change"] == 0.0
        assert abs(diagnostics["mean_surface_elevation"]) < 1.0e-10
        assert diagnostics["surface_l2_error"] < 0.15
        assert diagnostics["mode_amplitude_error"] < 0.15
        assert diagnostics["velocity_error"] < 0.15

    surface_errors = np.array([
        diagnostics["surface_l2_error"] for diagnostics in all_diagnostics
    ])
    mode_errors = np.array([
        diagnostics["mode_amplitude_error"]
        for diagnostics in all_diagnostics
    ])
    velocity_errors = np.array([
        diagnostics["velocity_error"] for diagnostics in all_diagnostics
    ])
    surface_orders = np.log2(surface_errors[:-1]/surface_errors[1:])
    mode_orders = np.log2(mode_errors[:-1]/mode_errors[1:])
    velocity_orders = np.log2(velocity_errors[:-1]/velocity_errors[1:])

    PETSc.Sys.Print("CN surface convergence orders:", surface_orders)
    PETSc.Sys.Print("CN mode-amplitude convergence orders:", mode_orders)
    PETSc.Sys.Print("CN velocity convergence orders:", velocity_orders)

    assert np.all(np.abs(surface_orders-2.0) < 0.25), surface_orders
    assert np.all(np.abs(mode_orders-2.0) < 0.25), mode_orders
    assert np.all(np.abs(velocity_orders-2.0) < 0.25), velocity_orders


@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_pressure_projection_without_free_surface(theta):
    """A gradient force retains the existing end-of-step pressure convention.

    Changing the force between steps exposes accidental theta weighting of
    pressure in runs without a free-surface boundary. The exact velocity is
    zero, and the top boundary fixes the pressure reference to zero.
    """
    mesh = RectangleMesh(4, 2, 1.0, 1.0)
    velocity_space = VectorFunctionSpace(mesh, "DG", 1)
    pressure_space = FunctionSpace(mesh, "CG", 2)
    mixed_space = MixedFunctionSpace([velocity_space, pressure_space])
    solution = Function(mixed_space)
    velocity, pressure = solution.subfunctions
    momentum_equation = LinearForcedMomentumEquation(
        mixed_space.sub(0), mixed_space.sub(0)
    )
    continuity_equation = ContinuityEquation(
        mixed_space.sub(1), mixed_space.sub(1)
    )
    force = Constant(0.0)
    timestepper = PressureProjectionTimeIntegrator(
        [momentum_equation, continuity_equation],
        solution,
        {"source": as_vector((0.0, force))},
        [{"pressure": 1}, {"velocity": 0}],
        0.1,
        {1: {"un": 0.0}, 2: {"un": 0.0}, 3: {"un": 0.0}},
        solver_parameters=DIRECT_SOLVER_PARAMETERS,
        predictor_solver_parameters=DIRECT_SOLVER_PARAMETERS,
        theta=theta,
    )
    z = SpatialCoordinate(mesh)[1]
    for step, strength in enumerate([1.0, 2.0]):
        force.assign(strength)
        timestepper.advance(0.1*step)
        pressure_error = np.sqrt(assemble(
            (pressure-strength*(z-1.0))**2*momentum_equation.dx
        ))
        velocity_norm = np.sqrt(assemble(
            inner(velocity, velocity)*momentum_equation.dx
        ))
        assert pressure_error < 1.0e-11, pressure_error
        assert velocity_norm < 1.0e-11, velocity_norm
