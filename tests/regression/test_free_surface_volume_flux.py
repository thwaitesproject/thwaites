"""Closed-tank freshwater input, pressure offsets, and volume balance.

The tests cover both theta values and correction solvers on a fixed mesh,
including uniform and spatially varying sources and explicit-zero equivalence.
"""
import numpy as np
import pytest

from firedrake.petsc import PETSc

from thwaites import Constant, Function, SpatialCoordinate, assemble, cos, inner

from free_surface_helpers import (
    SOLVER_CASES,
    SurfaceVolumeBalance,
    run_standing_wave,
    setup_closed_tank,
)


@pytest.mark.parametrize("solver_parameters", SOLVER_CASES)
@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_fixed_mesh_constant_surface_offset(theta, solver_parameters):
    """A nonzero constant pressure is a stationary physical surface mode."""
    length = 10_000.0
    depth = 100.0
    gravity = 9.81
    eta0 = 0.25
    dt = 10.0
    solution, timestepper, momentum_equation = setup_closed_tank(
        16, 2, length, depth, dt, gravity,
        solver_parameters=solver_parameters, theta=theta,
    )
    velocity, pressure = solution.subfunctions
    velocity.assign(0.0)
    pressure.assign(gravity*eta0)
    mesh = velocity.function_space().mesh()
    initial_coordinates = mesh.coordinates.dat.data_ro.copy()
    surface_measure = timestepper.equations[1].ds(4)
    volume_measure = momentum_equation.dx
    volume_balance = SurfaceVolumeBalance(pressure, gravity, surface_measure, dt)
    diagnostics = {
        "theta": theta,
        "max_surface_rms_error": 0.0,
        "max_mean_surface_drift": 0.0,
        "max_velocity_rms": 0.0,
        "max_velocity_dof": 0.0,
        "coordinate_change": 0.0,
    }
    for step in range(20):
        timestepper.advance(step*dt)
        volume_balance.check()
        eta = pressure/gravity
        values = {
            "max_surface_rms_error": np.sqrt(assemble(
                (eta-eta0)**2*surface_measure
            )/length),
            "max_mean_surface_drift": abs(
                assemble(eta*surface_measure)/length - eta0
            ),
            "max_velocity_rms": np.sqrt(assemble(
                inner(velocity, velocity)*volume_measure
            )/(length*depth)),
            "max_velocity_dof": np.max(np.abs(velocity.dat.data_ro)),
            "coordinate_change": np.max(np.abs(
                mesh.coordinates.dat.data_ro-initial_coordinates
            )),
        }
        for key, value in values.items():
            assert np.isfinite(value), (key, step, value)
            diagnostics[key] = max(diagnostics[key], float(value))

    diagnostics.update(volume_balance.diagnostics())
    PETSc.Sys.Print("fixed-mesh constant-offset diagnostics:", diagnostics)
    # Observed errors are below 1e-16 m and 6e-15 m/s with both solvers;
    # leave headroom for roundoff in this exactly stationary state.
    assert diagnostics["max_surface_rms_error"] < 1.0e-12, diagnostics
    assert diagnostics["max_mean_surface_drift"] < 1.0e-12, diagnostics
    assert diagnostics["max_velocity_rms"] < 1.0e-12, diagnostics
    assert diagnostics["max_velocity_dof"] < 1.0e-12, diagnostics
    assert diagnostics["coordinate_change"] == 0.0, diagnostics


@pytest.mark.parametrize("solver_parameters", SOLVER_CASES)
@pytest.mark.parametrize("theta", [1.0, 0.5])
@pytest.mark.parametrize("forcing", ["constant", "time-dependent", "spatial"])
def test_fixed_mesh_freshwater_volume_flux(forcing, theta, solver_parameters):
    """Positive input adds volume; variable input can also excite a wave."""
    length = 10_000.0
    depth = 100.0
    gravity = 9.81
    eta0 = 0.25
    dt = 10.0
    steps = 20
    rate = 1.0e-4  # m/s; 0.02 m added over the constant-flux run.
    solution, timestepper, momentum_equation = setup_closed_tank(
        16, 2, length, depth, dt, gravity,
        solver_parameters=solver_parameters, theta=theta,
        # Exercise the fixed beginning-of-step pressure with repeated solves.
        picard_iterations=2 if forcing == "time-dependent" else 1,
    )
    velocity, pressure = solution.subfunctions
    velocity.assign(0.0)
    pressure.assign(gravity*eta0)
    mesh = velocity.function_space().mesh()
    initial_coordinates = mesh.coordinates.dat.data_ro.copy()
    x = SpatialCoordinate(mesh)[0]
    surface_measure = timestepper.equations[1].ds(4)
    volume_measure = momentum_equation.dx
    volume_balance = SurfaceVolumeBalance(pressure, gravity, surface_measure, dt)
    surface_mode = cos(np.pi*x/length)
    old_rate = Constant(0.0)
    new_rate = Constant(0.0)
    surface_bc = timestepper.bcs[4]
    if forcing == "constant":
        # A Python scalar with no old endpoint means constant over the step.
        surface_bc["freshwater_flux"] = rate
    elif forcing == "time-dependent":
        # Exercise Function and Constant endpoints independently of the solver
        # pressure, with deliberately incorrect initial data overwritten below.
        new_flux = Function(pressure.function_space()).assign(-123.0)
        surface_bc["freshwater_flux"] = new_flux
        surface_bc["freshwater_flux_old"] = old_rate
    else:
        # Nonuniform, time-dependent flux; its cosine has zero mean.
        surface_bc["freshwater_flux"] = new_rate*(1+0.5*surface_mode)
        surface_bc["freshwater_flux_old"] = old_rate*(1+0.5*surface_mode)

    def prescribed_rate(time):
        if forcing == "constant":
            return rate
        # Quadratic forcing distinguishes endpoint theta weighting from simply
        # evaluating at t+theta*dt. It changes sign during the run.
        scaled_time = time/(steps*dt)
        return rate*(1-3*scaled_time+scaled_time**2)

    expected_eta = eta0
    # At Schur rtol=1e-8/atol=1e-10, uniform-input errors are below 1.1e-11 m
    # and 2.8e-11 m/s; spatial-input mean errors are below 2.4e-12 m.
    # Tightening only the diagnostic solve to rtol=1e-12 reduces these to
    # roundoff. Retain the original solver settings and allow solver noise.
    state_tolerance = 1.0e-10
    mean_tolerance = 1.0e-11
    diagnostics = {
        "forcing": forcing,
        "theta": theta,
        "max_surface_rms_error": 0.0,
        "max_mean_surface_error": 0.0,
        "max_surface_anomaly_rms": 0.0,
        "max_velocity_rms": 0.0,
        "max_velocity_dof": 0.0,
        "coordinate_change": 0.0,
    }
    for step in range(steps):
        time = step*dt
        rate_old = prescribed_rate(time)
        rate_new = prescribed_rate(time+dt)
        old_rate.assign(rate_old)
        new_rate.assign(rate_new)
        if forcing == "time-dependent":
            new_flux.assign(rate_new)
        rate_theta = (1-theta)*rate_old + theta*rate_new
        flux_theta = rate_theta
        if forcing == "spatial":
            flux_theta *= 1+0.5*surface_mode
        timestepper.advance(time)
        volume_balance.check(flux_theta)
        expected_eta += dt*rate_theta
        eta = pressure/gravity
        mean_eta = assemble(eta*surface_measure)/length
        values = {
            "max_mean_surface_error": abs(mean_eta-expected_eta),
            "max_surface_anomaly_rms": np.sqrt(assemble(
                (eta-mean_eta)**2*surface_measure
            )/length),
            "max_velocity_rms": np.sqrt(assemble(
                inner(velocity, velocity)*volume_measure
            )/(length*depth)),
            "max_velocity_dof": np.max(np.abs(velocity.dat.data_ro)),
            "coordinate_change": np.max(np.abs(
                mesh.coordinates.dat.data_ro-initial_coordinates
            )),
        }
        if forcing != "spatial":
            values["max_surface_rms_error"] = np.sqrt(assemble(
                (eta-expected_eta)**2*surface_measure
            )/length)
        for key, value in values.items():
            assert np.isfinite(value), (key, step, value)
            diagnostics[key] = max(diagnostics[key], float(value))
        assert diagnostics["max_mean_surface_error"] < mean_tolerance, diagnostics
        assert diagnostics["coordinate_change"] == 0.0, diagnostics
        if forcing != "spatial":
            assert diagnostics["max_surface_rms_error"] < state_tolerance, diagnostics
            assert diagnostics["max_velocity_rms"] < state_tolerance, diagnostics
            assert diagnostics["max_velocity_dof"] < state_tolerance, diagnostics

    diagnostics.update(volume_balance.diagnostics())
    diagnostics["final_mean_elevation"] = float(mean_eta)
    diagnostics["expected_mean_elevation"] = float(expected_eta)
    PETSc.Sys.Print("fixed-mesh freshwater diagnostics:", diagnostics)
    if forcing == "spatial":
        assert diagnostics["max_surface_anomaly_rms"] > 1.0e-4, diagnostics
        assert diagnostics["max_velocity_rms"] > 1.0e-5, diagnostics


@pytest.mark.parametrize("solver_parameters", SOLVER_CASES)
@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_fixed_mesh_zero_freshwater_flux(theta, solver_parameters):
    """Explicit zero input reproduces the omitted-flux standing wave."""
    unforced = run_standing_wave(theta=theta, solver_parameters=solver_parameters)
    zero_flux = run_standing_wave(
        theta=theta, solver_parameters=solver_parameters,
        freshwater_flux=Constant(0.0),
    )
    for key in ["surface_l2_error", "velocity_error", "mode_amplitude_error",
                "mean_surface_elevation", "coordinate_change"]:
        np.testing.assert_allclose(zero_flux[key], unforced[key],
                                   rtol=1.0e-10, atol=1.0e-12)
