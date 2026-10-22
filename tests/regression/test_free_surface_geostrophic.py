"""Linear geostrophic balance in a 3-D periodic, fixed-mesh channel.

Only the production pressure gradient and Coriolis momentum terms are used.
Surface volume integrals are in m^3, with no freshwater input.
"""
import numpy as np
import pytest

from firedrake.petsc import PETSc

from thwaites import (
    Constant,
    ContinuityEquation,
    ExtrudedMesh,
    Function,
    FunctionSpace,
    MixedFunctionSpace,
    PeriodicRectangleMesh,
    PressureProjectionTimeIntegrator,
    SpatialCoordinate,
    TestFunction,
    VectorFunctionSpace,
    as_vector,
    assemble,
    inner,
)
from thwaites.equations import BaseEquation
from thwaites.momentum_equation import CoriolisTerm, PressureGradientTerm

from free_surface_helpers import (
    DIRECT_SOLVER_PARAMETERS,
    SOLVER_CASES,
    SurfaceVolumeBalance,
)


class LinearGeostrophicMomentumEquation(BaseEquation):
    """Inviscid linear momentum, including physical Coriolis acceleration."""

    terms = [PressureGradientTerm, CoriolisTerm]


def run_geostrophic_channel(theta, solver_parameters, imbalanced=False):
    length = 10_000.0
    width = 5_000.0
    depth = 100.0
    gravity = 9.81
    coriolis = -1.0e-4
    slope = 1.0e-6
    eta0 = 0.25
    inertial_period = 2*np.pi/abs(coriolis)
    # Equilibrium tests stationarity, so need not resolve transient waves.
    # The control resolves the approximately 639 s cross-channel wave.
    dt = 10.0 if imbalanced else inertial_period/16
    steps = 32 if imbalanced else 5*16

    # y is periodic; the only lateral boundary IDs are x=0 (1), x=Lx (2).
    base_mesh = PeriodicRectangleMesh(4, 3, length, width, direction="y")
    mesh = ExtrudedMesh(base_mesh, layers=2, layer_height=depth/2)
    mesh.coordinates.dat.data[:, 2] -= depth
    initial_coordinates = mesh.coordinates.dat.data_ro.copy()
    velocity_space = VectorFunctionSpace(mesh, "DG", 1)
    pressure_space = FunctionSpace(mesh, "CG", 2)
    mixed_space = MixedFunctionSpace([velocity_space, pressure_space])
    solution = Function(mixed_space)
    velocity, pressure = solution.subfunctions
    momentum = LinearGeostrophicMomentumEquation(
        mixed_space.sub(0), mixed_space.sub(0),
    )
    continuity = ContinuityEquation(mixed_space.sub(1), mixed_space.sub(1))
    fields = {
        "gravity": Constant(gravity),
        "coriolis_frequency": Constant(coriolis),
    }
    bcs = {
        1: {"un": 0.0},
        2: {"un": 0.0},
        "bottom": {"un": 0.0},
        "top": {"free_surface": None},
    }
    timestepper = PressureProjectionTimeIntegrator(
        [momentum, continuity], solution, fields,
        [{"pressure": 1}, {"velocity": 0}], dt, bcs,
        solver_parameters=solver_parameters,
        predictor_solver_parameters=DIRECT_SOLVER_PARAMETERS,
        theta=theta,
        picard_iterations=1,
        # The constant pressure trace is a physical mean elevation.
        pressure_nullspace=None,
    )
    x = SpatialCoordinate(mesh)[0]
    initial_slope = slope*(1.25 if imbalanced else 1.0)
    eta_initial = eta0 + initial_slope*(x-length/2)
    pressure.interpolate(gravity*eta_initial)
    # CoriolisTerm uses f*k cross u = (-f*v, f*u, 0) on the LHS:
    # grad(p) + f*k cross u = (g*slope - f*v, 0, 0) = 0.
    velocity_initial = as_vector((0.0, gravity*slope/coriolis, 0.0))
    velocity.interpolate(velocity_initial)
    initial_velocity_dofs = velocity.dat.data_ro.copy()

    # Check cancellation of the actual discrete terms, including pressure
    # boundary traces, before exercising the predictor and projection split.
    test = TestFunction(velocity_space)
    pressure_term, coriolis_term = momentum._terms
    pressure_residual = pressure_term.residual(
        test, velocity, velocity, {**fields, "pressure": pressure}, bcs,
    )
    coriolis_residual = coriolis_term.residual(
        test, velocity, velocity, fields, bcs,
    )
    balance = assemble(pressure_residual+coriolis_residual)
    pressure_force = assemble(pressure_residual)
    relative_momentum_residual = float(
        np.linalg.norm(balance.dat.data_ro)
        / np.linalg.norm(pressure_force.dat.data_ro)
    )
    assert np.isfinite(relative_momentum_residual)
    if imbalanced:
        assert 0.1 < relative_momentum_residual < 0.3
    else:
        # Measured 5.5e-13, including cancellation of the mean pressure trace.
        assert relative_momentum_residual < 1.0e-11

    ds = continuity.ds("top")
    dx = momentum.dx
    area = length*width
    domain_volume = area*depth
    # Independent 3-D absolute limits in m^3. Measured equilibrium maxima
    # are 6e-8 per step / 2.8e-7 accumulated; the Schur control reaches
    # 5.6e-6 / 1.6e-5 with the shared rtol=1e-8, atol=1e-10 settings.
    volume_balance = SurfaceVolumeBalance(
        pressure, gravity, ds, dt,
        step_tolerance=1.0e-4 if imbalanced else 1.0e-6,
        total_tolerance=1.0e-3 if imbalanced else 1.0e-5,
    )
    assert abs(volume_balance.initial_volume-eta0*area) < 1.0e-7
    diagnostics = {
        "theta": theta,
        "correction_solver": solver_parameters["pc_type"],
        "imbalanced": imbalanced,
        "inertial_period": inertial_period,
        "dt": dt,
        "steps": steps,
        "final_time": steps*dt,
        "geostrophic_speed": gravity*slope/coriolis,
        "velocity_dofs": velocity_space.dim(),
        "pressure_dofs": pressure_space.dim(),
        "initial_relative_momentum_residual": relative_momentum_residual,
        "max_surface_rms_change": 0.0,
        "max_velocity_rms_change": 0.0,
        "max_cross_channel_rms": 0.0,
        "max_vertical_rms": 0.0,
        "max_velocity_dof_change": 0.0,
        "coordinate_change": 0.0,
    }
    for step in range(steps):
        timestepper.advance(step*dt)
        volume_balance.check()
        velocity_error = velocity-velocity_initial
        values = {
            "max_surface_rms_change": np.sqrt(assemble(
                (pressure/gravity-eta_initial)**2*ds,
            )/area),
            "max_velocity_rms_change": np.sqrt(assemble(
                inner(velocity_error, velocity_error)*dx,
            )/domain_volume),
            "max_cross_channel_rms": np.sqrt(assemble(
                velocity[0]**2*dx,
            )/domain_volume),
            "max_vertical_rms": np.sqrt(assemble(
                velocity[2]**2*dx,
            )/domain_volume),
            "max_velocity_dof_change": np.max(np.abs(
                velocity.dat.data_ro-initial_velocity_dofs,
            )),
            "coordinate_change": np.max(np.abs(
                mesh.coordinates.dat.data_ro-initial_coordinates,
            )),
        }
        assert np.all(np.isfinite(pressure.dat.data_ro)), step
        assert np.all(np.isfinite(velocity.dat.data_ro)), step
        assert np.array_equal(mesh.coordinates.dat.data_ro, initial_coordinates)
        for key, value in values.items():
            assert np.isfinite(value), (step, key, value)
            diagnostics[key] = max(diagnostics[key], float(value))
        if not imbalanced:
            # Observed <1e-14 m surface RMS, <2.4e-12 m/s velocity RMS,
            # and <6.4e-12 m/s coefficient error across both solvers.
            assert values["max_surface_rms_change"] < 1.0e-11, diagnostics
            assert values["max_velocity_rms_change"] < 1.0e-10, diagnostics
            assert values["max_cross_channel_rms"] < 1.0e-10, diagnostics
            assert values["max_vertical_rms"] < 1.0e-10, diagnostics
            assert values["max_velocity_dof_change"] < 1.0e-10, diagnostics

    diagnostics.update(volume_balance.diagnostics())
    PETSc.Sys.Print("fixed-mesh geostrophic diagnostics:", diagnostics)
    return diagnostics


@pytest.mark.parametrize("solver_parameters", SOLVER_CASES)
@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_fixed_mesh_geostrophic_equilibrium(theta, solver_parameters):
    """Nonzero geostrophic flow remains stationary for five inertial periods."""
    run_geostrophic_channel(theta, solver_parameters)


@pytest.mark.parametrize("solver_parameters", SOLVER_CASES)
@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_fixed_mesh_geostrophic_imbalanced_control(theta, solver_parameters):
    """A mean-preserving slope perturbation excites flow rather than freezing."""
    diagnostics = run_geostrophic_channel(
        theta, solver_parameters, imbalanced=True,
    )
    assert diagnostics["max_surface_rms_change"] > 1.0e-4, diagnostics
    assert diagnostics["max_velocity_rms_change"] > 1.0e-5, diagnostics
    assert diagnostics["max_cross_channel_rms"] > 1.0e-5, diagnostics
