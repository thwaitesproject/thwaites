"""Fixed-mesh verification of the combined pressure/free-surface equation.

The tests evolve a constant offset at rest and a small-amplitude,
two-dimensional standing wave in a closed rectangular tank.

For a tank with depth H and horizontal wavenumber k, linear wave theory gives

    omega**2 = g*k*tanh(k*H).

The backward-Euler test ends at half a period, where the free-surface
elevation should have reversed sign and the velocity should be zero again.
The Crank-Nicolson test ends at three eighths of a period, where both
elevation and velocity are non-zero and therefore exhibit the expected
second-order error.

Several-period runs also measure kinetic plus surface potential energy at
every step, separately from the accumulated standing-wave phase error.
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
    cos,
    cosh,
    inner,
    sin,
    sinh,
)
from thwaites.equations import BaseEquation
from thwaites.momentum_equation import MomentumSourceTerm, PressureGradientTerm


class LinearFreeSurfaceMomentumEquation(BaseEquation):
    """Linear, inviscid momentum equation used to isolate the surface mode."""

    terms = [PressureGradientTerm]


class LinearForcedMomentumEquation(BaseEquation):
    """Pressure and a body force for the existing fixed-boundary balance."""

    terms = [PressureGradientTerm, MomentumSourceTerm]


DIRECT_SOLVER_PARAMETERS = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_type": "aij",
}

ASSEMBLED_SCHUR_SOLVER_PARAMETERS = {
    'snes_type': 'ksponly',
    'ksp_type': 'preonly',
    'mat_type': 'matfree',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'full',
    # Velocity mass block.
    'fieldsplit_0': {
        'ksp_type': 'gmres',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled_ksp_type': 'preonly',
        'assembled_pc_type': 'bjacobi',
        'assembled_sub_pc_type': 'ilu',
    },
    # DG velocity allows the mass block to be inverted elementwise when
    # explicitly assembling the pressure-projection Schur complement.
    'fieldsplit_1': {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'thwaites.AssembledSchurPC',
        'schur_ksp_type': 'cg',
        'schur_ksp_max_it': 1000,
        'schur_ksp_rtol': 1e-8,
        'schur_ksp_atol': 1e-10,
        'schur_pc_type': 'gamg',
        'schur_pc_gamg_threshold': 0.01,
    },
}

SOLVER_CASES = [
    pytest.param(DIRECT_SOLVER_PARAMETERS, id="direct-lu"),
    pytest.param(ASSEMBLED_SCHUR_SOLVER_PARAMETERS, id="assembled-schur"),
]


def setup_closed_tank(
    nx,
    nz,
    length,
    depth,
    dt,
    gravity,
    solver_parameters=None,
    picard_iterations=1,
    theta=1,
):
    """Build the shared flat, fixed-mesh tank without initialising its state."""
    # RectangleMesh boundary ids are 1/2 for the end walls, 3 for the bottom,
    # and 4 for the top.  Coordinates are shifted to -H <= z <= 0, but never
    # changed after mesh construction.
    mesh = RectangleMesh(nx, nz, length, depth)
    mesh.coordinates.dat.data[:, 1] -= depth
    velocity_space = VectorFunctionSpace(mesh, "DG", 1)
    pressure_space = FunctionSpace(mesh, "CG", 2)
    mixed_space = MixedFunctionSpace([velocity_space, pressure_space])

    solution = Function(mixed_space, name="velocity_pressure")
    velocity, pressure = solution.subfunctions
    velocity.rename("velocity")
    pressure.rename("kinematic_pressure")

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

    if solver_parameters is None:
        solver_parameters = DIRECT_SOLVER_PARAMETERS

    timestepper = PressureProjectionTimeIntegrator(
        [momentum_equation, continuity_equation],
        solution,
        fields,
        coupling,
        dt,
        boundary_conditions,
        solver_parameters=solver_parameters,
        predictor_solver_parameters=DIRECT_SOLVER_PARAMETERS,
        picard_iterations=picard_iterations,
        theta=theta,
        # The surface mass term fixes the pressure reference. A rigid-lid
        # constant nullspace would remove a physically meaningful elevation.
        pressure_nullspace=None,
    )
    return solution, timestepper, momentum_equation


def run_standing_wave(
    nx=16,
    nz=2,
    steps_per_period=128,
    solver_parameters=None,
    picard_iterations=1,
    theta=1,
    period_fraction=0.5,
    track_energy=False,
):
    """Return wave errors and, optionally, energy and phase at every step.

    Energy is per unit reference density (and per unit transverse width in
    this 2-D tank); phase is unwrapped in radians from the two modal amplitudes.
    """
    length = 10_000.0
    depth = 100.0
    amplitude = 0.01
    gravity = 9.81
    wavenumber = np.pi / length
    frequency = np.sqrt(gravity*wavenumber*np.tanh(wavenumber*depth))
    period = 2*np.pi/frequency
    dt = period/steps_per_period

    solution, timestepper, momentum_equation = setup_closed_tank(
        nx, nz, length, depth, dt, gravity,
        solver_parameters, picard_iterations, theta,
    )
    velocity, pressure = solution.subfunctions
    mesh = velocity.function_space().mesh()
    initial_coordinates = mesh.coordinates.dat.data_ro.copy()
    x, z = SpatialCoordinate(mesh)
    vertical_structure = (
        cosh(wavenumber*(z+depth))/np.cosh(wavenumber*depth)
    )
    pressure_initial = (
        amplitude*gravity*cos(wavenumber*x)*vertical_structure
    )
    velocity.assign(0.0)
    pressure.interpolate(pressure_initial)

    top = 4
    surface_measure = momentum_equation.ds(top)
    volume_measure = momentum_equation.dx
    # Match the surface mass operator's quadrature: CG2 pressure squared is
    # degree 4, beyond the DG1 momentum equation's default degree-3 rule.
    energy_surface_measure = timestepper.equations[1].ds(top)
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

    requested_steps = steps_per_period*period_fraction
    number_of_steps = int(round(requested_steps))
    if not np.isclose(requested_steps, number_of_steps):
        raise ValueError(
            "steps_per_period*period_fraction must be an integer"
        )
    time = 0.0

    kinetic_energy = []
    potential_energy = []
    modal_phase = []

    def record_energy_and_phase():
        kinetic_energy.append(0.5*assemble(
            inner(velocity, velocity)*volume_measure
        ))
        potential_energy.append(0.5/gravity*assemble(
            pressure**2*energy_surface_measure
        ))
        # The elevation and velocity projections are cosine and sine
        # quadratures. Their angle measures phase independently of damping.
        elevation_mode = 2.0/(length*amplitude)*assemble(
            pressure/gravity*surface_mode*surface_measure
        )
        velocity_mode = assemble(
            inner(velocity, quarter_period_velocity)*volume_measure
        )/velocity_scale**2
        modal_phase.append(np.arctan2(velocity_mode, elevation_mode))

    if track_energy:
        record_energy_and_phase()
    for _ in range(number_of_steps):
        timestepper.advance(time)
        time += dt
        if track_energy:
            record_energy_and_phase()

    phase = frequency*time
    eta_numerical = pressure/gravity
    eta_exact = amplitude*cos(wavenumber*x)*np.cos(phase)
    velocity_exact = quarter_period_velocity*np.sin(phase)
    eta_error = np.sqrt(assemble(
        (eta_numerical-eta_exact)**2*surface_measure
    ))/eta_scale
    velocity_error = np.sqrt(assemble(
        inner(velocity-velocity_exact, velocity-velocity_exact)*volume_measure
    ))/velocity_scale
    mode_amplitude = (
        2.0/length*assemble(eta_numerical*surface_mode*surface_measure)
    )
    exact_mode_amplitude = amplitude*np.cos(phase)
    mode_amplitude_error = (
        abs(mode_amplitude-exact_mode_amplitude)/amplitude
    )
    mean_surface_elevation = (
        assemble(eta_numerical*surface_measure)/length
    )
    coordinate_change = np.max(np.abs(
        mesh.coordinates.dat.data_ro-initial_coordinates
    ))

    diagnostics = {
        "period": period,
        "dt": dt,
        "final_time": time,
        "theta": theta,
        "surface_l2_error": float(eta_error),
        "velocity_error": float(velocity_error),
        "mode_amplitude_error": float(mode_amplitude_error),
        "mean_surface_elevation": float(mean_surface_elevation),
        "coordinate_change": float(coordinate_change),
    }
    if track_energy:
        kinetic_energy = np.array(kinetic_energy)
        potential_energy = np.array(potential_energy)
        energy = kinetic_energy + potential_energy
        relative_energy_drift = energy/energy[0] - 1.0
        times = dt*np.arange(number_of_steps+1)
        phase_error = np.unwrap(modal_phase) - frequency*times
        diagnostics.update({
            "initial_energy": float(energy[0]),
            "final_relative_energy": float(energy[-1]/energy[0]),
            "max_relative_energy_drift": float(np.max(np.abs(
                relative_energy_drift
            ))),
            "max_relative_energy_increase": float(np.max(
                np.diff(energy)/energy[0]
            )),
            "final_phase_error": float(phase_error[-1]),
            "times": times,
            "kinetic_energy": kinetic_energy,
            "potential_energy": potential_energy,
            "relative_energy_drift": relative_energy_drift,
            "phase_error": phase_error,
        })
    PETSc.Sys.Print("fixed-mesh standing-wave diagnostics:", {
        key: value for key, value in diagnostics.items() if np.ndim(value) == 0
    })
    return diagnostics


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
