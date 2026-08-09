"""Fixed-mesh verification of the combined pressure/free-surface equation.

The test evolves half a period of a small-amplitude, two-dimensional standing
wave in a closed rectangular tank.

For a tank with depth H and horizontal wavenumber k, linear wave theory gives

    omega**2 = g*k*tanh(k*H).

The backward-Euler test ends at half a period, where the free-surface
elevation should have reversed sign and the velocity should be zero again.
The Crank-Nicolson test ends at three eighths of a period, where both
elevation and velocity are non-zero and therefore exhibit the expected 
second-order error.
"""
import pytest

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

ASSEMBLED_SCHUR_SOLVER_PARAMETERS = {
        'snes_type': 'ksponly',
        'snes_monitor': None,
        'ksp_type': 'preonly',  # we solve the full schur complement exactly, so no need for outer krylov
        'mat_type': 'matfree',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_fact_type': 'full',
        # velocity mass block:
        'fieldsplit_0': {
            'ksp_type': 'gmres',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'ksp_converged_reason': None,
            'assembled_ksp_type': 'preonly',
            'assembled_pc_type': 'bjacobi',
            'assembled_sub_pc_type': 'ilu',
            },
        # schur system: explicitly assemble the schur system
        # this only works with pressureprojectionicard if the velocity block is just the mass matrix
        # and if the velocity is DG so that this mass matrix can be inverted explicitly
        'fieldsplit_1': {
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'thwaites.AssembledSchurPC',
            'schur_ksp_type': 'cg',
            'schur_ksp_max_it': 1000,
            'schur_ksp_rtol': 1e-8,
            'schur_ksp_atol': 1e-10,
            'schur_ksp_converged_reason': None,
            'schur_pc_type': 'gamg',
            'schur_pc_gamg_threshold': 0.01
            },
        }


def run_standing_wave(
    nx=16,
    nz=2,
    steps_per_period=128,
    solver_parameters=None,
    predictor_solver_parameters=None,
    picard_iterations=1,
    theta=1,
    period_fraction=0.5,
    ):
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

    requested_steps = steps_per_period*period_fraction
    number_of_steps = int(round(requested_steps))
    if not np.isclose(requested_steps, number_of_steps):
        raise ValueError(
            "steps_per_period*period_fraction must be an integer"
        )
    time = 0.0
    
    for _ in range(number_of_steps):
        timestepper.advance(time)
        time += dt

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
    PETSc.Sys.Print("fixed-mesh standing-wave diagnostics:", diagnostics)
    return diagnostics

@pytest.mark.parametrize(
    "solver_parameters",
    [
        pytest.param(DIRECT_SOLVER_PARAMETERS, id="direct-lu"),
        pytest.param(
            ASSEMBLED_SCHUR_SOLVER_PARAMETERS,
            id="assembled-schur",
        ),
    ],
)
def test_fixed_mesh_standing_wave(solver_parameters, theta=1):
    steps = [64*2**i for i in range(4)]
    all_diagnostics = []
    for step_count in steps:
        diagnostics = run_standing_wave(steps_per_period=step_count, solver_parameters=solver_parameters, theta=theta)
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


def test_fixed_mesh_standing_wave_picard_invariance(
):
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

if __name__ == "__main__":
    test_fixed_mesh_standing_wave()
