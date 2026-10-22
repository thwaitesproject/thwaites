"""Shared closed-tank setup, solver options, and free-surface diagnostics."""
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


class SurfaceVolumeBalance:
    """Check reference volume; default absolute limits suit the 2-D tank.

    Integrals and tolerances have units m^2 per unit width in 2-D and m^3
    in 3-D. Three-dimensional callers should supply their own limits.
    """

    def __init__(self, pressure, gravity, surface_measure, dt, *,
                 step_tolerance=1.0e-8, total_tolerance=1.0e-6):
        self.eta = pressure/gravity
        self.ds = surface_measure
        self.dt = dt
        self.step_tolerance = step_tolerance
        self.total_tolerance = total_tolerance
        self.initial_volume = assemble(self.eta*self.ds)
        self.volume = self.initial_volume
        self.integrated_input = 0.0
        self.max_step_error = 0.0
        self.max_total_error = 0.0

    def check(self, flux_theta=None):
        volume = assemble(self.eta*self.ds)
        # A literal UFL zero form loses its domain in this Firedrake version.
        volume_input = (0.0 if flux_theta is None else
                        self.dt*assemble(flux_theta*self.ds))
        self.integrated_input += volume_input
        step_error = volume-self.volume-volume_input
        total_error = volume-self.initial_volume-self.integrated_input
        assert np.all(np.isfinite([volume, volume_input, step_error, total_error]))
        self.max_step_error = max(self.max_step_error, abs(step_error))
        self.max_total_error = max(self.max_total_error, abs(total_error))
        self.volume = volume
        # The default 2-D limits retain the measured Schur maxima across
        # forced/unforced runs: 7.8e-9 per step and 1.6e-7 cumulative (m^2).
        assert abs(step_error) < self.step_tolerance, self.diagnostics()
        assert abs(total_error) < self.total_tolerance, self.diagnostics()

    def diagnostics(self):
        return {
            "initial_volume": float(self.initial_volume),
            "final_volume": float(self.volume),
            "integrated_volume_input": float(self.integrated_input),
            "max_step_volume_error": float(self.max_step_error),
            "max_total_volume_error": float(self.max_total_error),
        }


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
    freshwater_flux=None,
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
    if freshwater_flux is not None:
        boundary_conditions[4]["freshwater_flux"] = freshwater_flux

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
    freshwater_flux=None,
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
        freshwater_flux=freshwater_flux,
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
    volume_balance = SurfaceVolumeBalance(
        pressure, gravity, energy_surface_measure, dt,
    )
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
        volume_balance.check()
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
        **volume_balance.diagnostics(),
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
