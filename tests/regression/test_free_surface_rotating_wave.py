"""Transient nonhydrostatic rotating waves and pressure-projection splitting.

For u_t-f*v=-p_x, v_t+f*u=0, w_t=-p_z, div(u)=0, put
C=cosh(mu*(z+H))/cosh(mu*H), S=sinh(mu*(z+H))/cosh(mu*H),
P=g*a, and s=omega*t+initial_phase. The signed mode is

    p = g*eta0 + P*cos(k*x)*C*cos(s)
    u = P*k*omega/(omega**2-f**2)*sin(k*x)*C*sin(s)
    v = P*k*f/(omega**2-f**2)*sin(k*x)*C*cos(s)
    w = -P*mu/omega*cos(k*x)*S*sin(s).

Continuity requires mu**2=k**2*omega**2/(omega**2-f**2).
At z=0, p_t/g=w requires omega**2=g*mu*tanh(mu*H).
k=pi/L makes u zero on both walls; S=0 makes w zero at the
bottom. All fields are y-independent and periodic. The mean pressure is
physical. Neither the continuous nor discrete problem has a rigid-lid
pressure nullspace. See the dated continuation note for the discrete split.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from firedrake import (
    TestFunctions, TestFunction, div, grad,
)
from firedrake.petsc import PETSc
from thwaites import (
    Constant, ContinuityEquation, ExtrudedMesh, Function, FunctionSpace,
    MixedFunctionSpace, PeriodicRectangleMesh, PressureProjectionTimeIntegrator,
    SpatialCoordinate, VectorFunctionSpace, as_vector, assemble, cos, cosh,
    inner, sin, sinh,
)
from free_surface_helpers import (
    ASSEMBLED_SCHUR_SOLVER_PARAMETERS, DIRECT_SOLVER_PARAMETERS, SOLVER_CASES,
    SurfaceVolumeBalance,
)
from rotating_wave_helpers import CoupledThetaReference, RotatingMomentumEquation


def rotating_channel(nx=16, nz=4):
    """Shared geometry and mode; f/omega=-sqrt(1-(5/7)**2)."""
    length, width, gravity, amplitude, eta0 = 1000., 500., 9.81, 0.01, 0.25
    depth = 0.5*length/np.pi
    k, mu = np.pi/length, 0.7/depth
    omega = np.sqrt(gravity*mu*np.tanh(mu*depth))
    f = -omega*np.sqrt(1-(k/mu)**2)
    mesh = ExtrudedMesh(
        PeriodicRectangleMesh(nx, 3, length, width, direction="y"),
        layers=nz, layer_height=depth/nz,
    )
    mesh.coordinates.dat.data[:, 2] -= depth
    V = VectorFunctionSpace(mesh, "DG", 1)
    Q = FunctionSpace(mesh, "CG", 2)
    W = MixedFunctionSpace([V, Q])
    momentum = RotatingMomentumEquation(W.sub(0), W.sub(0))
    continuity = ContinuityEquation(W.sub(1), W.sub(1))
    x, _, z = SpatialCoordinate(mesh)
    C = cosh(mu*(z+depth))/np.cosh(mu*depth)
    S = sinh(mu*(z+depth))/np.cosh(mu*depth)
    P = gravity*amplitude
    shapes = as_vector((P*k*omega/(omega**2-f**2)*sin(k*x)*C,
                        P*k*f/(omega**2-f**2)*sin(k*x)*C,
                        -P*mu/omega*cos(k*x)*S))
    # Starting away from extrema exposes all three component errors.
    phase0 = np.pi/7
    return SimpleNamespace(
        mesh=mesh, W=W, momentum=momentum, continuity=continuity,
        fields={"gravity": Constant(gravity), "coriolis_frequency": Constant(f)},
        bcs={1: {"un": 0.}, 2: {"un": 0.}, "bottom": {"un": 0.},
             "top": {"free_surface": None}},
        dx=momentum.dx, ds=continuity.ds("top"), area=length*width,
        gravity=gravity, eta0=eta0, amplitude=amplitude, omega=omega, f=f,
        period=2*np.pi/omega, phase0=phase0, shapes=shapes,
        pressure_shape=P*cos(k*x)*C, surface_shape=cos(k*x),
        coordinates=mesh.coordinates.dat.data_ro.copy(),
    )


def initial_state(c):
    state = Function(c.W)
    u, p = state.subfunctions
    p.interpolate(c.gravity*c.eta0+c.pressure_shape*np.cos(c.phase0))
    u.project(as_vector((c.shapes[0]*np.sin(c.phase0),
                         c.shapes[1]*np.cos(c.phase0),
                         c.shapes[2]*np.sin(c.phase0))),
              solver_parameters=DIRECT_SOLVER_PARAMETERS)
    return state


def integrator(c, state, dt, theta, picard, solver_parameters):
    if picard == 0:
        return CoupledThetaReference(c, state, dt, theta)
    # Exercise the production default itself in the single-iteration runs.
    options = {} if picard == 1 else {"picard_iterations": picard}
    return PressureProjectionTimeIntegrator(
        [c.momentum, c.continuity], state, c.fields,
        [{"pressure": 1}, {"velocity": 0}], dt, c.bcs,
        theta=theta, pressure_nullspace=None, **options,
        solver_parameters=solver_parameters,
        predictor_solver_parameters=DIRECT_SOLVER_PARAMETERS,
    )


def energy(c, state):
    u, p = state.subfunctions
    return 0.5*assemble(inner(u, u)*c.dx) + 0.5/c.gravity*assemble(
        (p-c.gravity*c.eta0)**2*c.ds,
    )


def errors(c, state, time, reference=None):
    u, p = state.subfunctions
    phase = c.phase0+c.omega*time
    if reference is None:
        pe = c.gravity*c.eta0+c.pressure_shape*np.cos(phase)
        ue = as_vector((c.shapes[0]*np.sin(phase),
                        c.shapes[1]*np.cos(phase),
                        c.shapes[2]*np.sin(phase)))
    else:
        ue, pe = reference.subfunctions
    values = [np.sqrt(assemble((p-pe)**2*c.ds)
                      / (c.gravity*c.amplitude)**2/(c.area/2))]
    values.extend(np.sqrt(assemble((u[i]-ue[i])**2*c.dx)
                          / assemble(c.shapes[i]**2*c.dx)) for i in range(3))
    # Pressure and cross-channel velocity are independent cosine/sine
    # quadratures, including the sign of each derived spatial shape.
    cosine = assemble((p/c.gravity-c.eta0)*c.surface_shape*c.ds)
    cosine /= assemble(c.amplitude*c.surface_shape**2*c.ds)
    sine = assemble(u[0]*c.shapes[0]*c.dx)/assemble(c.shapes[0]**2*c.dx)
    modal = complex(cosine, sine)*np.exp(-1j*phase)
    values.extend([abs(abs(modal)-1), abs(np.angle(modal))])
    assert np.all(np.isfinite(values))
    return np.array(values)


def evolve(c, theta, spp, picard=1, solver_parameters=None, periods=0.375):
    """Check volume, coordinates and finite state at every step, including t=0."""
    if solver_parameters is None:
        solver_parameters = DIRECT_SOLVER_PARAMETERS
    state = initial_state(c)
    dt = c.period/spp
    stepper = integrator(c, state, dt, theta, picard, solver_parameters)
    u, p = state.subfunctions
    assert np.all(np.isfinite(u.dat.data_ro))
    assert np.all(np.isfinite(p.dat.data_ro))
    assert np.array_equal(c.mesh.coordinates.dat.data_ro, c.coordinates)
    balance = SurfaceVolumeBalance(p, c.gravity, c.ds, dt,
                                   step_tolerance=1e-5, total_tolerance=1e-4)
    assert abs(balance.initial_volume-c.eta0*c.area) < 1e-5
    energies = [energy(c, state)]
    max_energy_identity_error = 0.
    max_split_work = 0.
    last_volume = balance.volume
    for step in range(round(periods*spp)):
        last_volume = balance.volume
        stepper.advance(step*dt)
        balance.check()
        assert np.array_equal(c.mesh.coordinates.dat.data_ro, c.coordinates)
        assert np.all(np.isfinite(u.dat.data_ro))
        assert np.all(np.isfinite(p.dat.data_ro))
        energies.append(energy(c, state))
        assert np.isfinite(energies[-1])
        old = stepper.old if picard == 0 else stepper.solution_old
        un, pn = old.subfunctions
        increment = assemble(inner(u-un, u-un)*c.dx) + assemble(
            (p-pn)**2/c.gravity*c.ds,
        )
        defect = energies[-1]-energies[-2]+(theta-0.5)*increment
        work = 0.
        if picard:
            # The correction omits C(u-u*). Its signed work explains the
            # departure from the coupled theta energy identity.
            ut = (1-theta)*un+theta*u
            coriolis = c.momentum._terms[1]
            work = -dt*theta*assemble(coriolis.residual(
                ut, u-stepper.u_star, u, c.fields, c.bcs,
            ))
        assert np.all(np.isfinite([increment, defect, work]))
        max_energy_identity_error = max(max_energy_identity_error,
                                        abs(defect-work)/energies[0])
        max_split_work = max(max_split_work, abs(work)/energies[0])
    time = round(periods*spp)*dt
    # Independently assemble physical residuals at the final step. The
    # omitted term explains the entire momentum residual, not only its work.
    v, q = TestFunctions(c.W)
    ut = (1-theta)*un+theta*u
    pt = (1-theta)*pn+theta*p
    physical_momentum = c.momentum.mass_term(v, u-un)-dt*c.momentum.residual(
        v, ut, ut, {**c.fields, "pressure": pt}, c.bcs,
    )
    omitted = 0
    if picard:
        omitted = -dt*theta*c.momentum._terms[1].residual(
            v, u-stepper.u_star, u, c.fields, c.bcs,
        )
    # Extract the velocity block before measuring its coefficient norm.
    residual = assemble(physical_momentum-omitted).subfunctions[0]
    scale = assemble(c.momentum.mass_term(v, u)).subfunctions[0]
    momentum_identity_error = np.linalg.norm(residual.dat.data_ro)
    momentum_identity_error /= np.linalg.norm(scale.dat.data_ro)
    physical_continuity = -dt*c.continuity.residual(q, p, p, {
        **c.fields, "velocity": ut, "old_pressure": pn,
        "dt": dt, "theta": theta,
    }, c.bcs)
    pressure_residual = assemble(physical_continuity).subfunctions[1]
    constant_test_residual = np.sum(pressure_residual.dat.data_ro)
    volume_residual_identity_error = abs(
        constant_test_residual-(balance.volume-last_volume),
    )
    result = dict(theta=theta, spp=spp, picard=picard, dt=dt,
                  final_time=time, frequency=c.omega, coriolis=c.f,
                  velocity_dofs=c.W.sub(0).dim(), pressure_dofs=c.W.sub(1).dim(),
                  initial_energy=energies[0],
                  errors=errors(c, state, time).tolist(),
                  final_energy=energies[-1]/energies[0],
                  max_energy_drift=max(abs(np.array(energies)/energies[0]-1)),
                  max_energy_increase=max(np.diff(energies)/energies[0]),
                  max_energy_identity_error=max_energy_identity_error,
                  max_split_work=max_split_work,
                  momentum_identity_error=momentum_identity_error,
                  volume_residual_identity_error=volume_residual_identity_error,
                  **balance.diagnostics())
    PETSc.Sys.Print("rotating-wave diagnostics:", result)
    return state, np.array(energies)/energies[0], result


TIME_METHODS = [
    pytest.param(1, DIRECT_SOLVER_PARAMETERS, id="default-lu"),
    pytest.param(1, ASSEMBLED_SCHUR_SOLVER_PARAMETERS, id="default-schur"),
    pytest.param(0, DIRECT_SOLVER_PARAMETERS, id="coupled-lu"),
]


@pytest.mark.parametrize("picard,solver_parameters", TIME_METHODS)
@pytest.mark.parametrize("theta", [1., 0.5])
def test_rotating_wave_temporal_accuracy(theta, picard, solver_parameters):
    c = rotating_channel()
    counts = (32, 64, 128) if theta == 1 else (8, 16, 32)
    fraction = 0.25 if theta == 1 else 0.375
    measurements = [evolve(c, theta, n, picard=picard,
                           solver_parameters=solver_parameters,
                           periods=fraction)[2] for n in counts]
    e = np.array([r["errors"] for r in measurements])
    orders = np.log2(e[:-1]/e[1:])
    PETSc.Sys.Print("rotating temporal orders:", theta, picard, orders)
    if theta == 1:
        # Surface, u, v, w and amplitude are first order. BE's phase lag
        # is second order for this oscillator; it is not a field-order test.
        assert np.all(orders[:, :5] > 0.8), orders
        assert np.all(orders[-1, :5] > 0.9), orders
        assert np.all(orders[:, :5] < 1.25), orders
        assert np.all((orders[:, 5] > 1.8) & (orders[:, 5] < 2.2)), orders
    else:
        # Coupled CN has no temporal amplitude damping, so its tiny modal
        # amplitude error is a spatial floor, rather than a temporal order.
        columns = [0, 1, 2, 3, 5] if picard == 0 else list(range(6))
        assert np.all(orders[:, columns] > 1.8), orders
        assert np.all(orders[-1, columns] > 1.85), orders
        assert np.all(orders[:, columns] < 2.2), orders
        if picard == 0:
            assert np.max(e[:, 4]) < 2e-5, e


def test_rotating_wave_spatial_error():
    """Doubling both resolutions changes the finest temporal errors by <6%."""
    coarse = evolve(rotating_channel(), 0.5, 32)[2]
    fine = evolve(rotating_channel(nx=32, nz=8), 0.5, 32)[2]
    ratio = np.array(fine["errors"])/np.array(coarse["errors"])
    PETSc.Sys.Print("rotating spatial error ratios:", ratio)
    assert np.all(abs(ratio-1) < 0.06), ratio


@pytest.mark.parametrize("solver_parameters", SOLVER_CASES)
@pytest.mark.parametrize("theta", [1., 0.5])
def test_rotating_projection_splitting(theta, solver_parameters):
    """Finite Picard is not invariant; eight iterations recover the reference."""
    # Two-iteration errors are much smaller than the default errors. Resolve
    # initial off-mode content as well as the dominant external wave.
    c = rotating_channel(nx=8, nz=3)
    discrepancies = []
    for spp in (16, 32, 64):
        reference, _, _ = evolve(c, theta, spp, picard=0)
        row = []
        for picard in (1, 2, 8):
            state, _, diagnostics = evolve(
                c, theta, spp, picard=picard,
                solver_parameters=solver_parameters,
            )
            difference = errors(c, state, 0.375*c.period, reference)[:4]
            PETSc.Sys.Print("rotating splitting errors:", theta, spp, picard,
                            difference)
            row.append(difference)
            # Measured shared-solver maxima: 1.9e-10 in relative energy,
            # 1.4e-12 in momentum and 1.5e-10 m^3 in the constant test.
            assert diagnostics["max_energy_identity_error"] < 1e-9
            assert diagnostics["momentum_identity_error"] < 1e-10
            assert diagnostics["volume_residual_identity_error"] < 1e-8
        # Measured shared-solver errors are many orders below splitting.
        assert np.all(row[1] < 0.03*row[0]), row
        assert np.max(row[2]) < 2e-7, row
        discrepancies.append(row)
    discrepancies = np.array(discrepancies)
    orders = np.log2(discrepancies[:-1, :2]/discrepancies[1:, :2])
    PETSc.Sys.Print("rotating splitting orders:", theta, orders)
    assert np.all((orders[:, 0] > 1.6) & (orders[:, 0] < 2.3)), orders
    assert np.all(orders[:, 1] > 3.0), orders


@pytest.mark.parametrize("solver_parameters", SOLVER_CASES)
@pytest.mark.parametrize("theta", [1., 0.5])
def test_rotating_wave_energy(theta, solver_parameters):
    """Three periods: coupled energy law and the measured splitting work."""
    c = rotating_channel(nx=4, nz=2)
    discrepancies = []
    for spp in (16, 32):
        _, ref_energy, ref_diagnostics = evolve(
            c, theta, spp, picard=0, periods=3.,
        )
        assert ref_diagnostics["max_energy_identity_error"] < 1e-11
        if theta == 0.5:
            assert max(abs(ref_energy-1)) < 1e-11
        else:
            assert np.max(np.diff(ref_energy)) < 1e-12
            assert ref_energy[-1] < 0.03
        row = []
        for picard in (1, 2, 8):
            _, history, diagnostics = evolve(
                c, theta, spp, picard=picard,
                solver_parameters=solver_parameters, periods=3.,
            )
            discrepancy = max(abs(history-ref_energy))
            PETSc.Sys.Print("rotating energy discrepancy:", theta, spp,
                            picard, discrepancy)
            row.append(discrepancy)
            assert diagnostics["max_energy_identity_error"] < 1e-9
            if theta == 1:
                assert np.max(np.diff(history)) < 1e-10
        assert row[0] > 1e-4, row
        assert row[1] < 0.03*row[0], row
        assert row[2] < 2e-7, row
        discrepancies.append(row)
    # The finite-Picard energy defect falls at least quadratically, with
    # four or more powers in the two-iteration case on this mode.
    reduction = np.array(discrepancies[0][:2])/discrepancies[1][:2]
    PETSc.Sys.Print("rotating energy discrepancy reductions:", theta, reduction)
    assert reduction[0] > 3., reduction
    assert reduction[1] > 12., reduction


def test_rotating_mode_equations():
    """Verify signs, dispersion and boundary traces with production terms."""
    c = rotating_channel(nx=4, nz=2)
    s = c.phase0
    u = as_vector((c.shapes[0]*np.sin(s), c.shapes[1]*np.cos(s),
                   c.shapes[2]*np.sin(s)))
    du = float(c.omega)*as_vector((c.shapes[0]*np.cos(s),
                                   -c.shapes[1]*np.sin(s),
                                   c.shapes[2]*np.cos(s)))
    p = c.gravity*c.eta0+c.pressure_shape*np.cos(s)
    dp = -c.omega*c.pressure_shape*np.sin(s)
    strong = du+as_vector((-c.f*u[1], c.f*u[0], 0))+grad(p)
    assert assemble(inner(strong, strong)*c.dx) < 1e-25
    assert assemble(div(u)**2*c.dx) < 1e-25
    assert assemble((dp/c.gravity-u[2])**2*c.ds) < 1e-25
    # Higher quadrature integrates the nonpolynomial analytical expressions;
    # time stepping retains the unmodified production operator quadratures.
    momentum = RotatingMomentumEquation(c.W.sub(0), c.W.sub(0), quad_degree=12)
    continuity = ContinuityEquation(c.W.sub(1), c.W.sub(1), quad_degree=12)
    v, q = TestFunction(c.W.sub(0)), TestFunction(c.W.sub(1))
    rm = assemble(momentum.mass_term(v, du)-momentum.residual(
        v, u, u, {**c.fields, "pressure": p}, c.bcs,
    ))
    rc = assemble(continuity.residual(q, p+dp, p, {
        **c.fields, "velocity": u, "old_pressure": p,
        "dt": 1., "theta": 1.,
    }, c.bcs))
    assert np.linalg.norm(rm.dat.data_ro) < 1e-9
    assert np.linalg.norm(rc.dat.data_ro) < 1e-9
