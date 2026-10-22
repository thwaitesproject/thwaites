"""A travelling, fully 3-D Poincare channel mode (no hydrostatic assumption).

The production signs are u_t-f*v=-p_x, v_t+f*u=-p_y, w_t=-p_z.
For psi=ell*y-omega*t+phi0, D=omega^2-f^2, a=f*ell/omega,
Q=cos(k*x)+(a/k)*sin(k*x), C=cosh(mu*(z+H))/cosh(mu*H),
S=sinh(mu*(z+H))/cosh(mu*H), the solution is

    p = g*eta0 + A*Q*C*cos(psi)
    u = A*(omega*Q'-f*ell*Q)/D*C*sin(psi)
    v = A*(omega*ell*Q-f*Q')/D*C*cos(psi)
    w = A*mu/omega*Q*S*sin(psi).

Here k=pi/Lx, ell=2*pi/Ly, mu^2=(k^2+ell^2)*omega^2/D,
omega^2=g*mu*tanh(mu*H), omega>|f|. Q'=a*Q at BOTH walls
makes u=0; S(-H)=0 makes w=0. The top has p=g*eta and p_t/g=w.
Every field is periodic in y; constant phase travels toward increasing y
at omega/ell. A simple cosine Q would violate the rotating wall condition.

The dated continuation note derives the weak equations, Picard feedback
and energy identity, including the wall contribution to rotational feedback.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI
from firedrake import (
    DirichletBC, FacetNormal, Function, NonlinearVariationalProblem,
    NonlinearVariationalSolver, TestFunction, TestFunctions, div, dot,
    dx, ds_t, grad, split,
)
from firedrake.petsc import PETSc
from thwaites import (
    Constant, ContinuityEquation, ExtrudedMesh, FunctionSpace,
    MixedFunctionSpace, PeriodicRectangleMesh, PressureProjectionTimeIntegrator,
    SpatialCoordinate, VectorFunctionSpace, as_vector, assemble, cos, cosh,
    inner, sin, sinh,
)
from free_surface_helpers import (
    ASSEMBLED_SCHUR_SOLVER_PARAMETERS, DIRECT_SOLVER_PARAMETERS, SOLVER_CASES,
    SurfaceVolumeBalance,
)
from rotating_wave_helpers import CoupledThetaReference, RotatingMomentumEquation


def coefficient_norm(f):
    """Owned coefficient norm; halo coefficients must not be counted twice."""
    a = f.dat.data_ro
    return np.sqrt(f.comm.allreduce(float(np.sum(a*a)), op=MPI.SUM))


def coefficient_sum(f):
    return f.comm.allreduce(float(np.sum(f.dat.data_ro)), op=MPI.SUM)


def channel(nx=16, ny=32, nz=4):
    length, width, gravity, amplitude, eta0 = 1000., 2000., 9.81, 0.01, 0.25
    k, ell = np.pi/length, 2*np.pi/width
    depth = 0.5/k
    mu = 1./depth
    omega = np.sqrt(gravity*mu*np.tanh(mu*depth))
    f = -omega*np.sqrt(1-(k*k+ell*ell)/mu**2)
    mesh = ExtrudedMesh(
        PeriodicRectangleMesh(nx, ny, length, width, direction="y"),
        layers=nz, layer_height=depth/nz,
    )
    mesh.coordinates.dat.data[:, 2] -= depth
    V = VectorFunctionSpace(mesh, "DG", 1)
    Qspace = FunctionSpace(mesh, "CG", 2)
    W = MixedFunctionSpace([V, Qspace])
    momentum = RotatingMomentumEquation(W.sub(0), W.sub(0))
    continuity = ContinuityEquation(W.sub(1), W.sub(1))
    x, y, z = SpatialCoordinate(mesh)
    Q = cos(k*x)+f*ell/(omega*k)*sin(k*x)
    Qp = -k*sin(k*x)+f*ell/omega*cos(k*x)
    C = cosh(mu*(z+depth))/np.cosh(mu*depth)
    S = sinh(mu*(z+depth))/np.cosh(mu*depth)
    A, D = gravity*amplitude, omega**2-f**2
    shapes = as_vector((A*(omega*Qp-f*ell*Q)/D*C,
                        A*(omega*ell*Q-f*Qp)/D*C,
                        A*mu/omega*Q*S))
    c = SimpleNamespace(
        mesh=mesh, W=W, momentum=momentum, continuity=continuity,
        fields={"gravity": Constant(gravity), "coriolis_frequency": Constant(f)},
        bcs={1: {"un": 0.}, 2: {"un": 0.}, "bottom": {"un": 0.},
             "top": {"free_surface": None}},
        dx=momentum.dx, ds=continuity.ds("top"),
        # Analytical errors use higher quadrature, distinct from energy and
        # operator residuals. This does not change any stepping quadrature.
        ax=dx(domain=mesh, degree=12), at=ds_t(domain=mesh, degree=12),
        area=length*width, gravity=gravity, eta0=eta0, amplitude=amplitude,
        omega=omega, f=f, k=k, ell=ell, mu=mu, depth=depth,
        period=2*np.pi/omega, phase0=np.pi/7, shapes=shapes,
        pressure_shape=A*Q*C, Q=Q, Qp=Qp, y=y,
        coordinates=mesh.coordinates.dat.data_ro.copy(),
    )
    c.surface_norm = assemble((amplitude*Q*cos(ell*y))**2*c.at)
    c.velocity_norms = np.array([
        assemble((shapes[i]*cos(ell*y))**2*c.ax) for i in range(3)
    ])
    return c


def exact(c, time):
    psi = c.ell*c.y-c.omega*time+c.phase0
    p = c.gravity*c.eta0+c.pressure_shape*cos(psi)
    u = as_vector((c.shapes[0]*sin(psi), c.shapes[1]*cos(psi),
                   c.shapes[2]*sin(psi)))
    du = float(c.omega)*as_vector((
        -c.shapes[0]*cos(psi), c.shapes[1]*sin(psi), -c.shapes[2]*cos(psi),
    ))
    dp = c.omega*c.pressure_shape*sin(psi)
    return u, p, du, dp


def interior_continuity(c, u):
    """DAE constraint tested against pressure functions with zero top trace."""
    q = TestFunctions(c.W)[1]
    r = assemble(-dot(grad(q), u)*c.continuity.dx)
    # This is a diagnostic test-space restriction, not a pressure BC.
    DirichletBC(c.W.sub(1), 0., "top").zero(r)
    return coefficient_norm(r.subfunctions[1])


def initial_state(c, compatible=True):
    """Mass projection with the interior weak constraint and free top flux.

    The auxiliary multiplier has zero top trace. It is not the physical
    pressure and never removes its physical constant elevation mode.
    """
    ue, pe, _, _ = exact(c, 0.)
    state = Function(c.W)
    u, p = state.subfunctions
    p.interpolate(pe)
    u.project(ue, solver_parameters=DIRECT_SOLVER_PARAMETERS)
    raw = interior_continuity(c, u)
    if compatible:
        auxiliary = Function(c.W)
        a, lam = split(auxiliary)
        v, q = TestFunctions(c.W)
        F = inner(v, a-ue)*c.ax + dot(v, grad(lam))*c.dx
        F += dot(grad(q), a)*c.continuity.dx
        problem = NonlinearVariationalProblem(
            F, auxiliary, bcs=DirichletBC(c.W.sub(1), 0., "top"),
        )
        NonlinearVariationalSolver(
            problem, solver_parameters=DIRECT_SOLVER_PARAMETERS,
            options_prefix="travelling_initial_constraint",
        ).solve()
        u.assign(auxiliary.subfunctions[0])
    c.initial_raw_continuity = raw
    c.initial_continuity = interior_continuity(c, u)
    return state


def modal(c, state):
    """Complex surface Fourier coefficient = exp(i*(phi0-omega*t))."""
    p = state.subfunctions[1]
    anomaly = p/c.gravity-c.eta0
    real = assemble(anomaly*c.amplitude*c.Q*cos(c.ell*c.y)*c.at)
    imag = -assemble(anomaly*c.amplitude*c.Q*sin(c.ell*c.y)*c.at)
    return complex(real, imag)/c.surface_norm


def field_errors(c, state, time, reference=None):
    u, p = state.subfunctions
    if reference is None:
        ue, pe, _, _ = exact(c, time)
    else:
        ue, pe = reference.subfunctions
    values = [np.sqrt(assemble(((p-pe)/c.gravity)**2*c.at)/c.surface_norm)]
    values.extend(np.sqrt(assemble((u[i]-ue[i])**2*c.ax)/c.velocity_norms[i])
                  for i in range(3))
    return np.array(values)


def energy(c, state):
    u, p = state.subfunctions
    return 0.5*assemble(inner(u, u)*c.dx) + 0.5/c.gravity*assemble(
        (p-c.gravity*c.eta0)**2*c.ds,
    )


def make_stepper(c, state, dt, theta, picard, solver_parameters):
    if picard == 0:
        return CoupledThetaReference(c, state, dt, theta)
    options = {} if picard == 1 else {"picard_iterations": picard}
    return PressureProjectionTimeIntegrator(
        [c.momentum, c.continuity], state, c.fields,
        [{"pressure": 1}, {"velocity": 0}], dt, c.bcs,
        theta=theta, pressure_nullspace=None, **options,
        solver_parameters=solver_parameters,
        predictor_solver_parameters=DIRECT_SOLVER_PARAMETERS,
    )


def evolve(c, theta, spp, picard=1, solver_parameters=None, periods=0.375,
           compatible=True):
    if solver_parameters is None:
        solver_parameters = DIRECT_SOLVER_PARAMETERS
    state = initial_state(c, compatible=compatible)
    initial = Function(c.W).assign(state)
    dt = c.period/spp
    steps = round(periods*spp)
    assert steps > 0 and np.isclose(steps, periods*spp)
    stepper = make_stepper(c, state, dt, theta, picard, solver_parameters)
    u, p = state.subfunctions
    balance = SurfaceVolumeBalance(p, c.gravity, c.ds, dt,
                                   step_tolerance=1e-5, total_tolerance=1e-4)
    assert abs(balance.volume-c.area*c.eta0) < 1e-5
    energies, modes = [], []
    max_identity, max_work, max_solver_work = 0., 0., 0.
    max_momentum_identity, max_continuity = 0., 0.
    max_volume_identity, max_omitted = 0., 0.
    v, q = TestFunctions(c.W)
    last_volume = balance.volume
    theta_u, theta_p = Function(c.W.sub(0)), Function(c.W.sub(1))
    max_coefficient = [0., 0.]

    def record():
        # Each rank checks owned coefficients; every rank receives the result.
        finite = all(np.all(np.isfinite(a.dat.data_ro)) for a in (u, p))
        fixed = np.array_equal(c.mesh.coordinates.dat.data_ro, c.coordinates)
        assert c.mesh.comm.allreduce(finite, op=MPI.LAND)
        assert c.mesh.comm.allreduce(fixed, op=MPI.LAND)
        for i, a in enumerate((u, p)):
            local_max = float(np.max(abs(a.dat.data_ro), initial=0.))
            maximum = c.mesh.comm.allreduce(local_max, op=MPI.MAX)
            max_coefficient[i] = max(max_coefficient[i], maximum)
        energies.append(energy(c, state))
        modes.append(modal(c, state))
        assert np.all(np.isfinite([energies[-1], modes[-1]]))

    record()
    for step in range(steps):
        last_volume = balance.volume
        stepper.advance(step*dt)
        balance.check()
        record()
        old = stepper.old if picard == 0 else stepper.solution_old
        un, pn = old.subfunctions
        ut, pt = (1-theta)*un+theta*u, (1-theta)*pn+theta*p
        increment = assemble(inner(u-un, u-un)*c.dx)
        increment += assemble((p-pn)**2/c.gravity*c.ds)
        defect = energies[-1]-energies[-2]+(theta-0.5)*increment
        physical_momentum = c.momentum.mass_term(v, u-un)
        physical_momentum -= dt*c.momentum.residual(
            v, ut, ut, {**c.fields, "pressure": pt}, c.bcs,
        )
        physical_continuity = -dt*c.continuity.residual(q, p, p, {
            **c.fields, "velocity": ut, "old_pressure": pn,
            "dt": dt, "theta": theta,
        }, c.bcs)
        omitted, work = 0, 0.
        if picard:
            coriolis = c.momentum._terms[1]
            omitted = -dt*theta*coriolis.residual(
                v, u-stepper.u_star, u, c.fields, c.bcs,
            )
            work = -dt*theta*assemble(coriolis.residual(
                ut, u-stepper.u_star, u, c.fields, c.bcs,
            ))
        rm = assemble(physical_momentum-omitted).subfunctions[0]
        rc = assemble(physical_continuity).subfunctions[1]
        mscale = coefficient_norm(assemble(
            c.momentum.mass_term(v, u),
        ).subfunctions[0])
        # Direct pairing of the remaining physical residual with theta state
        # gives the signed solver work, without assuming it vanishes.
        theta_u.assign(ut)
        theta_p.assign(pt-c.gravity*c.eta0)
        with rm.dat.vec_ro as rv, rc.dat.vec_ro as cv:
            with theta_u.dat.vec_ro as uv:
                solver_work = rv.dot(uv)
            with theta_p.dat.vec_ro as pv:
                solver_work += cv.dot(pv)
        max_identity = max(max_identity, abs(defect-work-solver_work)/energies[0])
        max_solver_work = max(max_solver_work, abs(solver_work)/energies[0])
        max_work = max(max_work, abs(work)/energies[0])
        max_momentum_identity = max(max_momentum_identity, coefficient_norm(rm)/mscale)
        max_continuity = max(max_continuity, coefficient_norm(rc))
        max_volume_identity = max(max_volume_identity, abs(
            coefficient_sum(rc)-(balance.volume-last_volume),
        ))
        if picard:
            max_omitted = max(max_omitted, coefficient_norm(
                assemble(omitted).subfunctions[0],
            )/mscale)
    time = steps*dt
    modes, energies = np.array(modes), np.array(energies)
    times = np.arange(steps+1)*dt
    phase = np.unwrap(np.angle(modes))
    phase_error = phase-(c.phase0-c.omega*times)
    field = field_errors(c, state, time)
    errors = np.r_[field, abs(abs(modes[-1])-1), abs(phase_error[-1])]
    # Off-mode surface content, measured independently of total field error.
    delta = phase[-1]
    fitted = c.amplitude*abs(modes[-1])*c.Q*cos(c.ell*c.y+delta)
    off_mode = np.sqrt(assemble((p/c.gravity-c.eta0-fitted)**2*c.at)/c.surface_norm)
    result = dict(
        theta=theta, spp=spp, picard=picard, ranks=c.mesh.comm.size,
        dt=dt, final_time=time, omega=c.omega, coriolis=c.f,
        velocity_dofs=c.W.sub(0).dim(), pressure_dofs=c.W.sub(1).dim(),
        errors=errors.tolist(), initialization_errors=field_errors(c, initial, 0.).tolist(),
        initial_raw_continuity=c.initial_raw_continuity,
        initial_continuity=c.initial_continuity, off_mode=off_mode,
        amplitude=abs(modes[-1]), phase=phase[-1], phase_error=phase_error[-1],
        relative_phase_speed=-(phase[-1]-phase[0])/(time*c.omega),
        initial_energy=energies[0], final_energy=energies[-1]/energies[0],
        max_energy_drift=max(abs(energies/energies[0]-1)),
        max_energy_increase=max(np.diff(energies))/energies[0],
        max_energy_identity_error=max_identity, max_solver_work=max_solver_work,
        max_split_work=max_work, momentum_identity_error=max_momentum_identity,
        max_omitted_residual=max_omitted, max_continuity_residual=max_continuity,
        volume_residual_identity_error=max_volume_identity,
        coordinate_change=0., max_velocity_coefficient=max_coefficient[0],
        max_pressure_coefficient=max_coefficient[1], **balance.diagnostics(),
    )
    assert np.all(np.isfinite(errors))
    PETSc.Sys.Print("propagating-wave diagnostics:", result)
    return state, dict(energy=energies/energies[0], phase=phase,
                       phase_error=phase_error, amplitude=abs(modes)), result


TIME_METHODS = [
    pytest.param(1, DIRECT_SOLVER_PARAMETERS, id="default-lu"),
    pytest.param(1, ASSEMBLED_SCHUR_SOLVER_PARAMETERS, id="default-schur"),
    pytest.param(0, DIRECT_SOLVER_PARAMETERS, id="coupled-lu"),
]


@pytest.mark.parametrize("picard,solver_parameters", TIME_METHODS)
@pytest.mark.parametrize("theta", [1., 0.5])
def test_propagating_temporal_accuracy(theta, picard, solver_parameters):
    c = channel()
    counts = (32, 64, 128) if theta == 1 else (6, 12, 24)
    rows = [evolve(c, theta, n, picard=picard,
                   solver_parameters=solver_parameters,
                   periods=.5)[2] for n in counts]
    e = np.array([r["errors"] for r in rows])
    orders = np.log2(e[:-1]/e[1:])
    PETSc.Sys.Print("propagating temporal orders:", theta, picard, orders)
    columns = [0, 1, 2, 3, 5] if picard == 0 and theta == 0.5 else list(range(6))
    if theta == 1:
        assert np.all((orders[:, :5] > 0.75) & (orders[:, :5] < 1.2)), orders
        assert np.all(orders[-1, :5] > 0.9), orders
        assert np.all((orders[:, 5] > 1.7) & (orders[:, 5] < 2.3)), orders
    else:
        assert np.all((orders[:, columns] > 1.8) & (orders[:, columns] < 2.2)), orders
    if picard == 0 and theta == .5:
        assert max(e[:, 4]) < 1e-4
    for row in rows:
        assert np.all(np.array(row["initialization_errors"])
                      < .3*np.array(row["errors"][:4]))
        assert row["relative_phase_speed"] > 0.
        assert row["initial_continuity"] < 1e-8
        assert row["momentum_identity_error"] < 1e-9
        assert row["max_energy_identity_error"] < 1e-10


def test_propagating_mode_equations():
    c = channel(nx=4, ny=8, nz=2)
    u, p, du, dp = exact(c, 0.)
    strong = du+as_vector((-c.f*u[1], c.f*u[0], 0))+grad(p)
    assert assemble(inner(strong, strong)*c.ax) < 1e-24
    assert assemble(div(u)**2*c.ax) < 1e-24
    assert assemble((dp/c.gravity-u[2])**2*c.at) < 1e-24
    n = FacetNormal(c.mesh)
    for wall in (1, 2, "bottom"):
        assert assemble(dot(n, u)**2*c.continuity.ds(wall)) < 1e-23
    for wall in (1, 2):
        assert assemble((c.Qp-c.f*c.ell/c.omega*c.Q)**2
                        * c.continuity.ds(wall)) < 1e-23
    residuals = []
    for degree in (5, 12):
        momentum = RotatingMomentumEquation(c.W.sub(0), c.W.sub(0), quad_degree=degree)
        continuity = ContinuityEquation(c.W.sub(1), c.W.sub(1), quad_degree=degree)
        v, q = TestFunction(c.W.sub(0)), TestFunction(c.W.sub(1))
        rm = assemble(momentum.mass_term(v, du)-momentum.residual(
            v, u, u, {**c.fields, "pressure": p}, c.bcs,
        ))
        rc = assemble(continuity.residual(q, p+dp, p, {
            **c.fields, "velocity": u, "old_pressure": p, "dt": 1., "theta": 1.,
        }, c.bcs))
        residuals.append([coefficient_norm(rm), coefficient_norm(rc)])
    PETSc.Sys.Print("propagating analytical weak residuals (q5/q12):", residuals)
    assert max(residuals[-1]) < 1e-8
    raw = initial_state(c, compatible=False)
    compatible = initial_state(c)
    PETSc.Sys.Print("propagating initial compatibility:",
                    interior_continuity(c, raw.subfunctions[0]),
                    interior_continuity(c, compatible.subfunctions[0]))
    assert c.initial_raw_continuity > 1e-3
    assert c.initial_continuity < 1e-8


def check_identities(d):
    assert d["max_energy_identity_error"] < 1e-10, d
    assert d["max_solver_work"] < 1e-8, d
    assert d["momentum_identity_error"] < 1e-9, d
    assert d["volume_residual_identity_error"] < 1e-8, d
    assert d["initial_continuity"] < 1e-8, d


@pytest.mark.parametrize("solver_parameters", SOLVER_CASES)
@pytest.mark.parametrize("theta", [1., 0.5])
def test_propagating_projection_splitting(theta, solver_parameters):
    c = channel(nx=4, ny=8, nz=2)
    rows = []
    for spp in (16, 32, 64):
        reference, _, _ = evolve(c, theta, spp, picard=0)
        row = []
        for picard in (1, 2, 12):
            state, _, d = evolve(c, theta, spp, picard=picard,
                                 solver_parameters=solver_parameters)
            difference = field_errors(c, state, .375*c.period, reference)
            ratio = modal(c, state)/modal(c, reference)
            difference = np.r_[difference, abs(abs(ratio)-1), abs(np.angle(ratio))]
            PETSc.Sys.Print("propagating splitting errors:", theta, spp,
                            picard, difference)
            row.append(difference)
            check_identities(d)
        assert np.all(row[1][:5] < 0.15*row[0][:5]), row
        assert max(row[2]) < 2e-7, row
        rows.append(row)
    rows = np.array(rows)
    orders = np.log2(rows[:-1, :2]/rows[1:, :2])
    PETSc.Sys.Print("propagating splitting orders:", theta, orders)
    # Wall feedback changes the Picard behavior from the y-independent mode.
    # No fourth-order two-iteration assertion is inherited from that test.
    assert np.all((orders[:, 0, :5] > 1.6) & (orders[:, 0, :5] < 2.4)), orders
    assert np.all(orders[:, 1, :4] > 2.5), orders


@pytest.mark.parametrize("solver_parameters", SOLVER_CASES)
@pytest.mark.parametrize("theta", [1., 0.5])
def test_propagating_energy(theta, solver_parameters):
    c = channel(nx=4, ny=8, nz=2)
    discrepancies = []
    for spp in (16, 32):
        _, reference, rd = evolve(c, theta, spp, picard=0, periods=3.)
        check_identities(rd)
        if theta == .5:
            assert max(abs(reference["energy"]-1)) < 1e-11
        else:
            assert max(np.diff(reference["energy"])) < 1e-12
            assert reference["energy"][-1] < .04
        row = []
        for picard in (1, 2, 12):
            _, history, d = evolve(c, theta, spp, picard=picard, periods=3.,
                                   solver_parameters=solver_parameters)
            check_identities(d)
            discrepancy = max(abs(history["energy"]-reference["energy"]))
            row.append(discrepancy)
            PETSc.Sys.Print("propagating energy discrepancy:", theta, spp,
                            picard, discrepancy)
            assert np.all(np.diff(history["phase"]) < 0.)
            assert abs(d["phase"]-c.phase0) > 4*np.pi
            if theta == 1:
                assert max(np.diff(history["energy"])) < 1e-10
        # CN work cancellation makes the default defect much smaller than
        # in the y-independent mode, but still far above solver accuracy.
        assert row[0] > 1e-6
        if theta == 1:
            assert row[1] < .15*row[0]
        else:
            # Signed rotational work cancels more strongly at one iteration
            # for this travelling mode. Field convergence does not imply
            # monotone improvement of its scalar energy defect.
            assert max(row[:2]) < .003
        assert row[2] < 2e-7
        discrepancies.append(row)
    reduction = np.array(discrepancies[0][:2])/discrepancies[1][:2]
    PETSc.Sys.Print("propagating energy reductions:", theta, reduction)
    assert reduction[0] > 3.
    # In BE the energy-error peak shifts along the damping timescale
    # ~1/dt. On these grids its maximum falls quadratically even though
    # fixed-time two-iteration field differences fall cubically.
    assert reduction[1] > (3.5 if theta == 1 else 6.)


def test_propagating_spatial_accuracy():
    measurements = []
    for nx, ny, nz in ((4, 8, 2), (8, 16, 4), (12, 24, 6)):
        c = channel(nx=nx, ny=ny, nz=nz)
        _, _, d = evolve(c, .5, 512, periods=.0625)
        measurements.append(d)
    e = np.array([d["errors"][:4] for d in measurements])
    orders = np.log(e[:-1]/e[1:])/np.log(np.array([2., 1.5]))[:, None]
    PETSc.Sys.Print("propagating spatial orders:", orders)
    assert np.all(orders > 1.7), orders
    assert np.all(orders < 3.5), orders
    # Halve dt at the finest mesh: temporal/splitting errors must not set
    # the spatial error. The initialization error is reported separately.
    _, _, refined = evolve(c, .5, 1024, periods=.0625)
    ratio = np.array(refined["errors"][:4])/e[-1]
    assert np.all(abs(ratio-1) < .03), ratio


def test_propagating_temporal_spatial_separation():
    # A second space at the finest temporal timestep bounds the space/init
    # floor. This check includes both the default projection and reference.
    for picard in (1, 0):
        coarse = evolve(channel(), .5, 24, picard=picard, periods=.5)[2]
        fine = evolve(channel(nx=20, ny=40, nz=5), .5, 24,
                      picard=picard, periods=.5)[2]
        ratio = np.array(fine["errors"])[[0, 1, 2, 3, 5]]
        ratio /= np.array(coarse["errors"])[[0, 1, 2, 3, 5]]
        PETSc.Sys.Print("propagating temporal/spatial separation:", picard, ratio)
        assert np.all(abs(ratio-1) < .08), ratio


@pytest.mark.parametrize("solver_parameters", SOLVER_CASES)
@pytest.mark.parametrize("theta", [1., 0.5])
def test_propagating_distributed_diagnostics(theta, solver_parameters):
    """Also run by a separate mpiexec -n 2 command (without xdist)."""
    c = channel(nx=4, ny=8, nz=2)
    reference, rh, _ = evolve(c, theta, 32, picard=0, periods=2.)
    state, history, d = evolve(c, theta, 32, periods=2.,
                               solver_parameters=solver_parameters)
    check_identities(d)
    assert np.all(np.diff(history["phase"]) < 0.)
    assert .9 < d["relative_phase_speed"] < 1.05
    assert abs(d["phase"]-c.phase0) > 3*np.pi
    assert d["off_mode"] < .05
    difference = field_errors(c, state, 2*c.period, reference)
    PETSc.Sys.Print("propagating distributed comparison:", difference,
                    max(abs(history["energy"]-rh["energy"])))
    assert max(difference) < .08
    assert max(abs(history["energy"]-rh["energy"])) < .04


def inverse_gradient(c, pressure, alpha=0.):
    """Apply (M+alpha*C)^-1 G, retaining production pressure traces."""
    velocity = Function(c.W.sub(0))
    v = TestFunction(c.W.sub(0))
    F = c.momentum.mass_term(v, velocity)
    F += c.momentum._terms[0].residual(
        v, velocity, velocity, {**c.fields, "pressure": pressure}, c.bcs,
    )
    if alpha:
        F -= alpha*c.momentum._terms[1].residual(
            v, velocity, velocity, c.fields, c.bcs,
        )
    NonlinearVariationalSolver(
        NonlinearVariationalProblem(F, velocity),
        solver_parameters=DIRECT_SOLVER_PARAMETERS,
        options_prefix="travelling_feedback_inverse",
    ).solve()
    return velocity


def test_propagating_picard_feedback():
    """Check H*e_j = alpha^2 G^T(M^-1-A^-1)G*e_l, including walls."""
    c = channel(nx=4, ny=8, nz=2)
    dt, theta = c.period/16, .5
    alpha = theta*dt
    state = initial_state(c)
    reference = Function(c.W).assign(state)
    coupled = CoupledThetaReference(c, reference, dt, theta)
    coupled.advance(0.)
    pc = reference.subfunctions[1]
    stepper = make_stepper(c, state, dt, theta, 1, DIRECT_SOLVER_PARAMETERS)
    stepper.initialize(state)
    q = TestFunction(c.W.sub(1))
    for _ in range(3):
        ep = Function(c.W.sub(1)).assign(state.subfunctions[1]-pc)
        stepper.picard_step()
        ej = state.subfunctions[1]-pc
        me, ae = inverse_gradient(c, ep), inverse_gradient(c, ep, alpha)
        mj = inverse_gradient(c, ej)
        H = q*ej/c.gravity*c.ds + alpha**2*dot(grad(q), mj)*c.continuity.dx
        rhs = alpha**2*dot(grad(q), me-ae)*c.continuity.dx
        scale = assemble(q*ep/c.gravity*c.ds
                         + alpha**2*dot(grad(q), me)*c.continuity.dx)
        error = coefficient_norm(assemble(H-rhs))/coefficient_norm(scale)
        PETSc.Sys.Print("propagating Picard feedback identity:", error)
        assert error < 1e-9
    # Although div(R grad e)=0 strongly, the wall flux is nonzero for ell!=0.
    # Dropping this trace would incorrectly predict the old mode's extra
    # two powers of dt after another Picard iteration.
    _, p, _, _ = exact(c, 0.)
    rotated = as_vector((-c.f*p.dx(1), c.f*p.dx(0), 0.))
    boundary = ContinuityEquation(c.W.sub(1), c.W.sub(1), quad_degree=12)
    volume = dot(grad(q), rotated)*c.ax
    walls = sum(q*dot(FacetNormal(c.mesh), rotated)*boundary.ds(wall)
                for wall in (1, 2))
    norm = coefficient_norm(assemble(walls))
    error = coefficient_norm(assemble(volume-walls))/norm
    PETSc.Sys.Print("propagating rotational wall feedback:", norm, error)
    assert norm > 1e-3
    assert error < 1e-9
