"""
1D Euler equations dataset: FD-WENOZ on fine grid, cell-average downsample.

State: w = (rho, m, E) with shape (3, N), m = rho * u.
Ideal gas:
    E = p / (gamma - 1) + 0.5 * rho * u^2

PDE:
    rho_t + m_x = 0
    m_t   + (m^2/rho + p)_x = 0
    E_t   + ((E+p)u)_x = 0

Default BC is periodic. Use CLI ``--bc outflow`` (or ``zero`` / ``reflect``) to
generate open-boundary data; ``meta['boundary']`` is set accordingly for training/eval.
Flux recovery uses periodic rolls only when bc is periodic; otherwise
replicated-edge neighbors are used (``recover_flux_open_np``).

Notes
-----
1. For dataset generation with periodic BC, we use a standard cell-centered grid
   on [x_left, x_right]:
       x_i = x_left + (i + 1/2) dx
2. Riemann plots are done on a larger domain to avoid premature boundary effects.
3. For Euler we prefer characteristic-wise WENO reconstruction; component-wise
   reconstruction is still available for comparison / experiments.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from conslaw.solver import FD_WENOZ, downsample_cell_average


# =============================================================================
# Global config
# =============================================================================
GAMMA = 1.4
RHO_MIN = 1e-8
P_MIN = 1e-12

# Base resolution for domain [0, 1]
NX = 256
UPSAMPLE = 4
DT = 5e-2
N_STEPS_PER_IC = 20
WENO_CFL = 0.4

DATA_XL = 0.0
DATA_XR = 1.0
BC = "periodic"

PLOT_XL = 0.0
PLOT_XR = 1.0
PLOT_X0 = 0.0


def make_cell_centered_grid(xl, xr, nx):
    dx = (xr - xl) / nx
    x = xl + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    return x, dx


def infer_edges_from_centers(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1D uniform grid with at least 2 points.")
    dx = x[1] - x[0]
    xl = x[0] - 0.5 * dx
    xr = x[-1] + 0.5 * dx
    return xl, xr, dx


# =============================================================================
# Euler flux / primitive-conserved transforms
# =============================================================================
def primitives_to_conserved(rho, u_vel, p, gamma=GAMMA):
    """
    (rho, u, p) -> (rho, m, E)
    """
    rho = np.asarray(rho, dtype=np.float64)
    u_vel = np.asarray(u_vel, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    rho = np.maximum(rho, RHO_MIN)
    p = np.maximum(p, P_MIN)

    m = rho * u_vel
    E = p / (gamma - 1.0) + 0.5 * rho * u_vel ** 2
    return np.stack([rho, m, E], axis=0)


def conserved_to_primitives(u, gamma=GAMMA, rho_min=RHO_MIN, p_min=P_MIN):
    """
    (rho, m, E) -> (rho, u, p)
    u: (3, N)
    """
    rho = np.maximum(np.asarray(u[0]), rho_min)
    m = np.asarray(u[1])
    E = np.asarray(u[2])
    u_vel = m / rho
    p = (gamma - 1.0) * (E - 0.5 * m * m / rho)
    p = np.maximum(np.asarray(p), p_min)
    return rho, u_vel, p


def enforce_physical_state(u, gamma=GAMMA, rho_min=RHO_MIN, p_min=P_MIN):
    u = np.asarray(u, dtype=np.float64).copy()
    rho = np.maximum(u[0], rho_min)
    m = u[1]
    E = u[2]

    kinetic = 0.5 * m * m / rho
    E_min = kinetic + p_min / (gamma - 1.0)
    E = np.maximum(E, E_min)

    u[0] = rho
    u[2] = E
    return u


def euler_flux(u, gamma=GAMMA, rho_min=RHO_MIN, p_min=P_MIN):
    """
    u: (3, N) = (rho, m, E)
    Returns F: (3, N) = (m, m^2/rho + p, (E+p)u)
    """
    rho = np.maximum(u[0], rho_min)
    m = u[1]
    E = u[2]

    u_vel = m / rho
    p = np.maximum((gamma - 1.0) * (E - 0.5 * m * m / rho), p_min)

    F0 = m
    F1 = m * u_vel + p
    F2 = (E + p) * u_vel
    return np.stack([F0, F1, F2], axis=0)


def euler_wave_speed(u, gamma=GAMMA, rho_min=RHO_MIN, p_min=P_MIN):
    """
    Global max characteristic speed |u| + c over the state.
    Returns a scalar.
    """
    rho = np.maximum(u[0], rho_min)
    m = u[1]
    E = u[2]

    u_vel = m / rho
    p = np.maximum((gamma - 1.0) * (E - 0.5 * m * m / rho), p_min)
    c = np.sqrt(gamma * p / rho)
    return np.abs(u_vel) + c

def euler_jacobian(u, gamma=GAMMA, rho_min=RHO_MIN):
    if u.ndim != 2 or u.shape[0] != 3:
        raise ValueError(f"Expected u.shape = (3, N), got {u.shape}")

    rho = np.maximum(u[0], rho_min)
    m   = u[1]
    E   = u[2]

    vel = m / rho
    p = (gamma - 1.0) * (E - 0.5 * m * m / rho)
    H = (E + p) / rho   # total enthalpy

    A = np.empty((3, 3, u.shape[1]), dtype=u.dtype)

    # first row
    A[0, 0] = 0.0
    A[0, 1] = 1.0
    A[0, 2] = 0.0

    # second row
    A[1, 0] = 0.5 * (gamma - 3.0) * vel**2
    A[1, 1] = (3.0 - gamma) * vel
    A[1, 2] = gamma - 1.0

    # third row
    A[2, 0] = vel * (0.5 * (gamma - 1.0) * vel**2 - H)
    A[2, 1] = H - (gamma - 1.0) * vel**2
    A[2, 2] = gamma * vel

    return A


def euler_characteristic_matrices_batch(
    uL,
    uR,
    gamma=GAMMA,
    rho_min=RHO_MIN,
    p_min=P_MIN,
):
    rhoL = np.maximum(uL[0], rho_min)
    mL = uL[1]
    EL = uL[2]
    velL = mL / rhoL
    pL = np.maximum((gamma - 1.0) * (EL - 0.5 * mL * mL / rhoL), p_min)
    HL = (EL + pL) / rhoL

    rhoR = np.maximum(uR[0], rho_min)
    mR = uR[1]
    ER = uR[2]
    velR = mR / rhoR
    pR = np.maximum((gamma - 1.0) * (ER - 0.5 * mR * mR / rhoR), p_min)
    HR = (ER + pR) / rhoR

    sqrt_rhoL = np.sqrt(rhoL)
    sqrt_rhoR = np.sqrt(rhoR)
    denom = np.maximum(sqrt_rhoL + sqrt_rhoR, 1e-14)

    vel = (sqrt_rhoL * velL + sqrt_rhoR * velR) / denom
    H = (sqrt_rhoL * HL + sqrt_rhoR * HR) / denom
    csq = np.maximum((gamma - 1.0) * (H - 0.5 * vel * vel), 1e-12)
    c = np.sqrt(csq)

    beta = (gamma - 1.0) / csq

    Nint = uL.shape[1]
    R = np.empty((Nint, 3, 3), dtype=uL.dtype)
    R[:, 0, 0] = 1.0
    R[:, 1, 0] = vel - c
    R[:, 2, 0] = H - vel * c
    R[:, 0, 1] = 1.0
    R[:, 1, 1] = vel
    R[:, 2, 1] = 0.5 * vel * vel
    R[:, 0, 2] = 1.0
    R[:, 1, 2] = vel + c
    R[:, 2, 2] = H + vel * c

    # Analytic left eigenvectors are much faster than a batched matrix inverse.
    L = np.empty((Nint, 3, 3), dtype=uL.dtype)
    L[:, 0, 0] = 0.25 * beta * vel * vel + 0.5 * vel / c
    L[:, 0, 1] = -0.5 * beta * vel - 0.5 / c
    L[:, 0, 2] = 0.5 * beta

    L[:, 1, 0] = 1.0 - 0.5 * beta * vel * vel
    L[:, 1, 1] = beta * vel
    L[:, 1, 2] = -beta

    L[:, 2, 0] = 0.25 * beta * vel * vel - 0.5 * vel / c
    L[:, 2, 1] = -0.5 * beta * vel + 0.5 / c
    L[:, 2, 2] = 0.5 * beta

    a_int = np.maximum.reduce([np.abs(vel - c), np.abs(vel), np.abs(vel + c)])
    return L, R, a_int


def build_euler_solver(
    gamma=GAMMA,
    bc=BC,
    reconstruction="characteristic",
    WENOtype="WENO-JS",
):
    def flux(u):
        return euler_flux(u, gamma=gamma)

    def alpha(u):
        return euler_wave_speed(u, gamma=gamma)

    if reconstruction not in ("component", "characteristic"):
        raise ValueError('reconstruction must be "component" or "characteristic".')

    char_decomp_batch = None
    if reconstruction == "characteristic":
        char_decomp_batch = lambda uL, uR: euler_characteristic_matrices_batch(
            uL, uR, gamma=gamma
        )

    return FD_WENOZ(
        flux=flux,
        dflux=None,
        alpha=alpha,
        char_decomp_batch=char_decomp_batch,
        flux_split="local_lf",
        eps=1e-20,
        bc=bc,
        WENOtype=WENOtype,
        n_comp=3,
    )


def default_x0_from_grid(x):
    xl, xr, _ = infer_edges_from_centers(x)
    return 0.5 * (xl + xr)


def sod_ic(x, x0=None):
    """
    Sod shock tube:
      left  = (1.0,   0.0, 1.0)
      right = (0.125, 0.0, 0.1)
    """
    x = np.asarray(x, dtype=np.float64)
    if x0 is None:
        x0 = default_x0_from_grid(x)

    rho = np.where(x < x0, 1.0, 0.125)
    u_vel = np.zeros_like(x)
    p = np.where(x < x0, 1.0, 0.1)
    return primitives_to_conserved(rho, u_vel, p)


def lax_ic(x, x0=None):
    """
    Lax shock tube:
      left  = (0.445, 0.698, 3.528)
      right = (0.500, 0.000, 0.571)
    """
    x = np.asarray(x, dtype=np.float64)
    if x0 is None:
        x0 = default_x0_from_grid(x)

    rho = np.where(x < x0, 0.445, 0.5)
    u_vel = np.where(x < x0, 0.698, 0.0)
    p = np.where(x < x0, 3.528, 0.571)
    return primitives_to_conserved(rho, u_vel, p)


def _bounded_fourier_series(x, xl, xr, rng, K):
    """
    Build a Fourier series S(x) with guaranteed |S(x)| <= 1
    by normalizing with the sum of absolute coefficients.
    """
    x = np.asarray(x, dtype=np.float64)
    L = xr - xl
    theta = 2.0 * np.pi * (x - xl) / L

    s = np.zeros_like(x, dtype=np.float64)
    denom = 0.0

    for k in range(1, K + 1):
        a = rng.uniform(-1.0, 1.0)
        b = rng.uniform(-1.0, 1.0)
        s += a * np.sin(k * theta) + b * np.cos(k * theta)
        denom += abs(a) + abs(b)

    if denom < 1e-14:
        return s
    return s / denom   # then |S(x)| <= 1


def _sample_primitive_state(rng, rho_range, u_range, p_range):
    rho = rng.uniform(*rho_range)
    u   = rng.uniform(*u_range)
    p   = rng.uniform(*p_range)
    return rho, u, p


def _sample_far_positive(rng, ref, val_range, rel_min):
    """
    For positive scalar v in [lo, hi], sample v so that
        |v - ref| / max(v, ref) >= rel_min
    i.e.
        v <= ref * (1-rel_min)   or   v >= ref / (1-rel_min)
    """
    lo, hi = val_range
    eps = 1e-14
    one_minus = max(1.0 - rel_min, eps)

    left_hi  = min(hi, ref * one_minus)
    right_lo = max(lo, ref / one_minus)

    intervals = []
    lengths = []

    if lo < left_hi:
        intervals.append((lo, left_hi))
        lengths.append(left_hi - lo)

    if right_lo < hi:
        intervals.append((right_lo, hi))
        lengths.append(hi - right_lo)

    if lengths:
        lengths = np.asarray(lengths, dtype=np.float64)
        lengths /= lengths.sum()
        k = rng.choice(len(intervals), p=lengths)
        a, b = intervals[k]
        return rng.uniform(a, b)

    # fallback: if range is too tight, pick the farther endpoint
    return lo if abs(lo - ref) >= abs(hi - ref) else hi


def _sample_far_real(rng, ref, val_range, abs_min):
    """
    Sample v in [lo, hi] so that |v - ref| >= abs_min.
    """
    lo, hi = val_range

    left_hi  = min(hi, ref - abs_min)
    right_lo = max(lo, ref + abs_min)

    intervals = []
    lengths = []

    if lo < left_hi:
        intervals.append((lo, left_hi))
        lengths.append(left_hi - lo)

    if right_lo < hi:
        intervals.append((right_lo, hi))
        lengths.append(hi - right_lo)

    if lengths:
        lengths = np.asarray(lengths, dtype=np.float64)
        lengths /= lengths.sum()
        k = rng.choice(len(intervals), p=lengths)
        a, b = intervals[k]
        return rng.uniform(a, b)

    return lo if abs(lo - ref) >= abs(hi - ref) else hi


def _sample_state_far_from(
    rng,
    prev_state,
    rho_range,
    u_range,
    p_range,
    rel_rho_min,
    rel_p_min,
    abs_u_min,
):
    rho0, u0, p0 = prev_state

    rho = _sample_far_positive(rng, rho0, rho_range, rel_rho_min)
    u   = _sample_far_real(rng, u0, u_range, abs_u_min)
    p   = _sample_far_positive(rng, p0, p_range, rel_p_min)

    return rho, u, p

def sod_like_ic(
    x,
    rng,
    rho_range=(0.05, 1.20),
    u_range=(-1.0, 1.0),
    p_range=(0.05, 1.20),
    rel_rho_min=0.30,
    rel_p_min=0.30,
    abs_u_min=0.30,
):
    x = np.asarray(x, dtype=np.float64)
    xl, xr, _ = infer_edges_from_centers(x)

    x0 = rng.uniform(xl + 0.2 * (xr - xl), xr - 0.2 * (xr - xl))

    left_state = _sample_primitive_state(rng, rho_range, u_range, p_range)
    right_state = _sample_state_far_from(
        rng,
        left_state,
        rho_range=rho_range,
        u_range=u_range,
        p_range=p_range,
        rel_rho_min=rel_rho_min,
        rel_p_min=rel_p_min,
        abs_u_min=abs_u_min,
    )

    rhoL, uL, pL = left_state
    rhoR, uR, pR = right_state

    rho = np.where(x < x0, rhoL, rhoR)
    u   = np.where(x < x0, uL, uR)
    p   = np.where(x < x0, pL, pR)

    rho = np.maximum(rho, 1e-8)
    p   = np.maximum(p, 1e-8)

    return primitives_to_conserved(rho, u, p)


def smooth_density_perturbation_ic(x, rng):
    x = np.asarray(x, dtype=np.float64)
    xl, xr, _ = infer_edges_from_centers(x)

    # modes = 1,2,3 are all possible
    K = int(rng.integers(1, 4))

    # Broader base states so smooth ICs cover a wider physical regime.
    rho0 = rng.uniform(0.30, 1.10)
    p0   = rng.uniform(0.30, 1.10)
    u0   = rng.uniform(-1.00, 1.00)

    # bounded Fourier shapes, each in [-1, 1]
    Sr = _bounded_fourier_series(x, xl, xr, rng, K)
    Su = _bounded_fourier_series(x, xl, xr, rng, K)
    Sp = _bounded_fourier_series(x, xl, xr, rng, K)

    # Stronger amplitudes while still keeping rho,p positive since |S| <= 1
    # and alpha_r, alpha_p stay below 1.
    alpha_r = rng.uniform(0.12, 0.65)
    alpha_p = rng.uniform(0.12, 0.65)
    alpha_u = rng.uniform(0.10, 0.70)

    rho = rho0 * (1.0 + alpha_r * Sr)
    p   = p0   * (1.0 + alpha_p * Sp)
    u   = u0   + alpha_u * Su

    rho = np.maximum(rho, 1e-6)
    p   = np.maximum(p, 1e-6)

    return primitives_to_conserved(rho, u, p)


def sample_random_ic(x, rng):
    r = rng.uniform()

    if r < 0.8:
        return sod_like_ic(x, rng), "sod_like"
    else:
        return smooth_density_perturbation_ic(x, rng), "smooth_perturbation"


def recover_flux_periodic_np(u, u_next, dt, dx):
    u = np.asarray(u, dtype=np.float64)
    u_next = np.asarray(u_next, dtype=np.float64)

    u_left = np.roll(u, 1, axis=-1)
    u_right = np.roll(u, -1, axis=-1)
    u_avg = 0.5 * (u_left + u_right)

    q = (dx / dt) * (u_avg - u_next)
    q = q - q.mean(axis=-1, keepdims=True)
    flux = np.zeros_like(q)
    flux[..., 1:] = np.cumsum(q[..., 1:], axis=-1)
    flux = flux - flux.mean(axis=-1, keepdims=True)
    return flux


def recover_flux_open_np(u, u_next, dt, dx):
    """
    Same discrete relation as periodic recover, but neighbors use edge replication
    (no wrap). No global mean removal on q/flux — allows net boundary flux.

    Matches the transmissive-style cell neighbor pattern (ghost = edge value).
    """
    u = np.asarray(u, dtype=np.float64)
    u_next = np.asarray(u_next, dtype=np.float64)
    u_left = np.concatenate([u[..., :1], u[..., :-1]], axis=-1)
    u_right = np.concatenate([u[..., 1:], u[..., -1:]], axis=-1)
    u_avg = 0.5 * (u_left + u_right)
    q = (dx / dt) * (u_avg - u_next)
    flux = np.zeros_like(q)
    flux[..., 1:] = np.cumsum(q[..., 1:], axis=-1)
    return flux


def _recover_flux_for_bc(u, u_next, dt, dx, bc: str):
    if bc == "periodic":
        return recover_flux_periodic_np(u, u_next, dt, dx)
    return recover_flux_open_np(u, u_next, dt, dx)


def generate_trajectory_euler(state0_fine, solver, T, dt_snap, dx_fine, cfl=WENO_CFL):
    state = np.asarray(state0_fine, dtype=np.float64).copy()
    times = np.arange(0.0, T + 1e-12, dt_snap)
    snaps = [state.copy()]
    t = 0.0
    for i in range(1, len(times)):
        t_target = times[i]
        dt_interval = t_target - t
        state = solver.solve(
            state,
            dx=dx_fine,
            T=dt_interval,
            cfl=cfl,
            return_all=False,
        )
        snaps.append(state.copy())
        t = t_target
    return np.stack(snaps, axis=0), times

def sample_smooth_ic(x, rng):
    return smooth_density_perturbation_ic(x, rng), "smooth_perturbation"


def _build_pv_flux_split_chunk(
    n_ic,
    seed,
    nx_low=NX,
    upsample=UPSAMPLE,
    T=1.0,
    dt_snap=DT,
    dt_min=None,
    dt_max=None,
    cfl=WENO_CFL,
    x_left=DATA_XL,
    x_right=DATA_XR,
    bc=BC,
    reconstruction="characteristic",
    show_progress=True,
    worker_name="worker",
):
    """
    Build pv+flux pairs for 1-step supervised learning.

    If dt_min/dt_max are provided (or differ from dt_snap), we sample dt_step ~ U(dt_min, dt_max)
    independently at each step, and also store dt_step per sample for dt-conditioned training.
    """
    rng = np.random.default_rng(seed)

    nx_fine = nx_low * upsample
    x_fine, dx_fine = make_cell_centered_grid(x_left, x_right, nx_fine)
    solver = build_euler_solver(bc=bc, reconstruction=reconstruction)

    n_steps_per_ic = int(round(T / dt_snap))
    if n_steps_per_ic <= 0:
        raise ValueError(f"n_steps_per_ic must be positive, got {n_steps_per_ic} (T={T}, dt_snap={dt_snap})")

    if dt_min is None:
        dt_min = dt_snap
    if dt_max is None:
        dt_max = dt_snap
    dt_min = float(dt_min)
    dt_max = float(dt_max)
    if not (dt_min > 0.0 and dt_max > 0.0 and dt_min <= dt_max):
        raise ValueError(f"Invalid dt range: dt_min={dt_min}, dt_max={dt_max}")

    dx_low = (x_right - x_left) / nx_low

    if show_progress:
        print(
            f"[Euler pv+flux][{worker_name}] chunk start: n_ic={n_ic}, nx_fine={nx_fine}, "
            f"n_steps_per_ic={n_steps_per_ic}, dt_range=[{dt_min},{dt_max}], dx_low={dx_low:.6e}",
            flush=True,
        )

    def down(u):
        return np.stack(
            [downsample_cell_average(u[i], upsample) for i in range(3)],
            axis=0,
        )

    X_list = []
    Y_list = []
    F_list = []
    DT_list = []
    labels = []

    for s in range(n_ic):
        if show_progress:
            print(f"[Euler pv+flux][{worker_name}] trajectory {s + 1}/{n_ic}: start", flush=True)

        state0_fine, ic_label = sample_random_ic(x_fine, rng)
        state0_fine = enforce_physical_state(state0_fine)

        states_coarse = [down(state0_fine)]
        dt_steps = []

        state = np.asarray(state0_fine, dtype=np.float64)
        for step_idx in range(n_steps_per_ic):
            dt_step = float(rng.uniform(dt_min, dt_max))
            if show_progress:
                print(
                    f"[Euler pv+flux][{worker_name}] traj {s + 1}/{n_ic} "
                    f"step {step_idx + 1}/{n_steps_per_ic}: sampled dt={dt_step:.6e}",
                    flush=True,
                )
            state = solver.solve(
                state,
                dx=dx_fine,
                T=dt_step,
                cfl=cfl,
                return_all=False,
            )
            state = enforce_physical_state(state)
            states_coarse.append(down(state))
            dt_steps.append(dt_step)
            if show_progress:
                print(
                    f"[Euler pv+flux][{worker_name}] traj {s + 1}/{n_ic} "
                    f"step {step_idx + 1}/{n_steps_per_ic}: solve+downsample done",
                    flush=True,
                )

        xs = np.stack([states_coarse[n] for n in range(n_steps_per_ic)], axis=0)
        ys = np.stack([states_coarse[n + 1] for n in range(n_steps_per_ic)], axis=0)
        dt_steps = np.asarray(dt_steps, dtype=np.float64)  # (n_steps_per_ic,)

        F = np.zeros((n_steps_per_ic, 3, nx_low), dtype=np.float64)
        for k in range(n_steps_per_ic):
            if show_progress:
                print(
                    f"[Euler pv+flux][{worker_name}] traj {s + 1}/{n_ic} "
                    f"step {k + 1}/{n_steps_per_ic}: recover flux (dt={dt_steps[k]:.6e})",
                    flush=True,
                )
            F[k] = _recover_flux_for_bc(xs[k], ys[k], dt_steps[k], dx_low, bc)
            if show_progress:
                print(
                    f"[Euler pv+flux][{worker_name}] traj {s + 1}/{n_ic} "
                    f"step {k + 1}/{n_steps_per_ic}: flux recovered",
                    flush=True,
                )

        X_list.append(xs)
        Y_list.append(ys)
        F_list.append(F)
        DT_list.append(dt_steps)
        labels.extend([ic_label] * n_steps_per_ic)

        if show_progress:
            print(f"[Euler pv+flux][{worker_name}] trajectory {s + 1}/{n_ic}: finished", flush=True)

    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    F_all = np.concatenate(F_list, axis=0)
    DT_all = np.concatenate(DT_list, axis=0)

    if show_progress:
        print(
            f"[Euler pv+flux][{worker_name}] chunk done: n_pairs={X_all.shape[0]}, "
            f"input.shape={X_all.shape}, flux.shape={F_all.shape}, dt.shape={DT_all.shape}",
            flush=True,
        )

    return X_all, Y_all, F_all, DT_all, labels


def _build_pv_flux_split_chunk_from_kwargs(kwargs):
    return _build_pv_flux_split_chunk(**kwargs)


def build_pv_flux_split(
    n_ic,
    seed,
    nx_low=NX,
    upsample=UPSAMPLE,
    T=1.0,
    dt_snap=5e-2,
    dt_min=None,
    dt_max=None,
    cfl=WENO_CFL,
    x_left=DATA_XL,
    x_right=DATA_XR,
    bc=BC,
    reconstruction="characteristic",
    num_workers=1,
):
    if num_workers is None:
        num_workers = 1
    num_workers = max(1, int(num_workers))
    print(
        "[Euler pv+flux] start "
        f"n_ic={n_ic}, nx_low={nx_low}, upsample={upsample}, "
        f"T={T}, dt_snap={dt_snap}, reconstruction={reconstruction}, "
        f"workers={num_workers}",
        flush=True,
    )

    if num_workers == 1 or n_ic <= 1:
        X_all, Y_all, F_all, DT_all, ic_labels = _build_pv_flux_split_chunk(
            n_ic=n_ic,
            seed=seed,
            nx_low=nx_low,
            upsample=upsample,
            T=T,
            dt_snap=dt_snap,
            dt_min=dt_min,
            dt_max=dt_max,
            cfl=cfl,
            x_left=x_left,
            x_right=x_right,
            bc=bc,
            reconstruction=reconstruction,
            show_progress=True,
            worker_name="main",
        )
    else:
        counts = [n_ic // num_workers] * num_workers
        for i in range(n_ic % num_workers):
            counts[i] += 1
        counts = [c for c in counts if c > 0]

        jobs = []
        for i, count in enumerate(counts):
            worker_name = f"worker-{i+1}/{len(counts)}"
            print(
                f"[Euler pv+flux] assign {worker_name}: {count} trajectories",
                flush=True,
            )
            jobs.append(
                dict(
                    n_ic=count,
                    seed=seed + 10000 * i,
                    nx_low=nx_low,
                    upsample=upsample,
                    T=T,
                    dt_snap=dt_snap,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    cfl=cfl,
                    x_left=x_left,
                    x_right=x_right,
                    bc=bc,
                    reconstruction=reconstruction,
                    show_progress=True,
                    worker_name=worker_name,
                )
            )

        results = []
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            future_to_job = {
                ex.submit(_build_pv_flux_split_chunk_from_kwargs, job): job
                for job in jobs
            }
            n_done = 0
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                worker_name = job["worker_name"]
                print(
                    f"[Euler pv+flux] {worker_name} finished, collecting results...",
                    flush=True,
                )
                out = future.result()
                results.append(out)
                n_done += 1
                print(
                    f"[Euler pv+flux] worker progress: {n_done}/{len(jobs)} chunks complete",
                    flush=True,
                )

        X_all = np.concatenate([r[0] for r in results], axis=0)
        Y_all = np.concatenate([r[1] for r in results], axis=0)
        F_all = np.concatenate([r[2] for r in results], axis=0)
        DT_all = np.concatenate([r[3] for r in results], axis=0)
        ic_labels = sum((r[4] for r in results), [])

    nx_fine = nx_low * upsample
    dx_low = (x_right - x_left) / nx_low

    data = {
        "input": torch.tensor(X_all, dtype=torch.float64),
        "output": torch.tensor(Y_all, dtype=torch.float64),
        "flux": torch.tensor(F_all, dtype=torch.float64),
        "dt": torch.tensor(DT_all, dtype=torch.float64),
        "meta": {
            "equation": "1D Euler (rho, rho*u, E)",
            "boundary": bc,
            "reconstruction": reconstruction,
            "domain": [float(x_left), float(x_right)],
            "nx": int(nx_low),
            "dx": float(dx_low),
            # For backward-compatibility: store a representative dt in meta["dt"].
            "dt": float(dt_snap),
            "dt_min": float(dt_min if dt_min is not None else dt_snap),
            "dt_max": float(dt_max if dt_max is not None else dt_snap),
            "gamma": float(GAMMA),
            "T": float(T),
            "n_snaps": int(round(T / dt_snap)) + 1,
            "upsample": int(upsample),
            "nx_fine": int(nx_fine),
            "reference": "point-value IC + FD-WENOZ fine grid + cell-average downsample",
            "weno_cfl": float(cfl),
            "n_ic": int(n_ic),
            "ic_types": ic_labels,
            "num_workers": int(num_workers),
        },
    }
    return data


def generate_pairs_one_ic(
    state0_fine,
    solver,
    dx_fine,
    upsample,
    dt,
    n_steps_per_ic,
    weno_cfl,
    repair_each_step=False,
):
    """
    state0_fine: (3, N_fine)
    Evolve with FD_WENOZ, downsample each snapshot by cell average -> (3, N_coarse)

    Returns
    -------
    xs, ys : (n_steps_per_ic, 3, N_coarse)
    """
    state = np.asarray(state0_fine, dtype=np.float64)
    state = enforce_physical_state(state)

    def down(u):
        return np.stack(
            [downsample_cell_average(u[i], upsample) for i in range(3)],
            axis=0,
        )

    states_coarse = [down(state)]

    for _ in range(n_steps_per_ic):
        state = solver.solve(
            state,
            dx=dx_fine,
            T=dt,
            cfl=weno_cfl,
            return_all=False,
        )

        if repair_each_step:
            state = enforce_physical_state(state)

        states_coarse.append(down(state))

    xs = np.stack([states_coarse[n] for n in range(n_steps_per_ic)], axis=0)
    ys = np.stack([states_coarse[n + 1] for n in range(n_steps_per_ic)], axis=0)
    return xs, ys


def solve_and_plot_sod_lax():
    """
    Solve standard Sod and Lax Riemann problems and plot (rho, u, p)
    on a larger domain using the current default boundary condition.
    """
    nx_plot = NX * UPSAMPLE
    x, dx = make_cell_centered_grid(PLOT_XL, PLOT_XR, nx_plot)

    solver = build_euler_solver(bc=BC)
    times = [0.25, 1.3]
    cases = [("sod", sod_ic), ("lax", lax_ic)]

    out_dir = os.path.join(os.path.dirname(__file__), "riemann_plots")
    os.makedirs(out_dir, exist_ok=True)

    for name, ic_fn in cases:
        u0 = ic_fn(x, x0=PLOT_X0)

        for T in times:
            sol = solver.solve(u0, dx=dx, T=T, cfl=WENO_CFL, return_all=False)
            rho, u_vel, p = conserved_to_primitives(sol)

            fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

            axes[0].plot(x, rho, lw=1.2)
            axes[0].set_ylabel(r"$\rho$")

            axes[1].plot(x, u_vel, lw=1.2)
            axes[1].set_ylabel(r"$u$")

            axes[2].plot(x, p, lw=1.2)
            axes[2].set_ylabel(r"$p$")
            axes[2].set_xlabel("x")

            fig.suptitle(f"{name.capitalize()} Riemann problem, t = {T}")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            fname = f"{name}_t{T:.2f}.png".replace(".", "p")
            fig_path = os.path.join(out_dir, fname)
            plt.savefig(fig_path, dpi=200)
            plt.close(fig)

            print(f"[{name}] saved figure at t={T}: {fig_path}")


def solve_and_plot_smooth_on_unit_interval():
    """
    Smooth initial condition on [0, 1] with NX points.

    Uses smooth_density_perturbation_ic, Euler evolution with FD_WENOZ (outflow /
    transmissive BC), and saves three figures at t = 0, 0.5, 1.0 showing
    (rho, u, p) as functions of x.
    """
    nx = NX
    x, dx = make_cell_centered_grid(0.0, 1.0, nx)

    rng = np.random.default_rng(0)
    u0 = smooth_density_perturbation_ic(x, rng)

    solver = build_euler_solver(bc=BC)
    times = [0.0, 0.5, 1.0]

    out_dir = os.path.join(os.path.dirname(__file__), "smooth_plots")
    os.makedirs(out_dir, exist_ok=True)

    for T in times:
        if T == 0.0:
            sol = u0
        else:
            sol = solver.solve(u0, dx=dx, T=T, cfl=WENO_CFL, return_all=False)

        rho, u_vel, p = conserved_to_primitives(sol)

        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

        axes[0].plot(x, rho, lw=1.2)
        axes[0].set_ylabel(r"$\rho$")

        axes[1].plot(x, u_vel, lw=1.2)
        axes[1].set_ylabel(r"$u$")

        axes[2].plot(x, p, lw=1.2)
        axes[2].set_ylabel(r"$p$")
        axes[2].set_xlabel("x")

        fig.suptitle(f"Smooth IC on [0, 1], t = {T}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        fname = f"smooth_t{T:.2f}.png".replace(".", "p")
        fig_path = os.path.join(out_dir, fname)
        plt.savefig(fig_path, dpi=200)
        plt.close(fig)

        print(f"[smooth] saved figure at t={T}: {fig_path}")


def _downsample_state(u_hi, factor):
    return np.stack(
        [downsample_cell_average(u_hi[k], factor) for k in range(u_hi.shape[0])],
        axis=0,
    )


def _rel_l1(a, b, eps=1e-12):
    return np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + eps)


def _run_riemann_characteristic_experiment(
    case_name,
    ic_fn,
    nx=NX,
    ref_upsample=8,
    T=0.2,
    bc=BC,
    cfl=WENO_CFL,
):
    """
    Compare component-wise vs characteristic-wise WENO on a Riemann problem.

    The reference is a finer-grid characteristic-WENO solution, downsampled back
    to the coarse grid. We save plots and a small metrics file to help check both
    correctness and accuracy.
    """
    x, dx = make_cell_centered_grid(0.0, 1.0, nx)
    x_ref, dx_ref = make_cell_centered_grid(0.0, 1.0, nx * ref_upsample)

    u0 = ic_fn(x, x0=0.5)
    u0_ref = ic_fn(x_ref, x0=0.5)

    solver_comp = build_euler_solver(bc=bc, reconstruction="component")
    solver_char = build_euler_solver(bc=bc, reconstruction="characteristic")
    solver_ref = build_euler_solver(bc=bc, reconstruction="characteristic")

    sol_comp = solver_comp.solve(u0, dx=dx, T=T, cfl=cfl, return_all=False)
    sol_char = solver_char.solve(u0, dx=dx, T=T, cfl=cfl, return_all=False)
    sol_ref_hi = solver_ref.solve(u0_ref, dx=dx_ref, T=T, cfl=cfl, return_all=False)
    sol_ref = _downsample_state(sol_ref_hi, ref_upsample)

    rho_comp, vel_comp, p_comp = conserved_to_primitives(sol_comp)
    rho_char, vel_char, p_char = conserved_to_primitives(sol_char)
    rho_ref, vel_ref, p_ref = conserved_to_primitives(sol_ref)

    metrics = {
        "rho_l1_component": _rel_l1(rho_comp, rho_ref),
        "rho_l1_characteristic": _rel_l1(rho_char, rho_ref),
        "u_l1_component": _rel_l1(vel_comp, vel_ref),
        "u_l1_characteristic": _rel_l1(vel_char, vel_ref),
        "p_l1_component": _rel_l1(p_comp, p_ref),
        "p_l1_characteristic": _rel_l1(p_char, p_ref),
    }

    out_dir = os.path.join(
        os.path.dirname(__file__), f"{case_name}_characteristic_experiment"
    )
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(x, rho_ref, label="ref-char-hi", lw=2.0)
    axes[0].plot(x, rho_comp, label="component", lw=1.2)
    axes[0].plot(x, rho_char, label="characteristic", lw=1.2)
    axes[0].set_ylabel(r"$\rho$")
    axes[0].legend()

    axes[1].plot(x, vel_ref, label="ref-char-hi", lw=2.0)
    axes[1].plot(x, vel_comp, label="component", lw=1.2)
    axes[1].plot(x, vel_char, label="characteristic", lw=1.2)
    axes[1].set_ylabel(r"$u$")
    axes[1].legend()

    axes[2].plot(x, p_ref, label="ref-char-hi", lw=2.0)
    axes[2].plot(x, p_comp, label="component", lw=1.2)
    axes[2].plot(x, p_char, label="characteristic", lw=1.2)
    axes[2].set_ylabel(r"$p$")
    axes[2].set_xlabel("x")
    axes[2].legend()

    fig.suptitle(
        f"{case_name.capitalize()}: component vs characteristic WENO\n"
        f"rho L1: {metrics['rho_l1_component']:.3e} / {metrics['rho_l1_characteristic']:.3e}, "
        f"u L1: {metrics['u_l1_component']:.3e} / {metrics['u_l1_characteristic']:.3e}, "
        f"p L1: {metrics['p_l1_component']:.3e} / {metrics['p_l1_characteristic']:.3e}"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig_path = os.path.join(out_dir, f"{case_name}_component_vs_characteristic.png")
    plt.savefig(fig_path, dpi=220)
    plt.close(fig)

    metrics_path = os.path.join(out_dir, f"{case_name}_metrics.txt")
    with open(metrics_path, "w", encoding="ascii") as f:
        f.write(f"T = {T}\n")
        f.write(f"nx = {nx}\n")
        f.write(f"ref_upsample = {ref_upsample}\n")
        for key, value in metrics.items():
            f.write(f"{key} = {value:.16e}\n")

    print(f"[{case_name}] saved plot: {fig_path}")
    print(f"[{case_name}] saved metrics: {metrics_path}")
    for key, value in metrics.items():
        print(f"[{case_name}] {key} = {value:.6e}")


def run_sod_characteristic_experiment(
    nx=NX,
    ref_upsample=8,
    T=0.2,
    bc=BC,
    cfl=WENO_CFL,
):
    _run_riemann_characteristic_experiment(
        case_name="sod",
        ic_fn=sod_ic,
        nx=nx,
        ref_upsample=ref_upsample,
        T=T,
        bc=bc,
        cfl=cfl,
    )


def run_lax_characteristic_experiment(
    nx=NX,
    ref_upsample=8,
    T=0.2,
    bc=BC,
    cfl=WENO_CFL,
):
    _run_riemann_characteristic_experiment(
        case_name="lax",
        ic_fn=lax_ic,
        nx=nx,
        ref_upsample=ref_upsample,
        T=T,
        bc=bc,
        cfl=cfl,
    )


import argparse
def main():
    # Solve Sod / Lax and also a smooth IC test.
    # solve_and_plot_sod_lax()
    # solve_and_plot_smooth_on_unit_interval()

    # Example for dataset generation:
    # data = build_split(
    #     n_ic=1,
    #     seed=5,
    #     nx=NX,
    #     upsample=UPSAMPLE,
    #     dt=DT,
    #     n_steps_per_ic=N_STEPS_PER_IC,
    #     x_left=DATA_XL,
    #     x_right=DATA_XR,
    #     weno_cfl=WENO_CFL,
    #     bc=BC,
    #     repair_each_step=False,
    # )
    # torch.save(data, "euler1d_train.pt")

    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--dt_snap", type=float, default=5e-2)
    ap.add_argument(
        "--dt_min",
        type=float,
        default=-1.0,
        help="If > 0, sample dt_step ~ U(dt_min, dt_max). If <= 0, use dt_snap (fixed dt).",
    )
    ap.add_argument(
        "--dt_max",
        type=float,
        default=-1.0,
        help="Only used if --dt_min > 0. If <= 0, use dt_snap (fixed dt).",
    )
    ap.add_argument("--nx_low", type=int, default=256)
    ap.add_argument("--upsample", type=int, default=4)
    ap.add_argument("--seed", type=int, default=624538)
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument(
        "--bc",
        type=str,
        default="periodic",
        choices=["periodic", "outflow", "zero", "reflect"],
        help="FD_WENOZ boundary for reference trajectories (outflow = transmissive extrapolation).",
    )
    ap.add_argument(
        "--train_name",
        type=str,
        default="train_pv.pt",
        help="Output .pt filename under out_dir for the train split.",
    )
    ap.add_argument(
        "--val_name",
        type=str,
        default="val_pv.pt",
        help="Output .pt filename under out_dir for the val split.",
    )
    ap.add_argument(
        "--test_name",
        type=str,
        default="test_pv.pt",
        help="Output .pt filename under out_dir for the test split.",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="dataset",
        choices=[
            "dataset",
            "sod_experiment",
            "lax_experiment",
            "smooth_plot",
            "riemann_plot",
        ],
    )
    ap.add_argument(
        "--reconstruction",
        type=str,
        default="characteristic",
        choices=["component", "characteristic"],
    )
    args = ap.parse_args()

    if args.mode == "sod_experiment":
        run_sod_characteristic_experiment(
            nx=args.nx_low,
            ref_upsample=args.upsample,
            T=0.2,
            bc=args.bc,
            cfl=WENO_CFL,
        )
        return

    if args.mode == "lax_experiment":
        run_lax_characteristic_experiment(
            nx=args.nx_low,
            ref_upsample=args.upsample,
            T=0.2,
            bc=args.bc,
            cfl=WENO_CFL,
        )
        return

    if args.mode == "smooth_plot":
        solve_and_plot_smooth_on_unit_interval()
        return

    if args.mode == "riemann_plot":
        solve_and_plot_sod_lax()
        return

    dt_min = args.dt_min if args.dt_min > 0.0 else None
    dt_max = args.dt_max if args.dt_max > 0.0 else None
    bc = args.bc

    print(f"[Euler dataset] generating train split (bc={bc})...", flush=True)
    train_data = build_pv_flux_split(
        n_ic = args.n_train,
        seed = args.seed,
        nx_low = args.nx_low,
        T = args.T,
        upsample = args.upsample,
        dt_snap = args.dt_snap,
        dt_min = dt_min,
        dt_max = dt_max,
        x_left = DATA_XL,
        x_right = DATA_XR,
        cfl = WENO_CFL,
        bc = bc,
        reconstruction = args.reconstruction,
        num_workers = args.num_workers,
    )

    print(f"[Euler dataset] generating val split (bc={bc})...", flush=True)
    val_data = build_pv_flux_split(
        n_ic = args.n_val,
        seed = args.seed + 10000,
        nx_low = args.nx_low,
        T = args.T,
        upsample = args.upsample,
        dt_snap = args.dt_snap,
        dt_min = dt_min,
        dt_max = dt_max,
        x_left = DATA_XL,
        x_right = DATA_XR,
        cfl = WENO_CFL,
        bc = bc,
        reconstruction = args.reconstruction,
        num_workers = args.num_workers,
    )

    print(f"[Euler dataset] generating test split (bc={bc})...", flush=True)
    test_data = build_pv_flux_split(
        n_ic = args.n_test,
        seed = args.seed + 20000,
        nx_low = args.nx_low,
        T = args.T,
        upsample = args.upsample,
        dt_snap = args.dt_snap,
        dt_min = dt_min,
        dt_max = dt_max,
        x_left = DATA_XL,
        x_right = DATA_XR,
        cfl = WENO_CFL,
        bc = bc,
        reconstruction = args.reconstruction,
        num_workers = args.num_workers,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, args.train_name)
    val_path = os.path.join(args.out_dir, args.val_name)
    test_path = os.path.join(args.out_dir, args.test_name)
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    torch.save(test_data, test_path)
    print(f"Saved Euler datasets to: {args.out_dir}", flush=True)
    print(f"  train: {train_path}", flush=True)
    print(f"  val:   {val_path}", flush=True)
    print(f"  test:  {test_path}", flush=True)
    print("Train shapes:", train_data["input"].shape, train_data["output"].shape, train_data["flux"].shape)

if __name__ == "__main__":
    main()