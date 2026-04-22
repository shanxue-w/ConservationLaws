import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from conslaw.solver import FD_WENOZ, downsample_cell_average

G_CONSTANT = 9.80665
H_MIN = 1e-12

NX = 256
UPSAMPLE = 4
DT = 5e-2
N_STEPS_PER_IC = 20
WENO_CFL = 0.4

XL = 0.0
XR = 1.0
L = XR - XL


# =========================
# Basic SWE utilities
# =========================
def primitives_to_conserved(h, u, g=G_CONSTANT):
    h = np.asarray(h, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    h = np.maximum(h, H_MIN)
    return np.stack([h, h * u], axis=0)


def conserved_to_primitives(U, g=G_CONSTANT):
    U = np.asarray(U, dtype=np.float64)
    h = np.maximum(U[0], H_MIN)
    u = U[1] / h
    return h, u


def flux(U, g=G_CONSTANT, h_min=H_MIN):
    h = np.maximum(U[0], h_min)
    q = U[1]
    return np.stack([q, q * q / h + 0.5 * g * h * h], axis=0)


def wave_speed(U, g=G_CONSTANT, h_min=H_MIN):
    h = np.maximum(U[0], h_min)
    u = np.abs(U[1] / h)
    c = np.sqrt(g * h)
    return u + c


def jacobi(U, g=G_CONSTANT, h_min=H_MIN):
    A = np.empty((2, 2, U.shape[1]), dtype=np.float64)
    h = np.maximum(U[0], h_min)
    u = U[1] / h

    A[0, 0] = 0.0
    A[0, 1] = 1.0
    A[1, 0] = g * h - u * u
    A[1, 1] = 2.0 * u
    return A


# =========================
# Characteristic decomposition
# =========================
def swe_characteristic_matrices_batch(UL, UR, g=G_CONSTANT, h_min=H_MIN):
    if UL.ndim != 2 or UR.ndim != 2 or UL.shape[0] != 2 or UR.shape[0] != 2:
        raise ValueError(f"Expected UL, UR with shape (2, Nint); got {UL.shape}, {UR.shape}")

    hL = np.maximum(UL[0], h_min)
    qL = UL[1]
    uL = qL / hL

    hR = np.maximum(UR[0], h_min)
    qR = UR[1]
    uR = qR / hR

    sqrt_hL = np.sqrt(hL)
    sqrt_hR = np.sqrt(hR)
    denom = np.maximum(sqrt_hL + sqrt_hR, 1e-14)

    u_bar = (sqrt_hL * uL + sqrt_hR * uR) / denom
    h_bar = 0.5 * (hL + hR)
    c = np.sqrt(g * np.maximum(h_bar, h_min))
    c_safe = np.maximum(c, 1e-14)

    lam_minus = u_bar - c
    lam_plus = u_bar + c

    Nint = UL.shape[1]
    R = np.empty((Nint, 2, 2), dtype=UL.dtype)
    Lmat = np.empty((Nint, 2, 2), dtype=UL.dtype)

    # Right eigenvectors (columns)
    R[:, 0, 0] = 1.0
    R[:, 1, 0] = u_bar - c
    R[:, 0, 1] = 1.0
    R[:, 1, 1] = u_bar + c

    # Inverse matrix
    Lmat[:, 0, 0] = (u_bar + c_safe) / (2.0 * c_safe)
    Lmat[:, 0, 1] = -0.5 / c_safe
    Lmat[:, 1, 0] = -(u_bar - c_safe) / (2.0 * c_safe)
    Lmat[:, 1, 1] = 0.5 / c_safe

    a_int = np.maximum(np.abs(lam_minus), np.abs(lam_plus))
    return Lmat, R, a_int


# =========================
# Solver builder
# =========================
def build_swe_solver(g=G_CONSTANT, bc="periodic", reconstruction="characteristic", WENOtype="WENO-Z"):
    def _flux(U):
        return flux(U, g=g)

    def _alpha(U):
        return wave_speed(U, g=g)

    if reconstruction not in ("component", "characteristic"):
        raise ValueError('reconstruction must be "component" or "characteristic".')

    char_decomp_batch = None
    if reconstruction == "characteristic":
        char_decomp_batch = lambda uL, uR: swe_characteristic_matrices_batch(uL, uR, g=g)

    return FD_WENOZ(
        flux=_flux,
        dflux=None,
        alpha=_alpha,
        char_decomp_batch=char_decomp_batch,
        n_comp=2,
        flux_split="local_lf",
        eps=1e-20,
        bc=bc,
        WENOtype=WENOtype,
    )


# =========================
# Grid / downsample / flux recovery
# =========================
def _downsample_state(u_hi, factor):
    return np.stack([downsample_cell_average(u_hi[k], factor) for k in range(u_hi.shape[0])], axis=0)


def recover_flux_periodic_np(u, u_next, dt, dx):
    u = np.asarray(u, dtype=np.float64)
    u_next = np.asarray(u_next, dtype=np.float64)

    u_left = np.roll(u, 1, axis=-1)
    u_right = np.roll(u, -1, axis=-1)
    u_avg = 0.5 * (u_left + u_right)

    q = (dx / dt) * (u_avg - u_next)
    q = q - q.mean(axis=-1, keepdims=True)

    flux_rec = np.zeros_like(q)
    flux_rec[..., 1:] = np.cumsum(q[..., 1:], axis=-1)
    flux_rec = flux_rec - flux_rec.mean(axis=-1, keepdims=True)
    return flux_rec


def make_periodic_cell_centers(xl, xr, nx):
    dx = (xr - xl) / nx
    x = xl + dx * np.arange(nx)
    return x, dx


# =========================
# IC helper functions
# =========================
def periodic_distance(x, center, period=L):
    return (x - center + 0.5 * period) % period - 0.5 * period


def random_fourier_series(theta, rng, k_max, decay=2.0, normalize="max"):
    s = np.zeros_like(theta)
    for k in range(1, k_max + 1):
        scale = 1.0 / (k ** decay)
        a = rng.normal(0.0, scale)
        b = rng.normal(0.0, scale)
        s += a * np.cos(k * theta) + b * np.sin(k * theta)

    s -= np.mean(s)

    if normalize == "std":
        s_std = np.std(s)
        if s_std > 1e-14:
            s /= s_std
    elif normalize == "max":
        s_max = np.max(np.abs(s))
        if s_max > 1e-14:
            s /= s_max
    else:
        raise ValueError("normalize must be 'max' or 'std'.")

    return s


# =========================
# Three IC families
# =========================
def bidirectional_smooth_ic(
    x,
    h0=1.0,
    u0=0.0,
    amp_ratio_range=(0.08, 0.22),
    modes=4,
    g=G_CONSTANT,
    h_floor=0.05,
    rng=None,
):
    """
    Weak / clean family:
    build smooth waves in the SWE Riemann variables.
    """
    x = np.asarray(x, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng()

    theta = 2.0 * np.pi * (x - XL) / L
    c0 = np.sqrt(g * h0)

    wp0 = u0 + 2.0 * c0
    wm0 = u0 - 2.0 * c0

    k_max = int(rng.integers(2, max(3, modes) + 1))
    wp_shape = random_fourier_series(theta, rng, k_max=k_max, decay=2.0, normalize="max")
    wm_shape = random_fourier_series(theta, rng, k_max=k_max, decay=2.0, normalize="max")

    amp_ratio = rng.uniform(*amp_ratio_range)
    amp = amp_ratio * c0

    wp = wp0 + amp * wp_shape
    wm = wm0 + amp * wm_shape

    u = 0.5 * (wp + wm)
    c = 0.25 * (wp - wm)
    c = np.maximum(c, np.sqrt(g * h_floor))
    h = (c * c) / g

    return primitives_to_conserved(h, u, g=g)


def coupled_multiscale_ic(
    x,
    h0=1.0,
    u0=0.0,
    h_eps_range=(0.10, 0.22),
    u_amp_ratio_range=(0.15, 0.45),
    min_slope=1.0,
    max_slope=3.0,
    max_mode=4,
    h_floor=0.05,
    g=G_CONSTANT,
    rng=None,
):
    """
    Medium-strength family:
    multiscale smooth perturbations in both h and u.
    """
    x = np.asarray(x, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng()

    theta = 2.0 * np.pi * (x - XL) / L
    c0 = np.sqrt(g * h0)

    p_h = rng.uniform(min_slope, max_slope)
    p_u = rng.uniform(min_slope, max_slope)
    k_max_h = int(rng.integers(2, max_mode + 1))
    k_max_u = int(rng.integers(2, max_mode + 1))

    s_h = random_fourier_series(theta, rng, k_max=k_max_h, decay=p_h, normalize="std")
    s_u = random_fourier_series(theta, rng, k_max=k_max_u, decay=p_u, normalize="max")

    h_eps = rng.uniform(*h_eps_range)
    u_amp = rng.uniform(*u_amp_ratio_range) * c0

    h = h0 + h_eps * s_h
    h = np.maximum(h, h_floor)
    u = u0 + u_amp * s_u

    return primitives_to_conserved(h, u, g=g)


def periodic_smoothed_riemann_ic(
    x,
    h_in_range=(1.2, 2.0),
    h_out_range=(0.2, 0.9),
    u_in_range=(-0.8, 0.8),
    u_out_range=(-0.8, 0.8),
    half_width_range=(0.08, 0.22),
    transition_width_range=(0.01, 0.04),
    g=G_CONSTANT,
    h_floor=0.05,
    rng=None,
):
    """
    Stronger family:
    periodic analogue of a dam-break / Riemann profile with two smooth interfaces.
    """
    x = np.asarray(x, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng()

    center = rng.uniform(XL, XR)
    half_width = rng.uniform(*half_width_range) * L
    trans_w = rng.uniform(*transition_width_range) * L
    trans_w = max(trans_w, 1e-4)

    h_in = rng.uniform(*h_in_range)
    h_out = rng.uniform(*h_out_range)
    if abs(h_in - h_out) < 0.25:
        if h_in >= h_out:
            h_in = min(h_in_range[1], h_out + 0.25)
        else:
            h_out = min(h_out_range[1], h_in + 0.25)

    u_in = rng.uniform(*u_in_range)
    u_out = rng.uniform(*u_out_range)

    d = np.abs(periodic_distance(x, center, L))
    indicator = 0.5 * (1.0 + np.tanh((half_width - d) / trans_w))

    if rng.uniform() < 0.5:
        h = h_out + (h_in - h_out) * indicator
        u = u_out + (u_in - u_out) * indicator
    else:
        h = h_in + (h_out - h_in) * indicator
        u = u_in + (u_out - u_in) * indicator

    h = np.maximum(h, h_floor)
    return primitives_to_conserved(h, u, g=g)


def sample_mixture_ic(x, rng=None):
    """
    Three-family mixture:
      - 30% bidirectional_smooth
      - 40% coupled_multiscale
      - 30% periodic_smoothed_riemann
    """
    if rng is None:
        rng = np.random.default_rng()

    r = rng.uniform()
    if r < 0.30:
        return bidirectional_smooth_ic(x, rng=rng), "bidirectional_smooth"
    elif r < 0.70:
        return coupled_multiscale_ic(x, rng=rng), "coupled_multiscale"
    else:
        return periodic_smoothed_riemann_ic(x, rng=rng), "periodic_smoothed_riemann"


# =========================
# Time evolution / dataset assembly
# =========================
def generate_trajectory_swe(state0_fine, solver, T, dt_snap, dx_fine, cfl=WENO_CFL):
    state = np.asarray(state0_fine, dtype=np.float64).copy()
    times = np.arange(0.0, T + 1e-12, dt_snap)
    snaps = [state.copy()]
    t = 0.0

    for i in range(1, len(times)):
        t_target = times[i]
        dt_interval = float(t_target - t)
        state = solver.solve(
            state,
            dx=dx_fine,
            T=dt_interval,
            cfl=cfl,
            return_all=False,
        )
        snaps.append(np.asarray(state, dtype=np.float64).copy())
        t = t_target

    return np.stack(snaps, axis=0), times


def generate_pairs_one_ic(state0_fine, solver, dx_fine, upsample, dt, n_steps_per_ic, weno_cfl):
    state = np.asarray(state0_fine, dtype=np.float64)

    def down(u):
        return _downsample_state(u, upsample)

    states_coarse = [down(state)]
    for _ in range(n_steps_per_ic):
        state = solver.solve(
            state,
            dx=dx_fine,
            T=dt,
            cfl=weno_cfl,
            return_all=False,
        )
        states_coarse.append(down(state))

    xs = np.stack([states_coarse[n] for n in range(n_steps_per_ic)], axis=0)
    ys = np.stack([states_coarse[n + 1] for n in range(n_steps_per_ic)], axis=0)
    return xs, ys


def _build_pv_flux_split_chunk(
    n_ic,
    seed,
    nx_low=NX,
    upsample=UPSAMPLE,
    T=1.0,
    dt_snap=DT,
    cfl=WENO_CFL,
    x_left=XL,
    x_right=XR,
    bc="periodic",
    reconstruction="characteristic",
    show_progress=True,
    worker_name="worker",
):
    rng = np.random.default_rng(seed)
    nx_fine = nx_low * upsample
    x_fine, dx_fine = make_periodic_cell_centers(x_left, x_right, nx_fine)

    solver = build_swe_solver(bc=bc, reconstruction=reconstruction)
    n_steps_per_ic = int(round(T / dt_snap))

    if show_progress:
        print(
            f"[SWE pv+flux][{worker_name}] chunk start: n_ic={n_ic}, nx_fine={nx_fine}, "
            f"n_steps_per_ic={n_steps_per_ic}, dx_fine={dx_fine:.6e}",
            flush=True,
        )

    X_list = []
    Y_list = []
    F_list = []
    labels = []

    for s in range(n_ic):
        if show_progress:
            print(f"[SWE pv+flux][{worker_name}] trajectory {s + 1}/{n_ic}: start", flush=True)

        state0_fine, ic_label = sample_mixture_ic(x_fine, rng=rng)

        xs, ys = generate_pairs_one_ic(
            state0_fine=state0_fine,
            solver=solver,
            dx_fine=dx_fine,
            upsample=upsample,
            dt=dt_snap,
            n_steps_per_ic=n_steps_per_ic,
            weno_cfl=cfl,
        )

        dx_low = (x_right - x_left) / nx_low
        F = np.zeros((n_steps_per_ic, 2, nx_low), dtype=np.float64)
        for k in range(n_steps_per_ic):
            F[k] = recover_flux_periodic_np(xs[k], ys[k], dt_snap, dx_low)

        X_list.append(xs)
        Y_list.append(ys)
        F_list.append(F)
        labels.extend([ic_label] * n_steps_per_ic)

        if show_progress:
            print(f"[SWE pv+flux][{worker_name}] trajectory {s + 1}/{n_ic}: finished", flush=True)

    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    F_all = np.concatenate(F_list, axis=0)
    n_pairs = X_all.shape[0]
    if show_progress:
        print(
            f"[SWE pv+flux][{worker_name}] chunk done: n_pairs={n_pairs}, "
            f"input.shape={X_all.shape}, flux.shape={F_all.shape}",
            flush=True,
        )
    return X_all, Y_all, F_all, labels


def _build_pv_flux_split_chunk_from_kwargs(kwargs):
    return _build_pv_flux_split_chunk(**kwargs)


def build_pv_flux_split(
    n_ic,
    seed,
    nx_low=NX,
    upsample=UPSAMPLE,
    T=1.0,
    dt_snap=DT,
    cfl=WENO_CFL,
    x_left=XL,
    x_right=XR,
    bc="periodic",
    reconstruction="characteristic",
    num_workers=1,
):
    if num_workers is None:
        num_workers = 1
    num_workers = max(1, int(num_workers))

    print(
        "[SWE pv+flux] start "
        f"n_ic={n_ic}, nx_low={nx_low}, upsample={upsample}, "
        f"T={T}, dt_snap={dt_snap}, reconstruction={reconstruction}, "
        f"workers={num_workers}",
        flush=True,
    )

    if num_workers == 1 or n_ic <= 1:
        print("[SWE pv+flux] running single worker (main process)", flush=True)
        X_all, Y_all, F_all, ic_labels = _build_pv_flux_split_chunk(
            n_ic=n_ic,
            seed=seed,
            nx_low=nx_low,
            upsample=upsample,
            T=T,
            dt_snap=dt_snap,
            cfl=cfl,
            x_left=x_left,
            x_right=x_right,
            bc=bc,
            reconstruction=reconstruction,
            show_progress=True,
            worker_name="main",
        )
        n_pairs = X_all.shape[0]
        print(
            f"[SWE pv+flux] main chunk complete: n_pairs={n_pairs}, "
            f"input.shape={X_all.shape}",
            flush=True,
        )
    else:
        counts = [n_ic // num_workers] * num_workers
        for i in range(n_ic % num_workers):
            counts[i] += 1
        counts = [c for c in counts if c > 0]

        jobs = []
        for i, count in enumerate(counts):
            worker_name = f"worker-{i + 1}/{len(counts)}"
            print(
                f"[SWE pv+flux] assign {worker_name}: {count} trajectories",
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
                    f"[SWE pv+flux] {worker_name} finished, collecting results...",
                    flush=True,
                )
                out = future.result()
                results.append(out)
                n_done += 1
                print(
                    f"[SWE pv+flux] worker progress: {n_done}/{len(jobs)} chunks complete",
                    flush=True,
                )

        X_all = np.concatenate([r[0] for r in results], axis=0)
        Y_all = np.concatenate([r[1] for r in results], axis=0)
        F_all = np.concatenate([r[2] for r in results], axis=0)
        ic_labels = sum((r[3] for r in results), [])
        n_pairs = X_all.shape[0]
        print(
            f"[SWE pv+flux] all workers done. total n_pairs={n_pairs}, "
            f"input.shape={X_all.shape}",
            flush=True,
        )

    nx_fine = nx_low * upsample
    dx_low = (x_right - x_left) / nx_low

    data = {
        "input": torch.tensor(X_all, dtype=torch.float64),
        "output": torch.tensor(Y_all, dtype=torch.float64),
        "flux": torch.tensor(F_all, dtype=torch.float64),
        "meta": {
            "equation": "1D shallow water (h, hu)",
            "boundary": bc,
            "reconstruction": reconstruction,
            "domain": [float(x_left), float(x_right)],
            "nx": int(nx_low),
            "dx": float(dx_low),
            "dt": float(dt_snap),
            "g": float(G_CONSTANT),
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
    print(
        f"[SWE pv+flux] build_pv_flux_split done: returning data with "
        f"input.shape={data['input'].shape}, output.shape={data['output'].shape}, "
        f"flux.shape={data['flux'].shape}",
        flush=True,
    )
    return data


# =========================
# Visualization
# =========================
def plot_smooth_periodic_evolution(
    out_dir,
    nx=256,
    T=0.5,
    dt_snap=5e-2,
    seed=None,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"[SWE plot] start: nx={nx}, T={T}, dt_snap={dt_snap}, seed={seed}", flush=True)

    x, dx = make_periodic_cell_centers(XL, XR, nx)
    solver = build_swe_solver(bc="periodic", reconstruction="characteristic")

    rng = np.random.default_rng(seed)
    state0, ic_label = sample_mixture_ic(x, rng=rng)
    print(f"[SWE plot] IC type: {ic_label}", flush=True)

    snaps, times = generate_trajectory_swe(
        state0,
        solver,
        T=T,
        dt_snap=dt_snap,
        dx_fine=dx,   # fixed bug here
        cfl=WENO_CFL,
    )

    idxs = np.linspace(0, len(times) - 1, num=4, dtype=int)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for k in idxs:
        h, u = conserved_to_primitives(snaps[k])
        label = f"t={times[k]:.2f}"
        axes[0].plot(x, h, lw=1.2, label=label)
        axes[1].plot(x, u, lw=1.2, label=label)

    axes[0].set_ylabel("h")
    axes[0].legend()
    axes[1].set_ylabel("u")
    axes[1].set_xlabel("x")

    fig.suptitle(f"SWE evolution, mixture IC: {ic_label}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(out_dir, f"smooth_periodic_evolution_{seed}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[SWE plot] saved figure: {fig_path}", flush=True)


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="plot", choices=["plot", "dataset"])
    ap.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), "./"))

    # plotting params
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--dt_snap", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=0)

    # dataset params
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--nx_low", type=int, default=256)
    ap.add_argument("--upsample", type=int, default=4)
    ap.add_argument("--T_dataset", type=float, default=1.0)
    ap.add_argument("--dt_snap_dataset", type=float, default=5e-2)
    ap.add_argument(
        "--num_workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    args = ap.parse_args()

    if args.mode == "plot":
        plot_smooth_periodic_evolution(
            out_dir=args.out_dir,
            nx=args.nx,
            T=args.T,
            dt_snap=args.dt_snap,
            seed=args.seed,
        )
        exit()

    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"[SWE dataset] plan: n_train={args.n_train}, n_val={args.n_val}, n_test={args.n_test}, "
        f"nx_low={args.nx_low}, upsample={args.upsample}, T={args.T_dataset}, "
        f"dt_snap={args.dt_snap_dataset}, num_workers={args.num_workers}, out_dir={args.out_dir}",
        flush=True,
    )

    print("[SWE dataset] generating train split...", flush=True)
    train_data = build_pv_flux_split(
            n_ic=args.n_train,
            seed=args.seed,
            nx_low=args.nx_low,
            upsample=args.upsample,
            T=args.T_dataset,
            dt_snap=args.dt_snap_dataset,
            cfl=WENO_CFL,
            x_left=XL,
            x_right=XR,
            bc="periodic",
            reconstruction="characteristic",
            num_workers=args.num_workers,
        )
    print("[SWE dataset] generating val split...", flush=True)
    val_data = build_pv_flux_split(
            n_ic=args.n_val,
            seed=args.seed + 10000,
            nx_low=args.nx_low,
            upsample=args.upsample,
            T=args.T_dataset,
            dt_snap=args.dt_snap_dataset,
            cfl=WENO_CFL,
            x_left=XL,
            x_right=XR,
            bc="periodic",
            reconstruction="characteristic",
            num_workers=args.num_workers,
        )
    print("[SWE dataset] generating test split...", flush=True)
    test_data = build_pv_flux_split(
            n_ic=args.n_test,
            seed=args.seed + 20000,
            nx_low=args.nx_low,
            upsample=args.upsample,
            T=args.T_dataset,
            dt_snap=args.dt_snap_dataset,
            cfl=WENO_CFL,
            x_left=XL,
            x_right=XR,
            bc="periodic",
            reconstruction="characteristic",
            num_workers=args.num_workers,
        )

    torch.save(train_data, os.path.join(args.out_dir, "train_pv.pt"))
    torch.save(val_data, os.path.join(args.out_dir, "val_pv.pt"))
    torch.save(test_data, os.path.join(args.out_dir, "test_pv.pt"))

    print(f"Saved SWE datasets to: {args.out_dir}", flush=True)
    print("Files: train_pv.pt, val_pv.pt, test_pv.pt", flush=True)
    print(
        "Train shapes:",
        train_data["input"].shape,
        train_data["output"].shape,
        train_data["flux"].shape,
        flush=True,
    )
    print(
        "Val shapes:",
        val_data["input"].shape,
        val_data["output"].shape,
        val_data["flux"].shape,
        flush=True,
    )
    print(
        "Test shapes:",
        test_data["input"].shape,
        test_data["output"].shape,
        test_data["flux"].shape,
        flush=True,
    )