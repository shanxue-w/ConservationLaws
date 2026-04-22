"""
2D Burgers dataset on a periodic square:

    u_t + (u^2 / 2)_x + (u^2 / 2)_y = 0

Point-value initial conditions are evolved on a fine grid with FD_WENOZ2D, then
downsampled by conservative cell averages. The learned target is a pair of
periodic flux fields (Fx, Fy) recovered with a canonical Fourier-space gauge.

This dataset uses the "no-avg" conservative update for flux recovery:
    u_next = u - dt * (Dxb Fx + Dyb Fy)
instead of the earlier variant that used u_avg (neighbor average).
"""

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from conslaw.solver import (
    FD_WENOZ2D,
    downsample_cell_average2d,
    get_numba_thread_count,
    set_numba_thread_count,
)


def make_periodic_grid(nx, ny, lx=1.0, ly=1.0):
    x = np.linspace(0.0, lx, nx, endpoint=False, dtype=np.float64)
    y = np.linspace(0.0, ly, ny, endpoint=False, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return xx, yy, lx / nx, ly / ny


def random_fourier_ic2d(xx, yy, kmax=4, amp_range=(0.4, 1.0), decay=1.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    kx_max = int(rng.integers(0, kmax + 1))
    ky_max = int(rng.integers(0, kmax + 1))
    while kx_max == 0 and ky_max == 0:
        kx_max = int(rng.integers(0, kmax + 1))
        ky_max = int(rng.integers(0, kmax + 1))

    u = np.zeros_like(xx, dtype=np.float64)
    for ky in range(0, ky_max + 1):
        for kx in range(0, kx_max + 1):
            if kx == 0 and ky == 0:
                continue
            knorm = np.sqrt(kx * kx + ky * ky)
            scale = 1.0 / max(knorm, 1.0) ** decay
            amp = rng.normal(0.0, scale)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            u += amp * np.cos(2.0 * np.pi * (kx * xx + ky * yy) + phase)

    umax = np.max(np.abs(u)) + 1e-12
    target_amp = rng.uniform(amp_range[0], amp_range[1])
    u = u / umax * target_amp
    u += rng.uniform(-0.5, 0.5) * target_amp
    return u.astype(np.float64)


def build_burgers2d_solver(use_numba=True):
    def flux(u):
        return 0.5 * u * u

    def alpha(u):
        return np.abs(u)

    return FD_WENOZ2D(
        flux_x=flux,
        flux_y=flux,
        alpha_x=alpha,
        alpha_y=alpha,
        flux_split="local_lf",
        bc="periodic",
        WENOtype="WENO-Z",
        numba_backend="scalar_periodic_burgers" if use_numba else None,
    )


def _warm_numba_burgers2d(nx_fine, ny_fine, dt_snap, cfl, lx=1.0, ly=1.0):
    solver = build_burgers2d_solver(use_numba=True)
    _, _, dx_fine, dy_fine = make_periodic_grid(nx_fine, ny_fine, lx=lx, ly=ly)
    u0 = np.zeros((ny_fine, nx_fine), dtype=np.float64)
    solver.advance(
        u0,
        dx=dx_fine,
        dy=dy_fine,
        T=min(float(dt_snap), 1e-3),
        cfl=cfl,
        return_all=False,
    )


def _init_worker_runtime(use_numba, numba_threads):
    pid = os.getpid()
    if use_numba:
        actual = set_numba_thread_count(numba_threads)
        print(
            f"[Burgers2D dataset][worker-init pid={pid}] numba enabled with {actual} thread(s)",
            flush=True,
        )
    else:
        actual = get_numba_thread_count()
        print(
            f"[Burgers2D dataset][worker-init pid={pid}] numba disabled; runtime threads={actual}",
            flush=True,
        )


def generate_trajectory(u0_fine, solver, T, dt_snap, dx_fine, dy_fine, cfl=0.4):
    times = np.arange(0.0, T + 1e-12, dt_snap)
    snaps = solver.solve_snapshots(
        u0_fine,
        dx=dx_fine,
        dy=dy_fine,
        dt_snap=dt_snap,
        n_snaps=len(times),
        cfl=cfl,
    )
    return snaps, times


def _backward_divergence_symbols(ny, nx, dy, dx):
    theta_y = 2.0 * np.pi * np.fft.fftfreq(ny)
    theta_x = 2.0 * np.pi * np.fft.fftfreq(nx)
    dy_sym = (1.0 - np.exp(-1j * theta_y)) / dy
    dx_sym = (1.0 - np.exp(-1j * theta_x)) / dx
    dy_grid, dx_grid = np.meshgrid(dy_sym, dx_sym, indexing="ij")
    return dy_grid, dx_grid


def make_flux_recovery_cache2d(ny, nx, dy, dx):
    dy_sym, dx_sym = _backward_divergence_symbols(ny, nx, dy, dx)
    denom = np.abs(dx_sym) ** 2 + np.abs(dy_sym) ** 2
    denom[0, 0] = 1.0
    return {
        "dy_sym": dy_sym,
        "dx_sym": dx_sym,
        "denom": denom,
    }


def recover_zero_mean_flux2d(u, u_next, dt, dx, dy, cache=None):
    """
    Recover a canonical periodic flux pair (Fx, Fy) from the 2D conservative step

        u_next = u - dt * (Dxb Fx + Dyb Fy)

    The recovered flux pair is the minimal-norm periodic solution in Fourier space.
    """
    u = np.asarray(u, dtype=np.float64)
    u_next = np.asarray(u_next, dtype=np.float64)
    squeeze_out = False
    if u.ndim == 2:
        u = u[None, ...]
        u_next = u_next[None, ...]
        squeeze_out = True

    # Discrete divergence relation under the same backward-difference symbols
    # used by `recover_zero_mean_flux2d`'s Fourier recovery:
    #   q = (u - u_next)/dt = Dxb Fx + Dyb Fy
    q = (u - u_next) / dt
    q = q - q.mean(axis=(-2, -1), keepdims=True)

    batch, ny, nx = q.shape
    if cache is None:
        cache = make_flux_recovery_cache2d(ny, nx, dy, dx)
    dy_sym = cache["dy_sym"]
    dx_sym = cache["dx_sym"]
    denom = cache["denom"]

    q_hat = np.fft.fftn(q, axes=(-2, -1))
    fx_hat = (np.conj(dx_sym)[None, ...] / denom[None, ...]) * q_hat
    fy_hat = (np.conj(dy_sym)[None, ...] / denom[None, ...]) * q_hat
    fx_hat[:, 0, 0] = 0.0
    fy_hat[:, 0, 0] = 0.0

    fx = np.fft.ifftn(fx_hat, axes=(-2, -1)).real
    fy = np.fft.ifftn(fy_hat, axes=(-2, -1)).real
    flux = np.stack([fx, fy], axis=1)
    return flux[0] if squeeze_out else flux


def build_split(
    n_ic,
    seed,
    nx_low=64,
    ny_low=64,
    upsample=4,
    T=0.3,
    dt_snap=2e-2,
    cfl=0.4,
    kmax=4,
    decay=1.5,
    amp_min=0.4,
    amp_max=1.0,
    lx=1.0,
    ly=1.0,
    num_workers=1,
    use_numba="auto",
):
    if num_workers is None:
        num_workers = 1
    num_workers = max(1, int(num_workers))

    if use_numba not in ("auto", "on", "off"):
        raise ValueError('use_numba must be one of: "auto", "on", "off".')

    if use_numba == "on":
        use_numba_main = True
        use_numba_workers = num_workers > 1
    elif use_numba == "off":
        use_numba_main = False
        use_numba_workers = False
    else:
        use_numba_main = True
        use_numba_workers = False

    effective_num_workers = num_workers
    cpu_count = os.cpu_count() or 1
    worker_numba_threads = 1
    if num_workers > 1 and use_numba == "auto":
        print(
            "[Burgers2D dataset] auto mode: disabling numba inside worker processes "
            "to avoid BrokenProcessPool from nested process/JIT parallelism",
            flush=True,
        )
    elif num_workers > 1 and use_numba == "on":
        worker_numba_threads = max(1, cpu_count // effective_num_workers)
        print(
            "[Burgers2D dataset] hybrid mode: enabling numba inside worker processes "
            f"with {worker_numba_threads} thread(s) per worker",
            flush=True,
        )

    print(
        "[Burgers2D dataset] start "
        f"n_ic={n_ic}, nx_low={nx_low}, ny_low={ny_low}, upsample={upsample}, "
        f"T={T}, dt_snap={dt_snap}, workers={num_workers}, use_numba={use_numba}, "
        f"effective_workers={effective_num_workers}, "
        f"main_numba={use_numba_main}, worker_numba={use_numba_workers}, "
        f"worker_numba_threads={worker_numba_threads}",
        flush=True,
    )

    if effective_num_workers == 1 or n_ic <= 1:
        X_all, Y_all, F_all = _build_split_chunk(
            n_ic=n_ic,
            seed=seed,
            nx_low=nx_low,
            ny_low=ny_low,
            upsample=upsample,
            T=T,
            dt_snap=dt_snap,
            cfl=cfl,
            kmax=kmax,
            decay=decay,
            amp_min=amp_min,
            amp_max=amp_max,
            lx=lx,
            ly=ly,
            show_progress=True,
            worker_name="main",
            use_numba=use_numba_main,
        )
    else:
        counts = [n_ic // effective_num_workers] * effective_num_workers
        for i in range(n_ic % effective_num_workers):
            counts[i] += 1
        counts = [c for c in counts if c > 0]

        jobs = []
        for i, count in enumerate(counts):
            worker_name = f"worker-{i+1}/{len(counts)}"
            print(
                f"[Burgers2D dataset] assign {worker_name}: {count} trajectories",
                flush=True,
            )
            jobs.append(
                dict(
                    n_ic=count,
                    seed=seed + 10000 * i,
                    nx_low=nx_low,
                    ny_low=ny_low,
                    upsample=upsample,
                    T=T,
                    dt_snap=dt_snap,
                    cfl=cfl,
                    kmax=kmax,
                    decay=decay,
                    amp_min=amp_min,
                    amp_max=amp_max,
                    lx=lx,
                    ly=ly,
                    show_progress=True,
                    worker_name=worker_name,
                    use_numba=use_numba_workers,
                )
            )

        results = []
        if use_numba_workers:
            print(
                "[Burgers2D dataset] warming numba kernels before spawning workers...",
                flush=True,
            )
            set_numba_thread_count(worker_numba_threads)
            _warm_numba_burgers2d(
                nx_low * upsample,
                ny_low * upsample,
                dt_snap=dt_snap,
                cfl=cfl,
                lx=lx,
                ly=ly,
            )
            print(
                f"[Burgers2D dataset] numba warmup done; main process threads={get_numba_thread_count()}",
                flush=True,
            )

        mp_ctx = mp.get_context("fork") if hasattr(os, "fork") else None
        with ProcessPoolExecutor(
            max_workers=len(jobs),
            mp_context=mp_ctx,
            initializer=_init_worker_runtime,
            initargs=(use_numba_workers, worker_numba_threads),
        ) as ex:
            future_to_job = {
                ex.submit(_build_split_chunk_from_kwargs, job): job
                for job in jobs
            }
            n_done = 0
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                worker_name = job["worker_name"]
                print(
                    f"[Burgers2D dataset] {worker_name} finished, collecting results...",
                    flush=True,
                )
                out = future.result()
                results.append(out)
                n_done += 1
                print(
                    f"[Burgers2D dataset] worker progress: {n_done}/{len(jobs)} chunks complete",
                    flush=True,
                )

        X_all = np.concatenate([r[0] for r in results], axis=0)
        Y_all = np.concatenate([r[1] for r in results], axis=0)
        F_all = np.concatenate([r[2] for r in results], axis=0)

    nx_fine = nx_low * upsample
    ny_fine = ny_low * upsample
    dx_low = lx / nx_low
    dy_low = ly / ny_low
    n_snaps = int(round(T / dt_snap)) + 1
    data = {
        "input": torch.tensor(X_all, dtype=torch.float64),
        "output": torch.tensor(Y_all, dtype=torch.float64),
        "flux": torch.tensor(F_all, dtype=torch.float64),
        "meta": {
            "equation": "u_t + (u^2/2)_x + (u^2/2)_y = 0",
            "boundary": "periodic",
            "domain": [0.0, lx, 0.0, ly],
            "nx": int(nx_low),
            "ny": int(ny_low),
            "dx": float(dx_low),
            "dy": float(dy_low),
            "dt": float(dt_snap),
            "T": float(T),
            "n_snaps": int(n_snaps),
            "upsample": int(upsample),
            "nx_fine": int(nx_fine),
            "ny_fine": int(ny_fine),
            "reference": "point-value IC + FD_WENOZ2D fine grid + cell-average downsample",
            "cfl": float(cfl),
            "num_workers": int(num_workers),
            "effective_num_workers": int(effective_num_workers),
            "use_numba": use_numba,
            "numba_backend": "scalar_periodic_burgers" if (use_numba_main or use_numba_workers) else "disabled",
        },
    }
    return data


def _finalize_split_data(
    split_name,
    outputs,
    nx_low,
    ny_low,
    upsample,
    T,
    dt_snap,
    cfl,
    lx,
    ly,
    requested_num_workers,
    effective_num_workers,
    use_numba,
    use_numba_main,
    use_numba_workers,
):
    if outputs:
        X_all = np.concatenate([r[0] for r in outputs], axis=0)
        Y_all = np.concatenate([r[1] for r in outputs], axis=0)
        F_all = np.concatenate([r[2] for r in outputs], axis=0)
    else:
        n_snaps = int(round(T / dt_snap)) + 1
        n_pairs = 0
        X_all = np.zeros((n_pairs, ny_low, nx_low), dtype=np.float64)
        Y_all = np.zeros_like(X_all)
        F_all = np.zeros((n_pairs, 2, ny_low, nx_low), dtype=np.float64)

    nx_fine = nx_low * upsample
    ny_fine = ny_low * upsample
    dx_low = lx / nx_low
    dy_low = ly / ny_low
    n_snaps = int(round(T / dt_snap)) + 1
    return {
        "input": torch.tensor(X_all, dtype=torch.float64),
        "output": torch.tensor(Y_all, dtype=torch.float64),
        "flux": torch.tensor(F_all, dtype=torch.float64),
        "meta": {
            "split": split_name,
            "equation": "u_t + (u^2/2)_x + (u^2/2)_y = 0",
            "boundary": "periodic",
            "domain": [0.0, lx, 0.0, ly],
            "nx": int(nx_low),
            "ny": int(ny_low),
            "dx": float(dx_low),
            "dy": float(dy_low),
            "dt": float(dt_snap),
            "T": float(T),
            "n_snaps": int(n_snaps),
            "upsample": int(upsample),
            "nx_fine": int(nx_fine),
            "ny_fine": int(ny_fine),
            "reference": "point-value IC + FD_WENOZ2D fine grid + cell-average downsample",
            "cfl": float(cfl),
            "num_workers": int(requested_num_workers),
            "effective_num_workers": int(effective_num_workers),
            "use_numba": use_numba,
            "numba_backend": "scalar_periodic_burgers" if (use_numba_main or use_numba_workers) else "disabled",
        },
    }


def _build_split_chunk_from_kwargs(kwargs):
    kwargs = dict(kwargs)
    kwargs.pop("split_name", None)
    return _build_split_chunk(**kwargs)


def _collect_parallel_jobs(
    jobs,
    max_workers,
    use_numba_workers,
    worker_numba_threads,
    warmup_kwargs=None,
    on_result=None,
):
    results = [] if on_result is None else None

    if use_numba_workers and warmup_kwargs is not None:
        print(
            "[Burgers2D dataset] warming numba kernels before spawning workers...",
            flush=True,
        )
        set_numba_thread_count(worker_numba_threads)
        _warm_numba_burgers2d(**warmup_kwargs)
        print(
            f"[Burgers2D dataset] numba warmup done; main process threads={get_numba_thread_count()}",
            flush=True,
        )

    mp_ctx = mp.get_context("fork") if hasattr(os, "fork") else None
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
        initializer=_init_worker_runtime,
        initargs=(use_numba_workers, worker_numba_threads),
    ) as ex:
        future_to_job = {
            ex.submit(_build_split_chunk_from_kwargs, job): job
            for job in jobs
        }
        n_done = 0
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            worker_name = job["worker_name"]
            split_name = job.get("split_name", "split")
            print(
                f"[Burgers2D dataset][{split_name}] {worker_name} finished, collecting results...",
                flush=True,
            )
            out = future.result()
            if on_result is None:
                results.append((split_name, out))
            else:
                on_result(split_name, out)
            n_done += 1
            print(
                f"[Burgers2D dataset] worker progress: {n_done}/{len(jobs)} chunks complete",
                flush=True,
            )
    return [] if on_result is not None else results


def _build_split_chunk(
    n_ic,
    seed,
    nx_low=64,
    ny_low=64,
    upsample=4,
    T=0.3,
    dt_snap=2e-2,
    cfl=0.4,
    kmax=4,
    decay=1.5,
    amp_min=0.4,
    amp_max=1.0,
    lx=1.0,
    ly=1.0,
    show_progress=True,
    worker_name="worker",
    use_numba=True,
):
    rng = np.random.default_rng(seed)
    nx_fine = nx_low * upsample
    ny_fine = ny_low * upsample
    xx_fine, yy_fine, dx_fine, dy_fine = make_periodic_grid(nx_fine, ny_fine, lx=lx, ly=ly)
    dx_low = lx / nx_low
    dy_low = ly / ny_low
    solver = build_burgers2d_solver(use_numba=use_numba)
    flux_cache = make_flux_recovery_cache2d(ny_low, nx_low, dy_low, dx_low)

    n_snaps = int(round(T / dt_snap)) + 1
    n_pairs = n_ic * (n_snaps - 1)
    X_all = np.zeros((n_pairs, ny_low, nx_low), dtype=np.float64)
    Y_all = np.zeros_like(X_all)
    F_all = np.zeros((n_pairs, 2, ny_low, nx_low), dtype=np.float64)

    idx = 0
    for s in range(n_ic):
        if show_progress:
            print(
                f"[Burgers2D dataset][{worker_name}] trajectory {s+1}/{n_ic}: start",
                flush=True,
            )
        u0 = random_fourier_ic2d(
            xx_fine, yy_fine,
            kmax=kmax,
            amp_range=(amp_min, amp_max),
            decay=decay,
            rng=rng,
        )
        state_fine = np.asarray(u0, dtype=np.float64).copy()
        u_in = downsample_cell_average2d(state_fine, upsample, upsample)

        for j in range(n_snaps - 1):
            state_fine = solver.advance(
                state_fine,
                dx=dx_fine,
                dy=dy_fine,
                T=dt_snap,
                cfl=cfl,
                return_all=False,
            )
            if show_progress:
                print(
                    f"[Burgers2D dataset][{worker_name}] trajectory {s+1}/{n_ic}, "
                    f"time snap {j+1}/{n_snaps-1} finished at t={(j+1)*dt_snap:.6f}",
                    flush=True,
                )

            u_out = downsample_cell_average2d(state_fine, upsample, upsample)
            flux = recover_zero_mean_flux2d(
                u_in, u_out, dt=dt_snap, dx=dx_low, dy=dy_low, cache=flux_cache
            )
            X_all[idx] = u_in
            Y_all[idx] = u_out
            F_all[idx] = flux
            idx += 1
            u_in = u_out
        if show_progress:
            print(
                f"[Burgers2D dataset][{worker_name}] trajectory {s+1}/{n_ic}: finished",
                flush=True,
            )

    return X_all, Y_all, F_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--n_val", type=int, default=20)
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--dt_snap", type=float, default=5e-2)
    ap.add_argument("--nx_low", type=int, default=128)
    ap.add_argument("--ny_low", type=int, default=128)
    ap.add_argument("--upsample", type=int, default=4)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--decay", type=float, default=1.5)
    ap.add_argument("--amp_min", type=float, default=0.4)
    ap.add_argument("--amp_max", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--num_workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    ap.add_argument(
        "--use_numba",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
    )
    args = ap.parse_args()

    split_specs = [
        ("train", args.n_train, args.seed),
        ("val", args.n_val, args.seed + 1000),
        ("test", args.n_test, args.seed + 2000),
    ]

    if args.use_numba not in ("auto", "on", "off"):
        raise ValueError('use_numba must be one of: "auto", "on", "off".')

    requested_num_workers = max(1, int(args.num_workers))
    effective_num_workers = requested_num_workers
    cpu_count = os.cpu_count() or 1
    if args.use_numba == "on":
        use_numba_main = True
        use_numba_workers = requested_num_workers > 1
    elif args.use_numba == "off":
        use_numba_main = False
        use_numba_workers = False
    else:
        use_numba_main = True
        use_numba_workers = False

    worker_numba_threads = 1
    if requested_num_workers > 1 and args.use_numba == "auto":
        print(
            "[Burgers2D dataset] auto mode: disabling numba inside worker processes "
            "to avoid nested process/JIT parallelism",
            flush=True,
        )
    elif requested_num_workers > 1 and args.use_numba == "on":
        worker_numba_threads = max(1, cpu_count // requested_num_workers)
        print(
            "[Burgers2D dataset] hybrid mode: enabling numba inside worker processes "
            f"with {worker_numba_threads} thread(s) per worker",
            flush=True,
        )

    total_ic = args.n_train + args.n_val + args.n_test
    os.makedirs(args.out_dir, exist_ok=True)
    print(
        "[Burgers2D dataset] global plan "
        f"total_ic={total_ic}, requested_workers={requested_num_workers}, "
        f"effective_workers={effective_num_workers}, use_numba={args.use_numba}, "
        f"main_numba={use_numba_main}, worker_numba={use_numba_workers}, "
        f"worker_numba_threads={worker_numba_threads}",
        flush=True,
    )

    all_jobs = []
    split_outputs = {"train": [], "val": [], "test": []}
    split_counts = {name: n_ic for name, n_ic, _ in split_specs}
    split_done = {name: 0 for name in split_outputs}
    if effective_num_workers > 1:
        for split_name, n_ic, seed_base in split_specs:
            print(
                f"[Burgers2D dataset] schedule split={split_name}, n_ic={n_ic}",
                flush=True,
            )
            for i in range(n_ic):
                worker_name = f"{split_name}-traj-{i+1}/{n_ic}"
                all_jobs.append(
                    dict(
                        split_name=split_name,
                        n_ic=1,
                        seed=seed_base + i,
                        nx_low=args.nx_low,
                        ny_low=args.ny_low,
                        upsample=args.upsample,
                        T=args.T,
                        dt_snap=args.dt_snap,
                        cfl=args.cfl,
                        kmax=args.kmax,
                        decay=args.decay,
                        amp_min=args.amp_min,
                        amp_max=args.amp_max,
                        lx=1.0,
                        ly=1.0,
                        show_progress=True,
                        worker_name=worker_name,
                        use_numba=use_numba_workers,
                    )
                )

    def save_split_now(split_name):
        data = _finalize_split_data(
            split_name,
            split_outputs[split_name],
            args.nx_low,
            args.ny_low,
            args.upsample,
            args.T,
            args.dt_snap,
            args.cfl,
            1.0,
            1.0,
            requested_num_workers,
            effective_num_workers,
            args.use_numba,
            use_numba_main,
            use_numba_workers,
        )
        out_path = os.path.join(args.out_dir, f"{split_name}_pv_noavg.pt")
        torch.save(data, out_path)
        print(f"Saved: {split_name}_pv_noavg.pt")
        print(
            f"{split_name.capitalize()} shapes: "
            f"{data['input'].shape}, {data['output'].shape}, {data['flux'].shape}"
        )
        split_outputs[split_name].clear()
        del data

    if effective_num_workers > 1 and all_jobs:
        for split_name, n_ic in split_counts.items():
            if n_ic == 0:
                save_split_now(split_name)

        def handle_result(split_name, out):
            split_outputs[split_name].append(out)
            split_done[split_name] += 1
            if split_done[split_name] == split_counts[split_name]:
                save_split_now(split_name)

        _collect_parallel_jobs(
            all_jobs,
            max_workers=min(effective_num_workers, len(all_jobs)),
            use_numba_workers=use_numba_workers,
            worker_numba_threads=worker_numba_threads,
            warmup_kwargs=(
                dict(
                    nx_fine=args.nx_low * args.upsample,
                    ny_fine=args.ny_low * args.upsample,
                    dt_snap=args.dt_snap,
                    cfl=args.cfl,
                    lx=1.0,
                    ly=1.0,
                )
                if use_numba_workers
                else None
            ),
            on_result=handle_result,
        )
    else:
        train_data = build_split(
            args.n_train,
            seed=args.seed,
            nx_low=args.nx_low,
            ny_low=args.ny_low,
            upsample=args.upsample,
            T=args.T,
            dt_snap=args.dt_snap,
            cfl=args.cfl,
            kmax=args.kmax,
            decay=args.decay,
            amp_min=args.amp_min,
            amp_max=args.amp_max,
            num_workers=args.num_workers,
            use_numba=args.use_numba,
        )
        torch.save(train_data, os.path.join(args.out_dir, "train_pv_noavg.pt"))
        print("Saved: train_pv_noavg.pt")
        print(f"Train shapes: {train_data['input'].shape}, {train_data['output'].shape}, {train_data['flux'].shape}")

        val_data = build_split(
            args.n_val,
            seed=args.seed + 1000,
            nx_low=args.nx_low,
            ny_low=args.ny_low,
            upsample=args.upsample,
            T=args.T,
            dt_snap=args.dt_snap,
            cfl=args.cfl,
            kmax=args.kmax,
            decay=args.decay,
            amp_min=args.amp_min,
            amp_max=args.amp_max,
            num_workers=args.num_workers,
            use_numba=args.use_numba,
        )
        torch.save(val_data, os.path.join(args.out_dir, "val_pv_noavg.pt"))
        print("Saved: val_pv_noavg.pt")
        print(f"Val shapes: {val_data['input'].shape}, {val_data['output'].shape}, {val_data['flux'].shape}")

        test_data = build_split(
            args.n_test,
            seed=args.seed + 2000,
            nx_low=args.nx_low,
            ny_low=args.ny_low,
            upsample=args.upsample,
            T=args.T,
            dt_snap=args.dt_snap,
            cfl=args.cfl,
            kmax=args.kmax,
            decay=args.decay,
            amp_min=args.amp_min,
            amp_max=args.amp_max,
            num_workers=args.num_workers,
            use_numba=args.use_numba,
        )
        torch.save(test_data, os.path.join(args.out_dir, "test_pv_noavg.pt"))
        print("Saved: test_pv_noavg.pt")
        print(f"Test shapes: {test_data['input'].shape}, {test_data['output'].shape}, {test_data['flux'].shape}")
    # print("Saved: train_pv_noavg.pt, val_pv_noavg.pt, test_pv_noavg.pt")
    # print("Train shapes:", train_data["input"].shape, train_data["output"].shape, train_data["flux"].shape)


if __name__ == "__main__":
    main()
