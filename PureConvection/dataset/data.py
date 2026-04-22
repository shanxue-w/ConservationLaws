"""
PureConvection dataset: point-value initial condition on fine grid,
FD-WENOZ evolution, cell-average downsample to coarse.
Includes numerical flux (zero-mean gauge) for flux-trained models.

Run from PureConvection/dataset: python data_pv_ic.py
Saves train_pv.pt, val_pv.pt, test_pv.pt (keys: input, output, flux, meta).
"""

import os
import sys
import argparse
import numpy as np
import torch

# clop from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from conslaw.solver import FD_WENOZ, downsample_cell_average


def recover_zero_mean_flux(u, u_next, dt, dx):
    """
    Recover zero-mean periodic flux F from
        u_next = u - (dt/dx) * (F - roll(F,1))
    Inputs:
        u, u_next: (B, N) torch tensors
        dt, dx: scalar
    Returns:
        flux_tgt: (B, N), zero-mean gauge fixed
    """
    q = -(dx / dt) * (u_next - u)
    q = q - q.mean(dim=1, keepdim=True)
    flux = torch.zeros_like(q)
    flux[:, 1:] = torch.cumsum(q[:, 1:], dim=1)
    flux = flux - flux.mean(dim=1, keepdim=True)
    return flux


L = 1.0
NX = 256
UPSAMPLE = 4
DT = 5e-2
N_STEPS_PER_IC = 10
C = 1.0
WENO_CFL = 0.4


def wrap_periodic(x, L=1.0):
    return np.mod(x, L)

def sample_periodic_breaks_with_min_gap(rng, n_breaks, L=1.0, min_gap=0.12, max_tries=2000):
    if n_breaks <= 0:
        return np.array([], dtype=np.float64)
    if n_breaks * min_gap >= L:
        raise ValueError(f"min_gap too large for {n_breaks} breaks.")
    for _ in range(max_tries):
        pts = np.sort(rng.uniform(0.0, L, size=n_breaks))
        gaps = np.diff(np.concatenate([pts, [pts[0] + L]]))
        if np.min(gaps) >= min_gap:
            return pts.astype(np.float64)
    raise RuntimeError(f"Failed to sample breaks with min_gap={min_gap}")

def random_piecewise_constant_ic(rng, L=1.0, min_gap=0.15):
    """Simple piecewise constant: 1--3 segments, values in [-0.8, 0.8]."""
    n_breaks = int(rng.integers(1, 3))  # 1 or 2 breaks -> 2 or 3 segments
    breaks = sample_periodic_breaks_with_min_gap(rng, n_breaks, L=L, min_gap=min_gap)
    points = np.concatenate(([0.0], breaks, [L]))
    n_seg = len(points) - 1
    vals = rng.uniform(-0.8, 0.8, size=n_seg)

    def u0(x):
        x = wrap_periodic(np.asarray(x), L)
        out = np.empty_like(x, dtype=np.float64)
        for k in range(n_seg):
            a, b = points[k], points[k + 1]
            mask = (x >= a) & (x < b) if k < n_seg - 1 else (x >= a) & (x <= b)
            out[mask] = vals[k]
        return out

    return u0, "piecewise_constant"


def random_piecewise_smooth_jump_ic(rng, L=1.0, min_gap=0.15):
    """Piecewise smooth with jumps: 1--2 segments, constant + low-k Fourier only (no linear term)."""
    n_breaks = int(rng.integers(1, 2))  # 1 break -> 2 segments
    breaks = sample_periodic_breaks_with_min_gap(rng, n_breaks, L=L, min_gap=min_gap)
    points = np.concatenate(([0.0], breaks, [L]))
    n_seg = len(points) - 1

    seg_params = []
    for k in range(n_seg):
        a0 = rng.uniform(-0.6, 0.6)
        amp1 = rng.uniform(0.0, 0.12)
        amp2 = rng.uniform(0.0, 0.08)
        k1 = 1
        k2 = int(rng.integers(1, 2))  # 1 or 2
        phi1 = rng.uniform(0.0, 2 * np.pi)
        phi2 = rng.uniform(0.0, 2 * np.pi)
        seg_params.append((a0, amp1, amp2, k1, k2, phi1, phi2))

    def u0(x):
        x = wrap_periodic(np.asarray(x), L)
        out = np.empty_like(x, dtype=np.float64)
        for k in range(n_seg):
            a, b = points[k], points[k + 1]
            a0, amp1, amp2, k1, k2, phi1, phi2 = seg_params[k]
            if k < n_seg - 1:
                mask = (x >= a) & (x < b)
            else:
                mask = (x >= a) & (x <= b)
            xm = x[mask]
            xi = (xm - a) / max(b - a, 1e-14)
            out[mask] = (
                a0
                + amp1 * np.sin(2 * np.pi * k1 * xi + phi1)
                + amp2 * np.cos(2 * np.pi * k2 * xi + phi2)
            )
        return out

    return u0, "piecewise_smooth_jump"


def random_smooth_periodic_ic(rng, L=1.0):
    """Smooth periodic: only 2--3 Fourier modes, decay=2 so low-k dominant."""
    kmax = int(rng.integers(2, 4))  # 2 or 3 modes only
    decay = 2.0
    a = rng.normal(size=kmax) / (np.arange(1, kmax + 1, dtype=np.float64) ** decay)
    b = rng.normal(size=kmax) / (np.arange(1, kmax + 1, dtype=np.float64) ** decay)
    x_probe = np.linspace(0.0, L, 4096, endpoint=False)
    u_probe = np.zeros_like(x_probe)
    for k in range(1, kmax + 1):
        u_probe += a[k - 1] * np.cos(2 * np.pi * k * x_probe)
        u_probe += b[k - 1] * np.sin(2 * np.pi * k * x_probe)
    m = np.max(np.abs(u_probe))
    random_scale = rng.uniform(0.5, 1.0)
    scale = random_scale / m if m > 1e-14 else 1.0
    a, b = scale * a, scale * b

    def u0(x):
        x = wrap_periodic(np.asarray(x), L)
        out = np.zeros_like(x, dtype=np.float64)
        for k in range(1, kmax + 1):
            out += a[k - 1] * np.cos(2 * np.pi * k * x)
            out += b[k - 1] * np.sin(2 * np.pi * k * x)
        return out

    return u0, "smooth_periodic"


def sample_random_ic(rng):
    """Sample IC: bias toward smooth (easier to learn)."""
    r = rng.uniform()
    if r < 0.6:
        return random_piecewise_constant_ic(rng)
    # if r < 0.55:
    #     return random_piecewise_smooth_jump_ic(rng)
    return random_smooth_periodic_ic(rng)


def build_solver(c=1.0):
    def flux(u):
        return np.asarray(c, dtype=u.dtype) * u
    def dflux(u):
        return np.full_like(u, c)
    return FD_WENOZ(flux=flux, dflux=dflux, flux_split="local_lf", bc="periodic")


def generate_pairs_one_ic(u0_func, solver, nx, upsample, dt, n_steps_per_ic, c, L, weno_cfl):
    nx_fine = nx * upsample
    dx_fine = L / nx_fine
    dx_low = L / nx
    x_fine = np.linspace(0.0, L, nx_fine, endpoint=False).astype(np.float64)
    u_fine = u0_func(x_fine)  # point values
    u_fine = np.asarray(u_fine, dtype=np.float64)

    states_coarse = [downsample_cell_average(u_fine, upsample)]

    for _ in range(n_steps_per_ic):
        u_fine = solver.solve(u_fine, dx=dx_fine, T=dt, cfl=weno_cfl, return_all=False)
        states_coarse.append(downsample_cell_average(u_fine, upsample))

    xs = np.stack([states_coarse[n] for n in range(n_steps_per_ic)], axis=0)
    ys = np.stack([states_coarse[n + 1] for n in range(n_steps_per_ic)], axis=0)

    # Recover numerical flux for each (u_in, u_out) pair (same as Burgers)
    fluxes = []
    for n in range(n_steps_per_ic):
        u_in = torch.tensor(states_coarse[n][None], dtype=torch.float64)
        u_out = torch.tensor(states_coarse[n + 1][None], dtype=torch.float64)
        f = recover_zero_mean_flux(u_in, u_out, dt, dx_low)[0].numpy()
        fluxes.append(f)
    fs = np.stack(fluxes, axis=0)

    return xs, ys, fs


def build_split(n_ic, seed, nx=256, upsample=4, dt=5e-2, n_steps_per_ic=10, c=1.0, L=1.0, weno_cfl=0.4):
    rng = np.random.default_rng(seed)
    solver = build_solver(c=c)

    X_all, Y_all, F_all, ic_types = [], [], [], []

    for i in range(n_ic):
        print(f"Generating IC {i+1}/{n_ic}...", end="\r")
        u0_func, ic_type = sample_random_ic(rng)
        xs, ys, fs = generate_pairs_one_ic(
            u0_func, solver, nx=nx, upsample=upsample, dt=dt,
            n_steps_per_ic=n_steps_per_ic, c=c, L=L, weno_cfl=weno_cfl,
        )
        X_all.append(xs)
        Y_all.append(ys)
        F_all.append(fs)
        ic_types.extend([ic_type] * len(xs))

    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    F_all = np.concatenate(F_all, axis=0)
    dx = L / nx
    nx_fine = nx * upsample

    data = {
        "input": torch.tensor(X_all, dtype=torch.float64),
        "output": torch.tensor(Y_all, dtype=torch.float64),
        "flux": torch.tensor(F_all, dtype=torch.float64),
        "meta": {
            "equation": "u_t + c u_x = 0",
            "boundary": "periodic",
            "domain": [0.0, L],
            "nx": nx,
            "dx": dx,
            "dt": dt,
            "c": c,
            "n_steps_per_ic": n_steps_per_ic,
            "upsample": upsample,
            "nx_fine": nx_fine,
            "reference": "point-value IC + FD-WENOZ fine grid + cell-average downsample",
            "weno_cfl": weno_cfl,
            "ic_types": ic_types,
        },
    }
    print()
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--nx", type=int, default=NX)
    ap.add_argument("--upsample", type=int, default=UPSAMPLE)
    ap.add_argument("--dt_snap", type=float, default=DT, help="time step (snap interval)")
    ap.add_argument("--n_steps_per_ic", type=int, default=N_STEPS_PER_IC)
    ap.add_argument("--c", type=float, default=C)
    ap.add_argument("--weno_cfl", type=float, default=WENO_CFL)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    train_data = build_split(
        args.n_train, seed=args.seed, nx=args.nx, upsample=args.upsample, dt=args.dt_snap,
        n_steps_per_ic=args.n_steps_per_ic, c=args.c, L=L, weno_cfl=args.weno_cfl,
    )
    val_data = build_split(
        args.n_val, seed=args.seed + 1000, nx=args.nx, upsample=args.upsample, dt=args.dt_snap,
        n_steps_per_ic=args.n_steps_per_ic, c=args.c, L=L, weno_cfl=args.weno_cfl,
    )
    test_data = build_split(
        args.n_test, seed=args.seed + 2000, nx=args.nx, upsample=args.upsample, dt=args.dt_snap,
        n_steps_per_ic=args.n_steps_per_ic, c=args.c, L=L, weno_cfl=args.weno_cfl,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(train_data, os.path.join(args.out_dir, "train_pv.pt"))
    torch.save(val_data, os.path.join(args.out_dir, "val_pv.pt"))
    torch.save(test_data, os.path.join(args.out_dir, "test_pv.pt"))

    print("Saved: train_pv.pt, val_pv.pt, test_pv.pt")
    print("Shapes: train", train_data["input"].shape, train_data["output"].shape, train_data["flux"].shape)


if __name__ == "__main__":
    main()
