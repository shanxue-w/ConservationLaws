"""
Burgers dataset: point-value initial condition on fine grid,
FD-WENOZ (CuPy) evolution, cell-average downsample to coarse.

Run from Burgers/dataset: python data_pv_ic.py
Saves train_pv.pt, val_pv.pt, test_pv.pt (distinct from FV-IC .npz).
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from conslaw.solver import FD_WENOZ, downsample_cell_average

def recover_zero_mean_flux(u, u_next, dt, dx):
    """
    Recover zero-mean periodic flux F from
        u_next = u - (dt/dx) * (F - roll(F,1))
    Inputs:
        u, u_next: (B, N)
        dt, dx: scalar
    Returns:
        flux_tgt: (B, N), zero-mean gauge fixed
    """
    # q_i = F_i - F_{i-1}
    q = -(dx / dt) * (u_next - u)   # (B, N)

    # Enforce periodic compatibility: sum_i q_i = 0
    q = q - q.mean(dim=1, keepdim=True)

    # Fix one gauge: set F_0 = 0, then reconstruct by cumulative sum
    flux = torch.zeros_like(q)
    flux[:, 1:] = torch.cumsum(q[:, 1:], dim=1)

    # Final gauge fix: make flux zero-mean
    flux = flux - flux.mean(dim=1, keepdim=True)
    return flux

def random_fourier_ic(x, kmax=5, amp_range=(0.4, 1.0), decay=1.0, rng=None):
    """Point-value IC: u(x) = sum_k a_k cos(2πkx) + b_k sin(2πkx), rescaled."""
    if rng is None:
        rng = np.random.default_rng()
    u = np.zeros_like(x, dtype=np.float64)
    for k in range(1, kmax + 1):
        scale = 1.0 / (k ** decay)
        ak = rng.normal(0.0, scale)
        bk = rng.normal(0.0, scale)
        u += ak * np.cos(2 * np.pi * k * x) + bk * np.sin(2 * np.pi * k * x)
    umax = np.max(np.abs(u)) + 1e-12
    target_amp = rng.uniform(amp_range[0], amp_range[1])
    u = u / umax * target_amp
    u += rng.uniform(-0.5, 0.5) * target_amp
    return u.astype(np.float64)


def build_solver():
    def flux(u):
        return 0.5 * u * u
    def dflux(u):
        return u
    return FD_WENOZ(flux=flux, dflux=dflux, flux_split="local_lf", bc="periodic")


def generate_trajectory(u0_fine, solver, T, dt_snap, dx_fine, cfl=0.4):
    """Evolve u0_fine to time T, return snapshots at dt_snap (each snapshot fine grid)."""
    u = np.asarray(u0_fine, dtype=np.float64).copy()
    times = np.arange(0.0, T + 1e-12, dt_snap)
    snaps = [u.copy()]
    t = 0.0
    for i in range(1, len(times)):
        t_target = times[i]
        dt_interval = t_target - t
        u = solver.solve(u, dx=dx_fine, T=dt_interval, cfl=cfl, return_all=False)
        t = t_target
        snaps.append(u.copy())
    return np.stack(snaps, axis=0), times


def build_split(n_ic, seed, nx_low=128, upsample=8, T=0.5, dt_snap=1e-2, cfl=0.4,
                kmax=5, decay=2.0, amp_min=0.5, amp_max=1.0, L=1.0):
    rng = np.random.default_rng(seed)
    dx_low = L / nx_low
    nx_fine = nx_low * upsample
    dx_fine = L / nx_fine
    x_fine = np.linspace(0.0, L, nx_fine, endpoint=False).astype(np.float64)
    solver = build_solver()

    n_snaps = int(round(T / dt_snap)) + 1
    X_all = np.zeros((n_ic * (n_snaps - 1), nx_low), dtype=np.float64)
    Y_all = np.zeros((n_ic * (n_snaps - 1), nx_low), dtype=np.float64)
    F_all = np.zeros((n_ic * (n_snaps - 1), nx_low), dtype=np.float64)

    idx = 0
    for s in range(n_ic):
        print(f"Generating trajectory {s+1}/{n_ic}...", end="\r")
        u0 = random_fourier_ic(
            x_fine, kmax=kmax, amp_range=(amp_min, amp_max), decay=decay, rng=rng
        )
        snaps, _ = generate_trajectory(u0, solver, T=T, dt_snap=dt_snap, dx_fine=dx_fine, cfl=cfl)
        for j in range(n_snaps - 1):
            u_coarse_in = downsample_cell_average(snaps[j], upsample)
            u_coarse_out = downsample_cell_average(snaps[j + 1], upsample)
            flux = recover_zero_mean_flux(
                torch.tensor(u_coarse_in)[None],
                torch.tensor(u_coarse_out)[None],
                dt_snap,
                dx_low,
            )[0].numpy()
            X_all[idx] = u_coarse_in
            Y_all[idx] = u_coarse_out
            F_all[idx] = flux
            idx += 1

    print()
    dx = L / nx_low
    data = {
        "input": torch.tensor(X_all, dtype=torch.float64),
        "output": torch.tensor(Y_all, dtype=torch.float64),
        "flux": torch.tensor(F_all, dtype=torch.float64),
        "meta": {
            "equation": "u_t + (u^2/2)_x = 0",
            "boundary": "periodic",
            "domain": [0.0, L],
            "nx": nx_low,
            "dx": dx,
            "dt": dt_snap,
            "T": T,
            "n_snaps": n_snaps,
            "upsample": upsample,
            "nx_fine": nx_fine,
            "reference": "point-value IC + FD-WENOZ fine grid + cell-average downsample",
            "cfl": cfl,
        },
    }
    return data



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_val", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--dt_snap", type=float, default=5e-2)
    ap.add_argument("--nx_low", type=int, default=256)
    ap.add_argument("--upsample", type=int, default=4)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--decay", type=float, default=2.0)
    ap.add_argument("--amp_min", type=float, default=0.5)
    ap.add_argument("--amp_max", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    train_data = build_split(
        args.n_train, seed=args.seed, nx_low=args.nx_low, upsample=args.upsample,
        T=args.T, dt_snap=args.dt_snap, cfl=args.cfl,
        kmax=args.kmax, decay=args.decay, amp_min=args.amp_min, amp_max=args.amp_max,
    )
    val_data = build_split(
        args.n_val, seed=args.seed + 1000, nx_low=args.nx_low, upsample=args.upsample,
        T=args.T, dt_snap=args.dt_snap, cfl=args.cfl,
        kmax=args.kmax, decay=args.decay, amp_min=args.amp_min, amp_max=args.amp_max,
    )
    test_data = build_split(
        args.n_test, seed=args.seed + 2000, nx_low=args.nx_low, upsample=args.upsample,
        T=args.T, dt_snap=args.dt_snap, cfl=args.cfl,
        kmax=args.kmax, decay=args.decay, amp_min=args.amp_min, amp_max=args.amp_max,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(train_data, os.path.join(args.out_dir, "train_pv.pt"))
    torch.save(val_data, os.path.join(args.out_dir, "val_pv.pt"))
    torch.save(test_data, os.path.join(args.out_dir, "test_pv.pt"))

    print("Saved: train_pv.pt, val_pv.pt, test_pv.pt")
    print("Shapes: train", train_data["input"].shape, train_data["output"].shape)


if __name__ == "__main__":
    main()
