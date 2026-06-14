"""
Time WENO, hybrid, and FNO rollouts for SWE without plotting.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataset.data import WENO_CFL, make_periodic_cell_centers, sample_mixture_ic  # noqa: E402
from eval import DEFAULT_FNO_CKPT, load_optional_fno, rollout_hybrid_conserved, solve_swe_weno_ref  # noqa: E402
from conslaw.checkpoints import compile_if_requested, load_hybrid_step_map_1d  # noqa: E402
from conslaw.models import count_params  # noqa: E402

torch.set_default_dtype(torch.float64)


def resolve_path(path: str) -> str:
    if not path or os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(_SCRIPT_DIR, path)
    return candidate if os.path.exists(candidate) else path


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed_call(device: torch.device, func, *args, **kwargs):
    synchronize(device)
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    synchronize(device)
    return result, time.perf_counter() - t0


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], n_steps: int) -> list[dict[str, object]]:
    weno_method = next(str(r["method"]) for r in rows if str(r["method"]).startswith("weno"))
    weno_mean = np.mean([float(r["elapsed_s"]) for r in rows if r["method"] == weno_method])
    summary = []
    for method in (weno_method, "hybrid", "fno"):
        vals = np.asarray([float(r["elapsed_s"]) for r in rows if r["method"] == method], dtype=np.float64)
        if vals.size == 0:
            continue
        mean_s = float(vals.mean())
        summary.append(
            {
                "method": method,
                "n_samples": int(vals.size),
                "n_steps": int(n_steps),
                "mean_s": mean_s,
                "std_s": float(vals.std(ddof=0)),
                "min_s": float(vals.min()),
                "max_s": float(vals.max()),
                "seconds_per_step": mean_s / max(n_steps, 1),
                "speedup_vs_weno": weno_mean / mean_s if mean_s > 0 else np.nan,
            }
        )
    return summary


def save_solution_plot(
    outdir: str,
    x: np.ndarray,
    times: np.ndarray,
    ref_traj: np.ndarray,
    pred_trajs: dict[str, np.ndarray],
    *,
    seed: int,
    plot_format: str,
) -> list[str]:
    if plot_format == "none":
        return []

    labels = [(0, "h"), (1, "hu")]
    fig, axes = plt.subplots(len(labels), 1, figsize=(6.4, 5.6), sharex=True)
    axes = np.atleast_1d(axes)
    colors = {"hybrid": "#2ca02c", "fno": "#ff9900"}
    names = {"hybrid": "Hybrid", "fno": "FNO"}
    for ax, (component, ylabel) in zip(axes, labels):
        ax.plot(x, ref_traj[-1, component], label="WENO", linewidth=2.0, color="#1f77b4")
        for name, traj in pred_trajs.items():
            ax.plot(x, traj[-1, component], label=names.get(name, name), linewidth=1.8, color=colors.get(name))
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("x")
    fig.tight_layout()

    formats = ["pdf", "png"] if plot_format == "both" else [plot_format]
    paths = []
    for fmt in formats:
        path = os.path.join(outdir, f"final_solution_seed{seed}.{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=200 if fmt == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/swe_hybrid_flowmap_dt.pt")
    ap.add_argument("--ckpt_fno", type=str, default=DEFAULT_FNO_CKPT)
    ap.add_argument("--nx_low", type=int, default=256)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--dt_snap", type=float, default=None)
    ap.add_argument("--cfl", type=float, default=WENO_CFL)
    ap.add_argument("--x_left", type=float, default=0.0)
    ap.add_argument("--x_right", type=float, default=1.0)
    ap.add_argument("--bc", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--outdir", type=str, default="eval_swe_time_out")
    ap.add_argument("--no_plot_solution", action="store_true")
    ap.add_argument(
        "--solution_plot_format",
        "-solution_plot_format",
        type=str,
        default="pdf",
        choices=("pdf", "png", "both", "none"),
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    model, ckpt = load_hybrid_step_map_1d(resolve_path(args.ckpt), device=device)
    model = compile_if_requested(model, device, no_compile=args.no_compile)
    dt_model = float(ckpt["dt"]) if str(ckpt.get("integrator", "")) == "u_plus_dt_rhs" else None
    model_fno, dt_fno = load_optional_fno(resolve_path(args.ckpt_fno), device, args.no_compile, resolve_path(DEFAULT_FNO_CKPT))

    data_dir = resolve_path(args.data_dir)
    train_pt = os.path.join(data_dir, "train_pv.pt")
    meta = torch.load(train_pt, map_location="cpu")["meta"] if os.path.isfile(train_pt) else {}
    dt_snap = float(args.dt_snap if args.dt_snap is not None else ckpt.get("dt", meta.get("dt")))
    bc_eval = args.bc or ckpt.get("solver_bc") or ("outflow" if ckpt.get("bc") == "outflow" else "periodic")
    times = np.arange(0.0, args.T + 1e-12, dt_snap)
    n_steps = len(times) - 1
    weno_method = f"weno{args.nx_low}"
    x, dx = make_periodic_cell_centers(args.x_left, args.x_right, args.nx_low)
    pin_mem = device.type == "cuda"

    if args.warmup > 0 and n_steps > 0:
        u_zero = np.zeros((2, args.nx_low), dtype=np.float64)
        for _ in range(args.warmup):
            rollout_hybrid_conserved(model, u_zero, n_steps, device, pin_mem, dt_model=dt_model)
            if model_fno is not None:
                rollout_hybrid_conserved(model_fno, u_zero, n_steps, device, pin_mem, dt_model=dt_fno)
        synchronize(device)

    print(f"[params] hybrid={count_params(model):,}, fno={count_params(model_fno) if model_fno is not None else 'unavailable'}")
    print(f"[time SWE] dt={dt_snap}, nx={args.nx_low}, cfl={args.cfl}, bc={bc_eval}, n_steps={n_steps}")

    rng = np.random.default_rng(args.seed)
    rows = []
    solution_pack = None
    for sample in range(args.n_samples):
        u0, ic_type = sample_mixture_ic(x, rng=rng)
        weno_traj, weno_s = timed_call(device, solve_swe_weno_ref, u0, times, dx, bc=bc_eval, cfl=args.cfl)
        hybrid_traj, hybrid_s = timed_call(device, rollout_hybrid_conserved, model, u0, n_steps, device, pin_mem, dt_model=dt_model)
        rows.extend(
            [
                {"sample": sample, "method": weno_method, "elapsed_s": weno_s, "n_steps": n_steps, "seconds_per_step": weno_s / max(n_steps, 1), "ic_type": ic_type},
                {"sample": sample, "method": "hybrid", "elapsed_s": hybrid_s, "n_steps": n_steps, "seconds_per_step": hybrid_s / max(n_steps, 1), "ic_type": ic_type},
            ]
        )
        pred_trajs = {"hybrid": hybrid_traj}
        if model_fno is not None:
            fno_traj, fno_s = timed_call(device, rollout_hybrid_conserved, model_fno, u0, n_steps, device, pin_mem, dt_model=dt_fno)
            pred_trajs["fno"] = fno_traj
            rows.append({"sample": sample, "method": "fno", "elapsed_s": fno_s, "n_steps": n_steps, "seconds_per_step": fno_s / max(n_steps, 1), "ic_type": ic_type})
        if solution_pack is None:
            solution_pack = (weno_traj.copy(), {name: traj.copy() for name, traj in pred_trajs.items()})
        print(f"[{sample + 1}/{args.n_samples}] WENO={weno_s:.6g}s hybrid={hybrid_s:.6g}s")

    for row in rows:
        weno_s = next(float(r["elapsed_s"]) for r in rows if r["sample"] == row["sample"] and r["method"] == weno_method)
        row["speedup_vs_weno"] = weno_s / float(row["elapsed_s"]) if float(row["elapsed_s"]) > 0 else np.nan

    summary = summarize(rows, n_steps)
    write_csv(os.path.join(args.outdir, "runtime_rows.csv"), rows)
    write_csv(os.path.join(args.outdir, "runtime_summary.csv"), summary)
    if solution_pack is not None and not args.no_plot_solution:
        ref_traj, pred_trajs = solution_pack
        for path in save_solution_plot(
            args.outdir,
            x,
            times,
            ref_traj,
            pred_trajs,
            seed=args.seed,
            plot_format=args.solution_plot_format,
        ):
            print(f"[plot] saved {path}")
    for row in summary:
        print(f"[summary] {row['method']}: mean={row['mean_s']:.6g}s, speedup_vs_weno={row['speedup_vs_weno']:.3g}")
    print(f"Saved timing CSVs under {args.outdir}")


if __name__ == "__main__":
    main()
