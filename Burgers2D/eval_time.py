"""
Time WENO, hybrid, and FNO rollouts for Burgers2D without plotting.
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

from dataset.data import make_periodic_grid, random_fourier_ic2d  # noqa: E402
from eval import (  # noqa: E402
    DEFAULT_FNO_CKPT,
    DEFAULT_HYBRID_CKPT,
    load_model_from_ckpt,
    load_optional_model,
    model_label_from_ckpt,
    resolve_rollout_dt,
    rollout_step_model_2d,
    solve_burgers2d_weno_ref,
)
from conslaw.models import count_params  # noqa: E402

torch.set_default_dtype(torch.float64)
DATASET_WENO_CFL = 0.4


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
    ref_traj: np.ndarray,
    pred_trajs: dict[str, np.ndarray],
    *,
    seed: int,
    plot_format: str,
) -> list[str]:
    if plot_format == "none":
        return []

    fields = [("WENO", ref_traj[-1])]
    names = {"hybrid": "Hybrid", "fno": "FNO"}
    for name, traj in pred_trajs.items():
        fields.append((names.get(name, name), traj[-1]))
    vmin = min(float(np.min(field)) for _, field in fields)
    vmax = max(float(np.max(field)) for _, field in fields)

    fig, axes = plt.subplots(1, len(fields), figsize=(4.0 * len(fields), 3.6), squeeze=False)
    last_im = None
    for ax, (title, field) in zip(axes[0], fields):
        last_im = ax.imshow(field, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    if last_im is not None:
        fig.colorbar(last_im, ax=axes[0].tolist(), fraction=0.046, pad=0.04)

    formats = ["pdf", "png"] if plot_format == "both" else [plot_format]
    paths = []
    for fmt in formats:
        path = os.path.join(outdir, f"final_solution_seed{seed}.{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=200 if fmt == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_HYBRID_CKPT)
    ap.add_argument("--ckpt_fno", type=str, default=DEFAULT_FNO_CKPT)
    ap.add_argument("--nx_low", type=int, default=128)
    ap.add_argument("--ny_low", type=int, default=128)
    ap.add_argument("--T", type=float, default=0.3)
    ap.add_argument("--dt_snap", type=float, default=None)
    ap.add_argument("--cfl", type=float, default=DATASET_WENO_CFL)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--decay", type=float, default=1.5)
    ap.add_argument("--amp_min", type=float, default=0.4)
    ap.add_argument("--amp_max", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=str, default="eval_burgers2d_time_out")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--use_numba", action="store_true")
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--no_plot_solution", action="store_true")
    ap.add_argument(
        "--solution_plot_format",
        "-solution_plot_format",
        type=str,
        default="pdf",
        choices=("pdf", "png", "both", "none"),
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    pin_mem = device.type == "cuda"

    primary_model, primary_ckpt, primary_path = load_model_from_ckpt(args.ckpt, device, no_compile=args.no_compile)
    primary_name = model_label_from_ckpt(primary_ckpt)
    models: dict[str, tuple[torch.nn.Module, dict]] = {primary_name: (primary_model, primary_ckpt)}
    model_fno, ckpt_fno, path_fno = load_optional_model(
        args.ckpt_fno,
        device,
        no_compile=args.no_compile,
        default_path=DEFAULT_FNO_CKPT,
    )
    if model_fno is not None and ckpt_fno is not None:
        models[model_label_from_ckpt(ckpt_fno)] = (model_fno, ckpt_fno)

    dt_snap = resolve_rollout_dt(args, primary_ckpt)
    times = np.arange(0.0, args.T + 1e-12, dt_snap)
    n_steps = len(times) - 1
    weno_method = f"weno{args.nx_low}" if args.nx_low == args.ny_low else f"weno{args.nx_low}x{args.ny_low}"
    xx, yy, dx, dy = make_periodic_grid(args.nx_low, args.ny_low)

    if args.warmup > 0 and n_steps > 0:
        u_zero = np.zeros((args.ny_low, args.nx_low), dtype=np.float64)
        for _ in range(args.warmup):
            for name, (model, _) in models.items():
                rollout_step_model_2d(model, u_zero, n_steps, dt_snap, device, pin_memory=pin_mem, label=name)
        synchronize(device)

    print(f"[params] hybrid={count_params(primary_model):,}, fno={count_params(model_fno) if model_fno is not None else 'unavailable'}")
    print(f"[models] hybrid={primary_path}, fno={path_fno or 'unavailable'}")
    print(f"[time Burgers2D] dt={dt_snap}, nx={args.nx_low}, ny={args.ny_low}, cfl={args.cfl}, n_steps={n_steps}")

    rng = np.random.default_rng(args.seed)
    rows = []
    solution_pack = None
    for sample in range(args.n_samples):
        u0 = random_fourier_ic2d(
            xx,
            yy,
            kmax=args.kmax,
            amp_range=(args.amp_min, args.amp_max),
            decay=args.decay,
            rng=rng,
        )
        weno_traj, weno_s = timed_call(
            device,
            solve_burgers2d_weno_ref,
            u0,
            times,
            dx,
            dy,
            cfl=args.cfl,
            use_numba=args.use_numba,
            verbose=args.verbose,
        )
        rows.append({"sample": sample, "method": weno_method, "elapsed_s": weno_s, "n_steps": n_steps, "seconds_per_step": weno_s / max(n_steps, 1)})
        pred_trajs = {}
        for name, (model, _) in models.items():
            pred_traj, elapsed = timed_call(
                device,
                rollout_step_model_2d,
                model,
                u0,
                n_steps,
                dt_snap,
                device,
                pin_memory=pin_mem,
                label=name,
            )
            pred_trajs[name] = pred_traj
            rows.append({"sample": sample, "method": name, "elapsed_s": elapsed, "n_steps": n_steps, "seconds_per_step": elapsed / max(n_steps, 1)})
        if solution_pack is None:
            solution_pack = (weno_traj.copy(), {name: traj.copy() for name, traj in pred_trajs.items()})
        print(f"[{sample + 1}/{args.n_samples}] WENO={weno_s:.6g}s")

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
