"""
Time coarse-grid WENO, Hybrid, and FNO rollouts for Euler2D.

This is a timing-only sibling script for ``eval_periodic.py`` and
``eval_pri.py``. It uses the first frame of selected test trajectories,
advances that frame with coarse-grid WENO on the model mesh, times the two
neural models, and saves final-time primitive-field plots without titles.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join("/tmp", "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from conslaw.models import count_params

from common_periodic import (
    DEFAULT_TEST_NAME as PERIODIC_TEST_NAME,
    configure_runtime as configure_periodic_runtime,
    load_trajectory_split as load_periodic_split,
    rollout_step_model as rollout_periodic_model,
)
from common_pri import (
    DEFAULT_TEST_NAME as PRI_TEST_NAME,
    conserved_to_primitive_numpy,
    load_trajectory_split as load_pri_split,
    primitive_to_conserved_numpy,
    rollout_step_model as rollout_pri_model,
)
from dataset.data import build_euler2d_solver as build_outflow_solver
from dataset.data_periodic import build_euler2d_solver as build_periodic_solver
from dataset.data_periodic import enforce_physical_euler2d
from eval_periodic import (
    DEFAULT_FNO_CKPT as PERIODIC_FNO_CKPT,
    DEFAULT_HYBRID_CKPT as PERIODIC_HYBRID_CKPT,
    load_model_from_ckpt as load_periodic_model,
    load_optional_model as load_optional_periodic_model,
    model_label_from_ckpt as periodic_model_label,
)
from eval_pri import (
    DEFAULT_FNO_CKPT as PRI_FNO_CKPT,
    DEFAULT_HYBRID_CKPT as PRI_HYBRID_CKPT,
    load_optional_model as load_optional_pri_model,
    load_required_model as load_pri_model,
    model_label_from_ckpt as pri_model_label,
    save_single_field,
)


def resolve_path(path: str) -> str:
    if not path or os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(_SCRIPT_DIR, path)
    return candidate if os.path.exists(candidate) else path


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def select_indices(n_total: int, n_samples: int, sample_seed: int | None) -> np.ndarray:
    if n_samples <= 0 or n_samples >= n_total:
        return np.arange(n_total, dtype=np.int64)
    if sample_seed is None:
        return np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(sample_seed)
    return rng.choice(n_total, size=n_samples, replace=False).astype(np.int64)


def timed_periodic_weno(op, u0: torch.Tensor, *, dx: float, dy: float, dt: float, n_steps: int, cfl: float, gamma: float):
    u_np = enforce_physical_euler2d(u0.cpu().numpy(), gamma=gamma)
    t0 = time.perf_counter()
    traj = op.solve_snapshots(
        u_np,
        dx,
        dy,
        dt_snap=dt,
        n_snaps=n_steps + 1,
        cfl=cfl,
        log_ic_state=True,
    )
    return traj.astype(np.float64, copy=False), time.perf_counter() - t0


def timed_pri_weno(op, u0_prim: torch.Tensor, *, dx: float, dy: float, dt: float, n_steps: int, cfl: float, gamma: float):
    u0_cons = primitive_to_conserved_numpy(u0_prim.cpu().numpy(), gamma=gamma)
    u0_cons = enforce_physical_euler2d(u0_cons, gamma=gamma)
    t0 = time.perf_counter()
    cons_traj = op.solve_snapshots(
        u0_cons,
        dx,
        dy,
        dt_snap=dt,
        n_snaps=n_steps + 1,
        cfl=cfl,
        log_ic_state=True,
    )
    prim_traj = np.stack([conserved_to_primitive_numpy(state, gamma=gamma) for state in cons_traj], axis=0)
    return prim_traj.astype(np.float64, copy=False), time.perf_counter() - t0


def timed_model_rollout(rollout_fn, model, u0: torch.Tensor, *, n_steps: int, dt: float, device: torch.device, pin_memory: bool):
    synchronize(device)
    t0 = time.perf_counter()
    traj = rollout_fn(
        model,
        u0,
        n_steps=n_steps,
        dt=dt,
        device=device,
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    synchronize(device)
    return traj.astype(np.float64, copy=False), time.perf_counter() - t0


def warmup_model(rollout_fn, model, u0: torch.Tensor, *, n_steps: int, dt: float, device: torch.device, pin_memory: bool) -> None:
    if n_steps <= 0:
        return
    _ = rollout_fn(
        model,
        u0,
        n_steps=n_steps,
        dt=dt,
        device=device,
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    synchronize(device)


def primitive_traj_for_case(case: str, traj: np.ndarray, gamma: float) -> np.ndarray:
    if case == "periodic":
        return np.stack([conserved_to_primitive_numpy(state, gamma=gamma) for state in traj], axis=0)
    return np.asarray(traj, dtype=np.float64)


def append_row(
    rows: list[dict[str, object]],
    *,
    sample_order: int,
    sample_idx: int,
    method: str,
    seconds: float,
    n_steps: int,
    dt: float,
    nx: int,
    ny: int,
) -> None:
    rows.append(
        {
            "sample_order": sample_order,
            "sample_idx": sample_idx,
            "method": method,
            "seconds": float(seconds),
            "n_steps": int(n_steps),
            "seconds_per_step": float(seconds) / max(int(n_steps), 1),
            "dt": float(dt),
            "nx": int(nx),
            "ny": int(ny),
        }
    )


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    methods = []
    for row in rows:
        method = str(row["method"])
        if method not in methods:
            methods.append(method)
    weno_mean = np.mean([float(row["seconds"]) for row in rows if str(row["method"]).startswith("weno")])
    for method in methods:
        values = np.asarray([float(row["seconds"]) for row in rows if row["method"] == method], dtype=np.float64)
        mean = float(values.mean())
        out.append(
            {
                "method": method,
                "n": int(values.size),
                "mean_seconds": mean,
                "std_seconds": float(values.std(ddof=0)),
                "min_seconds": float(values.min()),
                "max_seconds": float(values.max()),
                "speedup_vs_weno": weno_mean / mean if mean > 0.0 else np.nan,
            }
        )
    return out


def save_final_plots(
    outdir: str,
    case: str,
    sample_idx: int,
    ref_prim: np.ndarray,
    pred_prims: dict[str, np.ndarray],
    *,
    plot_format: str,
) -> list[str]:
    if plot_format == "none":
        return []
    fields = {"rho": 0, "u": 1, "v": 2, "p": 3}
    methods = {"weno": ref_prim[-1], **{name: traj[-1] for name, traj in pred_prims.items()}}
    formats = ["pdf", "png"] if plot_format == "both" else [plot_format]
    paths = []

    for method_name, state in methods.items():
        for field_name, channel in fields.items():
            field = np.asarray(state[channel], dtype=np.float64)
            for fmt in formats:
                path = os.path.join(outdir, f"final_{case}_{field_name}_{method_name}_sample{sample_idx:06d}.{fmt}")
                save_single_field(path, field)
                paths.append(path)
    return paths


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, default="periodic", choices=("periodic", "pri", "outflow"))
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--ckpt_fno", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--test_name", type=str, default=None)
    ap.add_argument("--test_path", type=str, default=None)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--seed", "--sample_seed", dest="sample_seed", type=int, default=0)
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--rollout_steps", type=int, default=0)
    ap.add_argument("--weno_cfl", type=float, default=None)
    ap.add_argument("--weno_reconstruction", type=str, default=None, choices=("component", "characteristic"))
    ap.add_argument("--weno_line_batch_size", type=int, default=16)
    ap.add_argument("--warmup_steps", type=int, default=1)
    ap.add_argument("--skip_warmup", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow_tf32", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--outdir", type=str, default="eval_euler2d_time_out")
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
    case = "pri" if args.case == "outflow" else args.case
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device)
    configure_periodic_runtime(device, allow_tf32=args.allow_tf32)
    pin_mem = device.type == "cuda"

    if case == "periodic":
        ckpt = args.ckpt or PERIODIC_HYBRID_CKPT
        ckpt_fno = args.ckpt_fno or PERIODIC_FNO_CKPT
        test_name = args.test_name or PERIODIC_TEST_NAME
        test_path = resolve_path(args.test_path or os.path.join(args.data_dir, test_name))
        states, meta = load_periodic_split(test_path, dtype=torch.float32)
        total = int(states.shape[0])
        selected = select_indices(total, args.n_samples, args.sample_seed)
        first_frames = states[torch.as_tensor(selected, dtype=torch.long), 0]
        del states
        primary_model, primary_ckpt, primary_path = load_periodic_model(ckpt, device, no_compile=args.no_compile)
        models = {periodic_model_label(primary_ckpt): (primary_model, primary_ckpt)}
        paths = {periodic_model_label(primary_ckpt): primary_path}
        fno_model, fno_ckpt, fno_path = load_optional_periodic_model(
            ckpt_fno,
            device,
            no_compile=args.no_compile,
            default_path=PERIODIC_FNO_CKPT,
        )
        if fno_model is not None and fno_ckpt is not None and fno_path is not None:
            name = periodic_model_label(fno_ckpt)
            models[name] = (fno_model, fno_ckpt)
            paths[name] = fno_path
        rollout_fn = rollout_periodic_model
        weno_builder = build_periodic_solver
        bc = str(meta.get("boundary", "periodic"))
    else:
        ckpt = args.ckpt or PRI_HYBRID_CKPT
        ckpt_fno = args.ckpt_fno or PRI_FNO_CKPT
        test_name = args.test_name or PRI_TEST_NAME
        test_path = resolve_path(args.test_path or os.path.join(args.data_dir, test_name))
        states, meta = load_pri_split(test_path, dtype=torch.float32)
        total = int(states.shape[0])
        selected = select_indices(total, args.n_samples, args.sample_seed)
        first_frames = states[torch.as_tensor(selected, dtype=torch.long), 0]
        del states
        primary_model, primary_ckpt = load_pri_model(resolve_path(ckpt), device, no_compile=args.no_compile)
        models = {pri_model_label(primary_ckpt): (primary_model, primary_ckpt)}
        paths = {pri_model_label(primary_ckpt): resolve_path(ckpt)}
        fno_model, fno_ckpt = load_optional_pri_model(
            resolve_path(ckpt_fno),
            device,
            no_compile=args.no_compile,
            default_path=resolve_path(PRI_FNO_CKPT),
        )
        if fno_model is not None and fno_ckpt is not None:
            name = pri_model_label(fno_ckpt)
            models[name] = (fno_model, fno_ckpt)
            paths[name] = resolve_path(ckpt_fno)
        rollout_fn = rollout_pri_model
        weno_builder = build_outflow_solver
        bc = str(meta.get("boundary", "outflow"))

    frame_dt = float(meta["dt"])
    gamma = float(meta.get("gamma", 1.4))
    dx = float(meta["dx"])
    dy = float(meta["dy"])
    nx = int(meta["nx"])
    ny = int(meta["ny"])
    meta_steps = int(meta.get("n_snaps", int(round(float(meta.get("T", 0.0)) / frame_dt)) + 1)) - 1
    horizon_steps = int(np.floor(float(args.T) / frame_dt + 1e-12)) if args.T is not None else meta_steps
    n_steps = horizon_steps if args.rollout_steps <= 0 else min(int(args.rollout_steps), horizon_steps)
    if n_steps < 1:
        raise ValueError(f"Need at least one rollout step, got {n_steps}.")

    weno_cfl = float(meta.get("cfl", 0.4) if args.weno_cfl is None else args.weno_cfl)
    reconstruction = args.weno_reconstruction or str(meta.get("reconstruction", "component"))
    weno_op = weno_builder(
        gamma=gamma,
        bc=bc,
        reconstruction=reconstruction,
        WENOtype="WENO-Z",
        verbose=True,
        line_batch_size=int(args.weno_line_batch_size),
    )
    warmup_steps = 0 if args.skip_warmup else min(max(int(args.warmup_steps), 0), n_steps)

    print(
        f"[data] case={case}, test_path={test_path}, n_samples={len(first_frames)}, "
        f"steps={n_steps}, dt={frame_dt:g}, nx={nx}, ny={ny}, bc={bc}, recon={reconstruction}, cfl={weno_cfl:g}"
    )
    print(f"[compile] neural models compile={'off' if args.no_compile else 'on'}")
    if warmup_steps:
        print(f"[warmup] neural models run {warmup_steps} warmup step(s) before timing")
    for name, (model, _) in sorted(models.items()):
        print(f"[model] {name}: params={count_params(model):,}, path={paths[name]}")

    rows: list[dict[str, object]] = []
    first_plot_pack = None
    for order, sample_idx in enumerate(selected):
        u0 = first_frames[order]
        print(f"[sample] {order + 1}/{len(selected)} idx={int(sample_idx)}")
        if case == "periodic":
            weno_traj, weno_seconds = timed_periodic_weno(
                weno_op,
                u0,
                dx=dx,
                dy=dy,
                dt=frame_dt,
                n_steps=n_steps,
                cfl=weno_cfl,
                gamma=gamma,
            )
        else:
            weno_traj, weno_seconds = timed_pri_weno(
                weno_op,
                u0,
                dx=dx,
                dy=dy,
                dt=frame_dt,
                n_steps=n_steps,
                cfl=weno_cfl,
                gamma=gamma,
            )
        append_row(rows, sample_order=order, sample_idx=int(sample_idx), method=f"weno{nx}", seconds=weno_seconds, n_steps=n_steps, dt=frame_dt, nx=nx, ny=ny)

        pred_trajs: dict[str, np.ndarray] = {}
        for name, (model, _) in models.items():
            if warmup_steps:
                warmup_model(rollout_fn, model, u0, n_steps=warmup_steps, dt=frame_dt, device=device, pin_memory=pin_mem)
            pred, seconds = timed_model_rollout(
                rollout_fn,
                model,
                u0,
                n_steps=n_steps,
                dt=frame_dt,
                device=device,
                pin_memory=pin_mem,
            )
            pred_trajs[name] = pred
            append_row(rows, sample_order=order, sample_idx=int(sample_idx), method=name, seconds=seconds, n_steps=n_steps, dt=frame_dt, nx=nx, ny=ny)
            print(f"[time] {name}: {seconds:.6f}s")
        print(f"[time] weno{nx}: {weno_seconds:.6f}s")

        if first_plot_pack is None:
            ref_prim = primitive_traj_for_case(case, weno_traj, gamma)
            pred_prims = {name: primitive_traj_for_case(case, traj, gamma) for name, traj in pred_trajs.items()}
            first_plot_pack = (int(sample_idx), ref_prim, pred_prims)

    summary = summarize(rows)
    write_csv(os.path.join(args.outdir, "runtime_rows.csv"), rows)
    write_csv(os.path.join(args.outdir, "runtime_summary.csv"), summary)
    if first_plot_pack is not None and not args.no_plot_solution:
        sample_idx, ref_prim, pred_prims = first_plot_pack
        for path in save_final_plots(
            args.outdir,
            case,
            sample_idx,
            ref_prim,
            pred_prims,
            plot_format=args.solution_plot_format,
        ):
            print(f"[plot] saved {path}")
    for row in summary:
        print(f"[summary] {row['method']}: mean={row['mean_seconds']:.6g}s, speedup_vs_weno={row['speedup_vs_weno']:.3g}")
    print(f"Saved timing CSVs under {args.outdir}")


if __name__ == "__main__":
    main()
