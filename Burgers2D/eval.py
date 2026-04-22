"""
Rollout Burgers2D dt-step checkpoints vs WENO2D reference and/or evaluate one-step test error.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from conslaw.checkpoints import compile_if_requested, load_hybrid_dt_step_2d
from conslaw.eval_reports import dataset_error_row, save_rollout_reports, save_test_metrics_csv
from conslaw.models import count_params
from conslaw.solver import downsample_cell_average2d

from dataset.data import build_burgers2d_solver, make_periodic_grid, random_fourier_ic2d

torch.set_default_dtype(torch.float64)

DEFAULT_HYBRID_CKPT = "checkpoints/burgers2d_hybrid_dt.pt"
DEFAULT_FNO_CKPT = "checkpoints/burgers2d_fno_dt_16.pt"
DEFAULT_CNN_CKPT = "checkpoints/burgers2d_cnn_dt.pt"
LINE_GRID_ALPHA = 0.3


def apply_line_grid(ax) -> None:
    ax.grid(True, alpha=LINE_GRID_ALPHA, linewidth=0.8)


def rel_l1(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + eps))


def rel_linf(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + eps))


def resolve_input_path(path: str, *, label: str) -> str:
    if not path:
        raise ValueError(f"Empty {label} path.")
    if os.path.isabs(path):
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"{label} not found: {path}")

    candidates = [path, os.path.join(_SCRIPT_DIR, path)]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"{label} not found: {path}")


def resolve_optional_path(path: str, default_path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path) or os.path.exists(path):
        return path
    script_path = os.path.join(_SCRIPT_DIR, path)
    if os.path.exists(script_path):
        return script_path
    if path == default_path:
        return path
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def display_model_name(name: str) -> str:
    return {
        "ref": "Ref",
        "hybrid": "Hybrid",
        "fno": "FNO",
        "cnn": "CNN",
    }.get(str(name).lower(), str(name))


def resolve_test_path(args: argparse.Namespace) -> str:
    if args.test_path:
        return resolve_input_path(args.test_path, label="Test dataset")
    return resolve_input_path(os.path.join(args.data_dir, "test_pv_noavg.pt"), label="Test dataset")


def resolve_rollout_dt(args: argparse.Namespace, primary_ckpt: dict) -> float:
    if args.dt_snap is not None:
        return float(args.dt_snap)
    dt_ckpt = primary_ckpt.get("dt")
    if dt_ckpt is not None:
        return float(dt_ckpt)
    train_pt = resolve_input_path(os.path.join(args.data_dir, "train_pv_noavg.pt"), label="Training dataset")
    meta = torch.load(train_pt, map_location="cpu")["meta"]
    dt_snap = float(meta["dt"])
    print(f"[eval] dt from {train_pt}: {dt_snap}")
    return dt_snap


def solve_burgers2d_weno_ref(
    u0_fine: np.ndarray,
    times: np.ndarray,
    dx_fine: float,
    dy_fine: float,
    *,
    cfl: float = 0.4,
    use_numba: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    solver = build_burgers2d_solver(use_numba=use_numba and not verbose)
    if len(times) <= 1:
        return np.asarray(u0_fine, dtype=np.float64)[None, ...]

    dt_all = np.diff(times)
    if np.allclose(dt_all, dt_all[0]) and not verbose:
        return solver.solve_snapshots(
            u0_fine,
            dx=dx_fine,
            dy=dy_fine,
            dt_snap=float(dt_all[0]),
            n_snaps=len(times),
            cfl=cfl,
        )

    u = np.asarray(u0_fine, dtype=np.float64).copy()
    snaps = [u.copy()]
    t = 0.0
    for out_idx in range(1, len(times)):
        t_target = float(times[out_idx])
        dt_interval = t_target - t
        u = solver.advance(u, dx=dx_fine, dy=dy_fine, T=dt_interval, cfl=cfl, return_all=False)
        t = t_target
        snaps.append(u.copy())
        if verbose:
            print(f"    [WENO] snapshot {out_idx + 1}/{len(times)} t={t_target:.4f}", flush=True)
    return np.stack(snaps, axis=0)


def block_average_2d(snaps_hi: np.ndarray, upsample: int) -> np.ndarray:
    return np.stack(
        [downsample_cell_average2d(snaps_hi[j], upsample, upsample) for j in range(snaps_hi.shape[0])],
        axis=0,
    )


@torch.no_grad()
def rollout_step_model_2d(
    step_model: torch.nn.Module,
    u0_low: np.ndarray,
    n_steps: int,
    dt: float,
    device: torch.device,
    *,
    pin_memory: bool,
    verbose: bool = False,
    label: str = "model",
) -> np.ndarray:
    step_model.eval()
    u = torch.from_numpy(u0_low).unsqueeze(0).unsqueeze(-1)
    if pin_memory and device.type == "cuda":
        u = u.pin_memory()
    u = u.to(device=device, dtype=torch.float64, non_blocking=pin_memory and device.type == "cuda")
    traj = [u0_low.copy()]
    for step in range(n_steps):
        if verbose:
            print(f"    [{label}] step {step + 1}/{n_steps} ...", end=" ", flush=True)
        dt_b = torch.full((u.size(0),), float(dt), device=device, dtype=torch.float64)
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        u_in = u.clone() if device.type == "cuda" else u
        u = step_model(u_in, dt_b).clone()
        traj.append(u.squeeze(0).squeeze(-1).cpu().numpy())
        if verbose:
            print("done", flush=True)
    return np.stack(traj, axis=0)


def load_model_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    *,
    no_compile: bool,
) -> tuple[torch.nn.Module, dict, str]:
    resolved_ckpt = resolve_input_path(ckpt_path, label="Checkpoint")
    model, ckpt = load_hybrid_dt_step_2d(resolved_ckpt, device=device)
    model = compile_if_requested(model, device, no_compile=no_compile)
    return model, ckpt, resolved_ckpt


def load_optional_model(
    ckpt_path: str,
    device: torch.device,
    *,
    no_compile: bool,
    default_path: str,
) -> tuple[torch.nn.Module | None, dict | None, str | None]:
    if not ckpt_path:
        return None, None, None
    resolved_ckpt = resolve_optional_path(ckpt_path, default_path)
    if not resolved_ckpt or not os.path.isfile(resolved_ckpt):
        if ckpt_path == default_path:
            print(f"[eval] checkpoint not found, skipping baseline: {ckpt_path}")
            return None, None, None
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model, ckpt = load_hybrid_dt_step_2d(resolved_ckpt, device=device)
    model = compile_if_requested(model, device, no_compile=no_compile)
    return model, ckpt, resolved_ckpt


def model_label_from_ckpt(ckpt: dict) -> str:
    kind = str(ckpt.get("kind", ""))
    if "cnn" in kind:
        return "cnn"
    if "fno" in kind:
        return "fno"
    return "hybrid"


@torch.no_grad()
def evaluate_test_dataset_2d(
    step_model: torch.nn.Module,
    test_pt_path: str,
    device: torch.device,
    *,
    pin_memory: bool,
    batch_size: int,
    model_name: str = "model",
) -> tuple[np.ndarray, np.ndarray]:
    data = torch.load(test_pt_path, map_location="cpu")
    u0_all = data["input"].unsqueeze(-1).to(torch.float64)
    u1_all = data["output"].unsqueeze(-1).to(torch.float64)
    dt_eval = float(data["meta"]["dt"])
    preds = []
    step_model.eval()
    n_total = int(u0_all.size(0))
    n_batches = max(1, (n_total + batch_size - 1) // batch_size)
    print(
        f"[test] {model_name}: {n_total} samples, batch_size={batch_size}, batches={n_batches}",
        flush=True,
    )
    for batch_idx, start in enumerate(range(0, u0_all.size(0), batch_size), start=1):
        u0 = u0_all[start : start + batch_size]
        if pin_memory and device.type == "cuda":
            u0 = u0.pin_memory()
        u0 = u0.to(device=device, dtype=torch.float64, non_blocking=pin_memory and device.type == "cuda")
        dt_b = torch.full((u0.size(0),), dt_eval, device=device, dtype=torch.float64)
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        preds.append(step_model(u0, dt_b).cpu().numpy())
        if batch_idx == 1 or batch_idx == n_batches or batch_idx % max(1, n_batches // 10) == 0:
            print(
                f"[test] {model_name}: batch {batch_idx}/{n_batches}",
                flush=True,
            )
    return np.concatenate(preds, axis=0), u1_all.numpy()


def save_rollout_plots(
    args: argparse.Namespace,
    times: np.ndarray,
    rollout_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    ref_traj: np.ndarray | None,
    pred_trajs: dict[str, np.ndarray] | None,
) -> None:
    fig_seed = f"_seed{args.seed}"

    def save_metric_curves_pdf(metric_label: str, filename: str, series: dict[str, np.ndarray]) -> None:
        if times.size <= 1:
            return
        times_plot = times[1:]
        ylabel = r"Relative $L^\infty$ error" if "linf" in metric_label.lower() else r"Relative $L^1$ error"
        fig, ax = plt.subplots(figsize=(6, 4))
        for model_name, values in series.items():
            mean = values.mean(axis=0)[1:]
            std = values.std(axis=0)[1:]
            ax.plot(times_plot, mean, label=display_model_name(model_name))
            ax.fill_between(times_plot, mean - std, mean + std, alpha=0.15)
        ax.set_yscale("log")
        ax.set_xlabel("t", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend()
        apply_line_grid(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, filename), bbox_inches="tight")
        plt.close(fig)

    def save_field_pdf(
        field: np.ndarray,
        *,
        path: str,
        cmap: str,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        im = ax.imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def save_xt_pdf(
        x_coords: np.ndarray,
        xt_values: np.ndarray,
        *,
        title: str,
        path: str,
        cmap: str,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        dx = float(np.median(np.diff(x_coords))) if x_coords.size > 1 else 1.0
        extent = [float(x_coords[0]), float(x_coords[-1] + dx), float(times[0]), float(times[-1])]
        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        im = ax.imshow(xt_values, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("t", fontsize=14)
        ax.set_box_aspect(1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def save_slice_pdf(
        x_coords: np.ndarray,
        curves: list[tuple[str, np.ndarray]],
        *,
        path: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, values in curves:
            ax.plot(x_coords, values, label=label, linewidth=1.8)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.legend()
        apply_line_grid(ax)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    save_metric_curves_pdf("rel L1", f"relL1_vs_time_hybrid_dt2d{fig_seed}.pdf", {name: values[0] for name, values in rollout_scores.items()})
    save_metric_curves_pdf(
        "rel Linf",
        f"relLinf_vs_time_hybrid_dt2d{fig_seed}.pdf",
        {name: values[1] for name, values in rollout_scores.items()},
    )

    if not args.plot_one or ref_traj is None or not pred_trajs:
        return

    model_names = list(pred_trajs.keys())
    plot_interval = 0.5
    plot_targets = np.arange(0.0, times[-1] + 1e-12, plot_interval)
    plot_indices: list[int] = []
    tol = 1e-9
    if times.size > 1:
        tol = max(tol, 0.5 * float(np.min(np.diff(times))) + 1e-12)
    for t_target in plot_targets:
        idx = int(np.argmin(np.abs(times - t_target)))
        if abs(float(times[idx]) - float(t_target)) <= tol and idx not in plot_indices:
            plot_indices.append(idx)
    if (len(times) - 1) not in plot_indices:
        plot_indices.append(len(times) - 1)
    plot_indices.sort()

    for idx in plot_indices:
        t_curr = float(times[idx])
        t_tag = f"{t_curr:.3f}".replace(".", "p")
        u_ref = ref_traj[idx]
        save_field_pdf(
            u_ref,
            path=os.path.join(args.outdir, f"solution_ref_t{t_tag}_hybrid_dt2d{fig_seed}.pdf"),
            cmap="viridis",
        )
        for name in model_names:
            u_pred = pred_trajs[name][idx]
            err = np.abs(u_pred - u_ref)
            save_field_pdf(
                u_pred,
                path=os.path.join(args.outdir, f"solution_{name}_t{t_tag}_hybrid_dt2d{fig_seed}.pdf"),
                cmap="viridis",
            )
            save_field_pdf(
                err,
                path=os.path.join(args.outdir, f"error_{name}_t{t_tag}_hybrid_dt2d{fig_seed}.pdf"),
                cmap="magma",
                vmin=0.0,
                vmax=max(float(err.max()), 1e-12),
            )

    final_idx = len(times) - 1
    final_ref = ref_traj[final_idx]
    final_curve_names = [name for name in ("hybrid", "fno") if name in pred_trajs]
    if final_curve_names:
        nx_low = final_ref.shape[1]
        x_coords = np.linspace(0.0, 1.0, nx_low, endpoint=False)
        for y_val in np.arange(0.1, 1.0, 0.1):
            j = int(round(float(y_val) * (final_ref.shape[0] - 1)))
            y_tag = f"{float(y_val):.1f}".replace(".", "p")
            curves = [("Ref", final_ref[j, :])]
            for name in final_curve_names:
                curves.append((display_model_name(name), pred_trajs[name][final_idx, j, :]))
            save_slice_pdf(
                x_coords,
                curves,
                path=os.path.join(args.outdir, f"slice_final_y{y_tag}_hybrid_dt2d{fig_seed}.pdf"),
            )

    ny_low, nx_low = u_ref.shape
    x_coords = np.linspace(0.0, 1.0, nx_low, endpoint=False)
    for y_val in args.yslices:
        y_clamped = min(max(float(y_val), 0.0), 1.0)
        j = int(round(y_clamped * (ny_low - 1)))
        y_tag = f"{y_clamped:.3f}".replace(".", "p")
        ref_xt = ref_traj[:, j, :]
        save_xt_pdf(
            x_coords,
            ref_xt,
            title="",
            path=os.path.join(args.outdir, f"xt_ref_y{y_tag}_hybrid_dt2d{fig_seed}.pdf"),
            cmap="viridis",
        )
        for name in model_names:
            pred_xt = pred_trajs[name][:, j, :]
            err_xt = np.abs(pred_xt - ref_xt)
            save_xt_pdf(
                x_coords,
                pred_xt,
                title="",
                path=os.path.join(args.outdir, f"xt_{name}_y{y_tag}_hybrid_dt2d{fig_seed}.pdf"),
                cmap="viridis",
            )
            save_xt_pdf(
                x_coords,
                err_xt,
                title="",
                path=os.path.join(args.outdir, f"xt_error_{name}_y{y_tag}_hybrid_dt2d{fig_seed}.pdf"),
                cmap="magma",
                vmin=0.0,
                vmax=max(float(err_xt.max()), 1e-12),
            )


def run_rollout_eval(
    args: argparse.Namespace,
    models: dict[str, tuple[torch.nn.Module, dict]],
    model_paths: dict[str, str],
    dt_snap: float,
    device: torch.device,
    *,
    pin_memory: bool,
) -> None:
    times = np.arange(0.0, args.T + 1e-12, dt_snap)
    n_steps = len(times) - 1
    nx_high = args.nx_low * args.upsample
    ny_high = args.ny_low * args.upsample
    xx_fine, yy_fine, dx_fine, dy_fine = make_periodic_grid(nx_high, ny_high)
    dx_low = 1.0 / args.nx_low
    dy_low = 1.0 / args.ny_low
    rng = np.random.default_rng(args.seed)

    print(
        f"[rollout] dt={dt_snap}, nx_low={args.nx_low}, ny_low={args.ny_low}, "
        f"upsample={args.upsample}, n_samples={args.n_samples}"
    )
    for name in sorted(models.keys()):
        ckpt = models[name][1]
        print(
            f"[model {name}] path={model_paths[name]}, "
            f"dt/dx/dy=({float(ckpt.get('dt', 0.0))},{float(ckpt.get('dx', 0.0))},{float(ckpt.get('dy', 0.0))})"
        )

    if not args.no_compile and not args.no_warmup and n_steps > 0:
        u_warm = np.zeros((args.ny_low, args.nx_low), dtype=np.float64)
        for name, (model, _) in models.items():
            print(f"[warmup] {name} rollout ...", flush=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            rollout_step_model_2d(
                model,
                u_warm,
                n_steps,
                dt_snap,
                device,
                pin_memory=pin_memory,
                verbose=False,
                label=name,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        print("[warmup] done", flush=True)

    rollout_scores = {
        name: (
            np.zeros((args.n_samples, len(times)), dtype=np.float64),
            np.zeros((args.n_samples, len(times)), dtype=np.float64),
        )
        for name in models
    }
    solver_times: list[float] = []
    model_times = {name: [] for name in models}
    one_ref = None
    one_preds = None

    for sample_idx in range(args.n_samples):
        if args.verbose:
            print(f"[sample {sample_idx + 1}/{args.n_samples}] IC...", flush=True)
        u0 = random_fourier_ic2d(
            xx_fine,
            yy_fine,
            kmax=args.kmax,
            amp_range=(args.amp_min, args.amp_max),
            decay=args.decay,
            rng=rng,
        )
        t0 = time.perf_counter()
        snaps_hi = solve_burgers2d_weno_ref(
            u0,
            times,
            dx_fine,
            dy_fine,
            cfl=args.cfl,
            use_numba=args.use_numba,
            verbose=args.verbose,
        )
        solver_times.append(time.perf_counter() - t0)
        ref_low = block_average_2d(snaps_hi, args.upsample)

        pred_trajs: dict[str, np.ndarray] = {}
        for name, (model, _) in models.items():
            print(
                f"[rollout] sample {sample_idx + 1}/{args.n_samples}: running {name} ...",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = rollout_step_model_2d(
                model,
                ref_low[0],
                n_steps,
                dt_snap,
                device,
                pin_memory=pin_memory,
                verbose=args.verbose,
                label=name,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            model_times[name].append(elapsed)
            pred_trajs[name] = pred
            print(
                f"[rollout] sample {sample_idx + 1}/{args.n_samples}: {name} done in {elapsed:.3f}s",
                flush=True,
            )

            l1, linf = rollout_scores[name]
            for j in range(len(times)):
                l1[sample_idx, j] = rel_l1(pred[j], ref_low[j])
                linf[sample_idx, j] = rel_linf(pred[j], ref_low[j])

        if args.plot_one and one_ref is None:
            one_ref = ref_low.copy()
            one_preds = {name: pred.copy() for name, pred in pred_trajs.items()}

        print(f"[{sample_idx + 1}/{args.n_samples}] done")

    timing_msg = f"[timing] solver mean {np.mean(solver_times):.3f}s"
    for name in sorted(models.keys()):
        timing_msg += f", {name} mean {np.mean(model_times[name]):.3f}s"
    print(timing_msg)

    npz_payload = {
        "times": times,
        "dt": np.array([dt_snap], dtype=np.float64),
        "dx_low": np.array([dx_low], dtype=np.float64),
        "dy_low": np.array([dy_low], dtype=np.float64),
        "solver_time_s": np.asarray(solver_times, dtype=np.float64),
    }
    report_payload = {}
    for name, (l1, linf) in rollout_scores.items():
        npz_payload[f"l1_{name}"] = l1
        npz_payload[f"linf_{name}"] = linf
        npz_payload[f"{name}_time_s"] = np.asarray(model_times[name], dtype=np.float64)
        report_payload[name] = {"l1": l1, "linf": linf}
        print(
            f"[rollout] {name}: final relL1={l1[:, -1].mean():.3e}, "
            f"final relLinf={linf[:, -1].mean():.3e}"
        )
    np.savez_compressed(os.path.join(args.outdir, "metrics_hybrid_dt2d.npz"), **npz_payload)
    summary_path, curves_path = save_rollout_reports(args.outdir, times, report_payload)
    print(f"[rollout] saved csv: {summary_path}, {curves_path}")

    save_rollout_plots(args, times, rollout_scores, one_ref, one_preds)


def run_test_eval(
    args: argparse.Namespace,
    models: dict[str, tuple[torch.nn.Module, dict]],
    test_path: str,
    device: torch.device,
    *,
    pin_memory: bool,
) -> None:
    test_rows = []
    target = None
    for name, (model, _) in models.items():
        t0 = time.perf_counter()
        pred, maybe_target = evaluate_test_dataset_2d(
            model,
            test_path,
            device,
            pin_memory=pin_memory,
            batch_size=args.test_batch,
            model_name=name,
        )
        elapsed = time.perf_counter() - t0
        if target is None:
            target = maybe_target
        test_rows.append(dataset_error_row(name, pred, target))
        print(f"[test] {name}: finished in {elapsed:.3f}s", flush=True)
    test_csv_path = save_test_metrics_csv(args.outdir, test_rows)
    print(f"[test] saved csv: {test_csv_path}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_HYBRID_CKPT)
    ap.add_argument("--ckpt_fno", type=str, default=DEFAULT_FNO_CKPT)
    ap.add_argument("--ckpt_cnn", type=str, default=DEFAULT_CNN_CKPT)
    ap.add_argument("--hybrid_only", action="store_true", help="Evaluate only the primary --ckpt model.")
    ap.add_argument("--eval_mode", type=str, default="rollout", choices=("rollout", "test", "both"))
    ap.add_argument("--test_path", type=str, default=None)
    ap.add_argument("--test_batch", type=int, default=32)
    ap.add_argument("--nx_low", type=int, default=128)
    ap.add_argument("--ny_low", type=int, default=128)
    ap.add_argument("--upsample", type=int, default=4)
    ap.add_argument("--T", type=float, default=0.3)
    ap.add_argument("--dt_snap", type=float, default=None)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--decay", type=float, default=1.5)
    ap.add_argument("--amp_min", type=float, default=0.4)
    ap.add_argument("--amp_max", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=str, default="eval_burgers2d_hybrid_dt_out")
    ap.add_argument("--plot_one", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--no_warmup", action="store_true")
    ap.add_argument("--use_numba", action="store_true")
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--yslices", type=float, nargs="+", default=[0.4])
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    pin_mem = device.type == "cuda"

    primary_model, primary_ckpt, primary_path = load_model_from_ckpt(
        args.ckpt,
        device,
        no_compile=args.no_compile,
    )
    primary_name = model_label_from_ckpt(primary_ckpt)
    models: dict[str, tuple[torch.nn.Module, dict]] = {primary_name: (primary_model, primary_ckpt)}
    model_paths = {primary_name: primary_path}
    model_fno = None

    if not args.hybrid_only:
        model_fno, ckpt_fno, path_fno = load_optional_model(
            args.ckpt_fno,
            device,
            no_compile=args.no_compile,
            default_path=DEFAULT_FNO_CKPT,
        )
        if model_fno is not None and ckpt_fno is not None and path_fno is not None:
            name_fno = model_label_from_ckpt(ckpt_fno)
            models[name_fno] = (model_fno, ckpt_fno)
            model_paths[name_fno] = path_fno

        model_cnn, ckpt_cnn, path_cnn = load_optional_model(
            args.ckpt_cnn,
            device,
            no_compile=args.no_compile,
            default_path=DEFAULT_CNN_CKPT,
        )
        if model_cnn is not None and ckpt_cnn is not None and path_cnn is not None:
            name_cnn = model_label_from_ckpt(ckpt_cnn)
            models[name_cnn] = (model_cnn, ckpt_cnn)
            model_paths[name_cnn] = path_cnn

    print(f"[params] hybrid={count_params(primary_model):,}")
    if model_fno is not None:
        print(f"[params] fno={count_params(model_fno):,}")
    else:
        print("[params] fno=unavailable")

    dt_snap = resolve_rollout_dt(args, primary_ckpt)
    print("[models] " + ", ".join(f"{name}={model_paths[name]}" for name in sorted(models.keys())))

    if args.eval_mode in ("rollout", "both"):
        run_rollout_eval(args, models, model_paths, dt_snap, device, pin_memory=pin_mem)

    if args.eval_mode in ("test", "both"):
        test_path = resolve_test_path(args)
        print(f"[data] test_path={test_path}")
        run_test_eval(args, models, test_path, device, pin_memory=pin_mem)

    print(f"Saved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
