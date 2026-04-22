"""
Evaluate Euler2D dt-step checkpoints on trajectory test data.
"""

from __future__ import annotations

import argparse
import os
import sys

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

from common_periodic import (
    DEFAULT_TEST_NAME,
    configure_runtime,
    default_dt_ckpt_path,
    load_trajectory_split,
    predict_pairs,
    rollout_step_model,
    select_trajectories,
    states_to_one_step_pairs,
)


DEFAULT_HYBRID_CKPT = default_dt_ckpt_path("hybrid", "periodic")
DEFAULT_FNO_CKPT = default_dt_ckpt_path("fno", "periodic")
DEFAULT_CNN_CKPT = default_dt_ckpt_path("cnn", "periodic")
LINE_GRID_ALPHA = 0.3


def apply_line_grid(ax) -> None:
    ax.grid(True, alpha=LINE_GRID_ALPHA, linewidth=0.8)


def rel_l1(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + eps))


def rel_linf(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + eps))


def display_model_name(name: str) -> str:
    return {
        "ref": "Ref",
        "hybrid": "Hybrid",
        "fno": "FNO",
        "cnn": "CNN",
    }.get(str(name).lower(), str(name))


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


def resolve_test_split_path(args: argparse.Namespace) -> str:
    if args.test_path:
        return resolve_input_path(args.test_path, label="Test dataset")
    return resolve_input_path(os.path.join(args.data_dir, args.test_name), label="Test dataset")


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


def evaluate_test_dataset(
    model: torch.nn.Module,
    test_states: torch.Tensor,
    dt: float,
    device: torch.device,
    *,
    batch_size: int,
    pin_memory: bool,
) -> tuple[np.ndarray, np.ndarray]:
    u0_test, u1_test = states_to_one_step_pairs(test_states, dtype=torch.float32)
    pred = predict_pairs(
        model,
        u0_test,
        dt,
        device,
        dtype=torch.float32,
        batch_size=batch_size,
        pin_memory=pin_memory,
    )
    return pred, u1_test.numpy()


def save_density_plots(
    outdir: str,
    times: np.ndarray,
    ref_traj: np.ndarray,
    pred_trajs: dict[str, np.ndarray],
    plot_channel: int,
    yslices: list[float],
) -> None:
    final_ref = ref_traj[-1, plot_channel]
    model_names = list(pred_trajs.keys())

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
        ax.set_ylabel("value")
        ax.legend()
        apply_line_grid(ax)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def save_field_pdf(
        field: np.ndarray,
        *,
        title: str,
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
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_box_aspect(1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    final_fields = [pred_trajs[name][-1, plot_channel] for name in model_names]
    final_figsize = (4.6 * (1 + len(model_names)), 4.2)
    fig_final, axes_final = plt.subplots(1, 1 + len(model_names), figsize=final_figsize)
    axes_final = np.atleast_1d(axes_final)
    for ax, field in zip(
        axes_final,
        [final_ref, *final_fields],
    ):
        im = ax.imshow(field, origin="lower", cmap="viridis")
        ax.set_xticks([])
        ax.set_yticks([])
        fig_final.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_final.tight_layout()
    fig_final.savefig(os.path.join(outdir, f"final_channel{plot_channel}_states.pdf"), bbox_inches="tight")
    plt.close(fig_final)

    fig_err, axes_err = plt.subplots(1, len(model_names), figsize=final_figsize)
    axes_err = np.atleast_1d(axes_err)
    for ax, name in zip(axes_err, model_names):
        err = np.abs(pred_trajs[name][-1, plot_channel] - final_ref)
        im = ax.imshow(err, origin="lower", cmap="magma", vmin=0.0, vmax=max(float(err.max()), 1e-12))
        ax.set_xticks([])
        ax.set_yticks([])
        fig_err.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_err.tight_layout()
    fig_err.savefig(os.path.join(outdir, f"final_channel{plot_channel}_errors.pdf"), bbox_inches="tight")
    plt.close(fig_err)

    ny_low, nx_low = final_ref.shape
    x_coords = np.linspace(0.0, 1.0, nx_low, endpoint=False)
    final_curve_names = [name for name in ("hybrid", "fno") if name in pred_trajs]
    if final_curve_names:
        for y_val in yslices:
            y_clamped = min(max(float(y_val), 0.0), 1.0)
            j = int(round(y_clamped * (ny_low - 1)))
            y_tag = f"{y_clamped:.3f}".replace(".", "p")
            curves = [("Ref", final_ref[j, :])]
            for name in final_curve_names:
                curves.append((display_model_name(name), pred_trajs[name][-1, plot_channel, j, :]))
            save_slice_pdf(
                x_coords,
                curves,
                path=os.path.join(outdir, f"slice_channel{plot_channel}_final_y{y_tag}.pdf"),
            )

    for y_val in yslices:
        y_clamped = min(max(float(y_val), 0.0), 1.0)
        j = int(round(y_clamped * (ny_low - 1)))
        y_tag = f"{y_clamped:.3f}".replace(".", "p")
        ref_xt = ref_traj[:, plot_channel, j, :]
        save_xt_pdf(
            x_coords,
            ref_xt,
            title="",
            path=os.path.join(outdir, f"xt_channel{plot_channel}_ref_y{y_tag}.pdf"),
            cmap="viridis",
        )
        for name in model_names:
            pred_xt = pred_trajs[name][:, plot_channel, j, :]
            err_xt = np.abs(pred_xt - ref_xt)
            save_xt_pdf(
                x_coords,
                pred_xt,
                title="",
                path=os.path.join(outdir, f"xt_channel{plot_channel}_{name}_y{y_tag}.pdf"),
                cmap="viridis",
            )
            save_xt_pdf(
                x_coords,
                err_xt,
                title="",
                path=os.path.join(outdir, f"xt_channel{plot_channel}_error_{name}_y{y_tag}.pdf"),
                cmap="magma",
                vmin=0.0,
                vmax=max(float(err_xt.max()), 1e-12),
            )


def run_rollout_eval(
    args: argparse.Namespace,
    models: dict[str, tuple[torch.nn.Module, dict]],
    test_states: torch.Tensor,
    dt: float,
    device: torch.device,
    *,
    pin_memory: bool,
) -> None:
    if args.T is not None:
        max_time = float(args.T)
        if max_time < 0.0:
            raise ValueError("--T must be non-negative.")
        max_steps = int(np.floor(max_time / dt + 1e-12))
        n_snaps_keep = min(int(test_states.shape[1]), max_steps + 1)
        if n_snaps_keep < 2:
            raise ValueError(
                f"--T={max_time:g} is too small for dt={dt:g}; need at least one rollout step."
            )
        test_states = test_states[:, :n_snaps_keep]
        print(
            f"[rollout] truncating trajectories to T<={max_time:g} "
            f"({n_snaps_keep} snapshots, final t={(n_snaps_keep - 1) * dt:g})"
        )

    n_eval = int(test_states.shape[0])
    times = np.arange(int(test_states.shape[1]), dtype=np.float64) * dt
    rollout_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    one_ref = None
    one_preds = None

    for name, (model, _) in models.items():
        l1 = np.zeros((n_eval, len(times)), dtype=np.float64)
        linf = np.zeros((n_eval, len(times)), dtype=np.float64)
        for idx in range(n_eval):
            ref = test_states[idx].cpu().numpy()
            pred = rollout_step_model(
                model,
                test_states[idx, 0],
                n_steps=int(test_states.shape[1] - 1),
                dt=dt,
                device=device,
                dtype=torch.float32,
                pin_memory=pin_memory,
            )
            for j in range(len(times)):
                l1[idx, j] = rel_l1(pred[j], ref[j])
                linf[idx, j] = rel_linf(pred[j], ref[j])
            if args.plot_one and idx == 0:
                one_ref = ref
                if one_preds is None:
                    one_preds = {}
                one_preds[name] = pred
        rollout_scores[name] = (l1, linf)
        print(
            f"[rollout] {name}: final relL1={l1[:, -1].mean():.3e}, "
            f"final relLinf={linf[:, -1].mean():.3e}"
        )

    npz_payload = {"times": times, "dt": np.array([dt], dtype=np.float64)}
    report_payload = {}
    for name, (l1, linf) in rollout_scores.items():
        npz_payload[f"l1_{name}"] = l1
        npz_payload[f"linf_{name}"] = linf
        report_payload[name] = {"l1": l1, "linf": linf}
    np.savez_compressed(os.path.join(args.outdir, "rollout_metrics.npz"), **npz_payload)
    summary_path, curves_path = save_rollout_reports(args.outdir, times, report_payload)
    print(f"[rollout] saved csv: {summary_path}, {curves_path}")

    if times.size > 1:
        times_plot = times[1:]
    else:
        times_plot = times

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, (l1, _) in rollout_scores.items():
        mean = l1.mean(axis=0)[1:] if l1.shape[1] > 1 else l1.mean(axis=0)
        std = l1.std(axis=0)[1:] if l1.shape[1] > 1 else l1.std(axis=0)
        ax.plot(times_plot, mean, label=display_model_name(name))
        ax.fill_between(times_plot, mean - std, mean + std, alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("t", fontsize=14)
    ax.set_ylabel(r"Relative $L^1$ error", fontsize=14)
    ax.legend()
    apply_line_grid(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "relL1_vs_time.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, (_, linf) in rollout_scores.items():
        values = linf.mean(axis=0)[1:] if linf.shape[1] > 1 else linf.mean(axis=0)
        ax.plot(times_plot, values, label=display_model_name(name))
    ax.set_yscale("log")
    ax.set_xlabel("t", fontsize=14)
    ax.set_ylabel(r"Relative $L^\infty$ error", fontsize=14)
    ax.legend()
    apply_line_grid(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "relLinf_vs_time.pdf"), bbox_inches="tight")
    plt.close(fig)

    if args.plot_one and one_ref is not None and one_preds is not None:
        save_density_plots(args.outdir, times, one_ref, one_preds, args.plot_channel, args.yslices)


def run_test_eval(
    args: argparse.Namespace,
    models: dict[str, tuple[torch.nn.Module, dict]],
    test_states: torch.Tensor,
    dt: float,
    device: torch.device,
    *,
    pin_memory: bool,
) -> None:
    test_rows = []
    target = None
    for name, (model, _) in models.items():
        pred, maybe_target = evaluate_test_dataset(
            model,
            test_states,
            dt,
            device,
            batch_size=args.test_batch,
            pin_memory=pin_memory,
        )
        if target is None:
            target = maybe_target
        test_rows.append(dataset_error_row(name, pred, target))
    test_csv_path = save_test_metrics_csv(args.outdir, test_rows)
    print(f"[test] saved csv: {test_csv_path}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_HYBRID_CKPT)
    ap.add_argument("--ckpt_fno", type=str, default=DEFAULT_FNO_CKPT)
    ap.add_argument("--ckpt_cnn", type=str, default=DEFAULT_CNN_CKPT)
    ap.add_argument("--hybrid_only", action="store_true", help="Evaluate only the primary --ckpt model.")
    ap.add_argument("--eval_mode", type=str, default="both", choices=("rollout", "test", "both"))
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--test_name", type=str, default=DEFAULT_TEST_NAME)
    ap.add_argument("--test_path", type=str, default=None)
    ap.add_argument("--test_batch", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=0, help="0 means use all test trajectories for rollout.")
    ap.add_argument("--T", type=float, default=None, help="Optional rollout horizon; truncate test trajectories to t<=T.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow_tf32", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--outdir", type=str, default="eval_euler2d_periodic_dt_out")
    ap.add_argument("--plot_one", action="store_true")
    ap.add_argument("--plot_channel", type=int, default=0, help="Conserved channel to visualize, 0=rho.")
    ap.add_argument("--yslices", type=float, nargs="+", default=[0.5], help="y-slices for x-t plots in [0, 1].")
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    configure_runtime(device, allow_tf32=args.allow_tf32)
    pin_mem = device.type == "cuda"

    primary_model, primary_ckpt, primary_path = load_model_from_ckpt(
        args.ckpt,
        device,
        no_compile=args.no_compile,
    )
    models: dict[str, tuple[torch.nn.Module, dict]] = {
        model_label_from_ckpt(primary_ckpt): (primary_model, primary_ckpt)
    }
    model_paths = {model_label_from_ckpt(primary_ckpt): primary_path}
    model_param_counts = {model_label_from_ckpt(primary_ckpt): count_params(primary_model)}

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
            model_param_counts[name_fno] = count_params(model_fno)

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
            model_param_counts[name_cnn] = count_params(model_cnn)

    print(f"[params] hybrid={model_param_counts.get('hybrid', 'unavailable'):,}" if "hybrid" in model_param_counts else "[params] hybrid=unavailable")
    print(f"[params] fno={model_param_counts.get('fno', 'unavailable'):,}" if "fno" in model_param_counts else "[params] fno=unavailable")
    if "cnn" in model_param_counts:
        print(f"[params] cnn={model_param_counts['cnn']:,}")

    test_path = resolve_test_split_path(args)
    test_states, test_meta = load_trajectory_split(test_path)
    dt = float(test_meta["dt"])
    test_states = select_trajectories(test_states, args.n_samples if args.n_samples > 0 else None)
    print(
        f"[data] test_path={test_path}, n_traj={int(test_states.shape[0])}, "
        f"n_snaps={int(test_states.shape[1])}, dt={dt}, "
        f"nx={int(test_meta['nx'])}, ny={int(test_meta['ny'])}, boundary={test_meta.get('boundary', '?')}"
    )
    print(
        "[models] "
        + ", ".join(
            f"{name}={model_paths[name]} (bc={models[name][1].get('bc', models[name][1].get('args', {}).get('bc', '?'))})"
            for name in sorted(models.keys())
        )
    )

    if args.eval_mode in ("rollout", "both"):
        run_rollout_eval(args, models, test_states, dt, device, pin_memory=pin_mem)

    if args.eval_mode in ("test", "both"):
        run_test_eval(args, models, test_states, dt, device, pin_memory=pin_mem)

    print(f"Saved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
