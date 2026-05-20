"""Evaluate Euler2D primitive-variable dt-step checkpoints."""

from __future__ import annotations

import argparse
import gc
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from conslaw.eval_reports import (
    finalize_dataset_error_row,
    init_dataset_error_stats,
    save_rollout_reports,
    save_test_metrics_csv,
    save_test_metrics_summary_csv,
    update_dataset_error_stats,
)
from conslaw.models import count_params

from common_pri import (
    DEFAULT_TEST_NAME,
    PRIMITIVE_CHANNELS,
    configure_runtime,
    load_primitive_dt_step_2d,
    load_trajectory_split,
    rollout_step_model,
    states_to_one_step_pairs,
)


DEFAULT_HYBRID_CKPT = "checkpoints/euler2d_hybrid_pri_outflow_1e-2_dt.pt"
DEFAULT_FNO_CKPT = "checkpoints/euler2d_fno_pri_dt_24.pt"
LINE_GRID_ALPHA = 0.3
FIELD_MATH_LABELS = {
    "rho": r"$\rho$",
    "u": r"$u$",
    "v": r"$v$",
    "p": r"$p$",
}
ERROR_MATH_LABELS = {
    "rho": r"$|\rho-\rho_{\mathrm{ref}}|$",
    "u": r"$|u-u_{\mathrm{ref}}|$",
    "v": r"$|v-v_{\mathrm{ref}}|$",
    "p": r"$|p-p_{\mathrm{ref}}|$",
}


def apply_line_grid(ax) -> None:
    ax.grid(True, alpha=LINE_GRID_ALPHA, linewidth=0.8)


def rel_l1(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + eps))


def rel_linf(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + eps))


def model_label_from_ckpt(ckpt: dict) -> str:
    kind = str(ckpt.get("kind", ""))
    if "cnn" in kind:
        return "cnn"
    if "fno" in kind:
        return "fno"
    return "hybrid"


def display_model_name(name: str) -> str:
    return {
        "ref": "Ref",
        "hybrid": "Hybrid",
        "fno": "FNO",
        "cnn": "CNN",
    }.get(str(name).lower(), str(name))


def field_math_label(key: str) -> str:
    return FIELD_MATH_LABELS.get(key, str(key))


def error_math_label(key: str) -> str:
    return ERROR_MATH_LABELS.get(key, field_math_label(key))


def release_eval_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_required_model(path: str, device: torch.device, *, no_compile: bool) -> tuple[torch.nn.Module, dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return load_primitive_dt_step_2d(path, device=device, no_compile=no_compile)


def load_optional_model(
    path: str,
    device: torch.device,
    *,
    no_compile: bool,
    default_path: str,
) -> tuple[torch.nn.Module | None, dict | None]:
    if not path:
        return None, None
    if not os.path.isfile(path):
        if path == default_path:
            print(f"[eval-pri] checkpoint not found, skipping: {path}")
            return None, None
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return load_primitive_dt_step_2d(path, device=device, no_compile=no_compile)


def select_trajectory_indices(n_total: int, n_samples: int | None, seed: int | None) -> np.ndarray:
    if n_samples is None or n_samples <= 0 or n_samples >= n_total:
        return np.arange(n_total, dtype=np.int64)
    if seed is None:
        return np.arange(int(n_samples), dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=int(n_samples), replace=False).astype(np.int64)


def sample_tag(sample_idx: int, sample_seed: int | None) -> str:
    if sample_seed is None:
        return f"sample_idx{sample_idx:06d}"
    return f"sample_seed{sample_seed}_idx{sample_idx:06d}"


def primitive_fields(state_chw: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(state_chw[i], dtype=np.float64) for i, key in enumerate(PRIMITIVE_CHANNELS)}


def merged_field_limit(fields_by_model: dict[str, dict[str, np.ndarray]], key: str) -> tuple[float, float]:
    vals = [fields[key] for fields in fields_by_model.values() if key in fields]
    return min(float(np.min(v)) for v in vals), max(float(np.max(v)) for v in vals)


class _OneDecimalScalarFormatter(ScalarFormatter):
    """ScalarFormatter with one decimal coefficient and compact scientific notation."""

    def _set_format(self) -> None:
        self.format = "%1.1f"
        if self._usetex or self._useMathText:
            self.format = r"$\mathdefault{%s}$" % self.format


def _style_field_colorbar(cbar) -> None:
    fmt = _OneDecimalScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    cbar.formatter = fmt
    cbar.ax.yaxis.set_major_formatter(fmt)
    cbar.update_ticks()


@torch.no_grad()
def evaluate_pairs_streaming(
    model_name: str,
    model: torch.nn.Module,
    u0: torch.Tensor,
    u1: torch.Tensor,
    dt: float,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 32,
    pin_memory: bool = False,
) -> dict[str, object]:
    stats = init_dataset_error_stats()
    model.eval()
    for start in range(0, int(u0.size(0)), batch_size):
        stop = min(start + batch_size, int(u0.size(0)))
        batch = u0[start:stop]
        target_batch = u1[start:stop]
        if pin_memory and device.type == "cuda":
            batch = batch.pin_memory()
        batch = batch.to(device=device, dtype=dtype, non_blocking=pin_memory and device.type == "cuda")
        dt_b = torch.full((batch.size(0),), float(dt), device=device, dtype=dtype)
        pred = model(batch, dt_b)
        update_dataset_error_stats(stats, pred.detach().cpu().numpy(), target_batch.numpy())
    return finalize_dataset_error_row(model_name, stats)


def save_single_field(
    out_path: str,
    field: np.ndarray,
    *,
    label: str | None = None,
    cmap: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    if label:
        ax.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _style_field_colorbar(cbar)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_field_comparison_panel(
    out_path: str,
    fields_by_model: dict[str, dict[str, np.ndarray]],
    *,
    key: str,
    cmap: str = "turbo",
    limits: tuple[float, float] | None = None,
) -> None:
    names = list(fields_by_model.keys())
    ncols = len(names)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.8), squeeze=False)
    vmin, vmax = limits if limits is not None else (None, None)
    for ax, name in zip(axes[0], names):
        im = ax.imshow(fields_by_model[name][key], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{display_model_name(name)} {field_math_label(key)}")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _style_field_colorbar(cbar)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_error_comparison_panel(
    out_path: str,
    ref_fields: dict[str, np.ndarray],
    pred_fields_by_model: dict[str, dict[str, np.ndarray]],
    *,
    key: str,
) -> None:
    names = list(pred_fields_by_model.keys())
    if not names:
        return
    ncols = len(names)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.8), squeeze=False)
    errors = {name: np.abs(fields[key] - ref_fields[key]) for name, fields in pred_fields_by_model.items()}
    vmax = max(max(float(np.max(err)), 1e-12) for err in errors.values())
    for ax, name in zip(axes[0], names):
        im = ax.imshow(errors[name], origin="lower", cmap="turbo", vmin=0.0, vmax=vmax)
        ax.set_title(f"{display_model_name(name)} {error_math_label(key)}")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _style_field_colorbar(cbar)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_rollout_sample_plots(
    outdir: str,
    ref_traj: np.ndarray,
    pred_trajs: dict[str, np.ndarray],
    *,
    tag: str,
) -> None:
    final_fields = {"ref": primitive_fields(ref_traj[-1])}
    for name, traj in pred_trajs.items():
        final_fields[name] = primitive_fields(traj[-1])

    ref_fields = final_fields["ref"]
    pred_fields = {name: fields for name, fields in final_fields.items() if name != "ref"}
    for key in PRIMITIVE_CHANNELS:
        limits = merged_field_limit(final_fields, key)
        save_field_comparison_panel(
            os.path.join(outdir, f"primitive_{key}_compare_{tag}.pdf"),
            final_fields,
            key=key,
            limits=limits,
        )
        for name, fields in final_fields.items():
            save_single_field(
                os.path.join(outdir, f"primitive_{key}_{name}_{tag}.pdf"),
                fields[key],
                vmin=limits[0],
                vmax=limits[1],
            )
        save_error_comparison_panel(
            os.path.join(outdir, f"error_{key}_compare_{tag}.pdf"),
            ref_fields,
            pred_fields,
            key=key,
        )
        for name, fields in pred_fields.items():
            err = np.abs(fields[key] - ref_fields[key])
            save_single_field(
                os.path.join(outdir, f"error_{key}_{name}_{tag}.pdf"),
                err,
                cmap="turbo",
                vmin=0.0,
                vmax=max(float(np.max(err)), 1e-12),
            )


def iter_test_models(args: argparse.Namespace, device: torch.device):
    model, ckpt = load_required_model(args.ckpt, device, no_compile=args.no_compile)
    yield model_label_from_ckpt(ckpt), model, ckpt
    del model, ckpt
    release_eval_memory(device)

    model_fno, ckpt_fno = load_optional_model(args.ckpt_fno, device, no_compile=args.no_compile, default_path=DEFAULT_FNO_CKPT)
    if model_fno is not None and ckpt_fno is not None:
        yield model_label_from_ckpt(ckpt_fno), model_fno, ckpt_fno


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_HYBRID_CKPT)
    ap.add_argument("--ckpt_fno", type=str, default=DEFAULT_FNO_CKPT)
    ap.add_argument("--eval_mode", type=str, default="both", choices=("rollout", "test", "both"))
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--test_name", type=str, default=DEFAULT_TEST_NAME)
    ap.add_argument("--test_batch", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=0)
    ap.add_argument("--sample_seed", type=int, default=None)
    ap.add_argument("--rollout_steps", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow_tf32", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--outdir", type=str, default="eval_euler2d_pri_dt_out")
    ap.add_argument("--plot_one", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    configure_runtime(device, allow_tf32=args.allow_tf32)
    pin_mem = device.type == "cuda"
    run_rollout = args.eval_mode in ("rollout", "both")
    run_test = args.eval_mode in ("test", "both")

    models: dict[str, tuple[torch.nn.Module, dict]] = {}
    if run_rollout:
        primary_model, primary_ckpt = load_required_model(args.ckpt, device, no_compile=args.no_compile)
        models[model_label_from_ckpt(primary_ckpt)] = (primary_model, primary_ckpt)
        model_fno, ckpt_fno = load_optional_model(args.ckpt_fno, device, no_compile=args.no_compile, default_path=DEFAULT_FNO_CKPT)
        if model_fno is not None and ckpt_fno is not None:
            models[model_label_from_ckpt(ckpt_fno)] = (model_fno, ckpt_fno)
        print("[params] " + ", ".join(f"{name}={count_params(model):,}" for name, (model, _) in models.items()))

    test_path = os.path.join(args.data_dir, args.test_name)
    all_test_states, test_meta = load_trajectory_split(test_path)
    dt = float(test_meta["dt"])
    total_traj = int(all_test_states.shape[0])
    selected_idx = select_trajectory_indices(total_traj, args.n_samples if args.n_samples > 0 else None, args.sample_seed)
    test_states = all_test_states[torch.as_tensor(selected_idx, dtype=torch.long)]
    print(
        f"[data-pri] test_path={test_path}, n_traj={int(test_states.shape[0])}/{total_traj}, "
        f"n_snaps={int(test_states.shape[1])}, dt={dt}, nx={int(test_meta['nx'])}, ny={int(test_meta['ny'])}, "
        f"boundary={test_meta.get('boundary', '?')}, sample_seed={args.sample_seed}"
    )

    if run_rollout:
        n_eval = int(test_states.shape[0])
        max_rollout_steps = int(test_states.shape[1] - 1)
        rollout_steps = max_rollout_steps if args.rollout_steps <= 0 else min(int(args.rollout_steps), max_rollout_steps)
        times = np.arange(rollout_steps + 1, dtype=np.float64) * dt
        rollout_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        one_ref = None
        one_preds = None
        print(f"[rollout-pri] steps={rollout_steps}/{max_rollout_steps}")
        for name, (model, _) in models.items():
            l1 = np.zeros((n_eval, len(times)), dtype=np.float64)
            linf = np.zeros((n_eval, len(times)), dtype=np.float64)
            for idx in range(n_eval):
                ref = test_states[idx, : rollout_steps + 1].cpu().numpy()
                pred = rollout_step_model(
                    model,
                    test_states[idx, 0],
                    n_steps=rollout_steps,
                    dt=dt,
                    device=device,
                    dtype=torch.float32,
                    pin_memory=pin_mem,
                )
                for j in range(len(times)):
                    l1[idx, j] = rel_l1(pred[j], ref[j])
                    linf[idx, j] = rel_linf(pred[j], ref[j])
                if args.plot_one and idx == 0:
                    one_ref = ref
                    one_preds = {} if one_preds is None else one_preds
                    one_preds[name] = pred
            rollout_scores[name] = (l1, linf)
            print(f"[rollout-pri] {name}: final relL1={l1[:, -1].mean():.3e}, final relLinf={linf[:, -1].mean():.3e}")

        npz_payload = {"times": times, "dt": np.array([dt], dtype=np.float64)}
        report_payload = {}
        for name, (l1, linf) in rollout_scores.items():
            npz_payload[f"l1_{name}"] = l1
            npz_payload[f"linf_{name}"] = linf
            report_payload[name] = {"l1": l1, "linf": linf}
        np.savez_compressed(os.path.join(args.outdir, "rollout_metrics.npz"), **npz_payload)
        summary_path, curves_path = save_rollout_reports(args.outdir, times, report_payload)
        print(f"[rollout-pri] saved csv: {summary_path}, {curves_path}")

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
        ax.set_ylabel(r"Primitive relative $L^1$ error", fontsize=14)
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
        ax.set_ylabel(r"Primitive relative $L^\infty$ error", fontsize=14)
        ax.legend()
        apply_line_grid(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "relLinf_vs_time.pdf"), bbox_inches="tight")
        plt.close(fig)

        if args.plot_one and one_ref is not None and one_preds is not None:
            tag = f"{sample_tag(int(selected_idx[0]), args.sample_seed)}_steps{rollout_steps}"
            save_rollout_sample_plots(args.outdir, one_ref, one_preds, tag=tag)

    if run_test:
        if run_rollout:
            models.clear()
            release_eval_memory(device)
        u0_test, u1_test = states_to_one_step_pairs(test_states, dtype=torch.float32)
        print(f"[test-pri] one-step pairs={int(u0_test.shape[0])}")
        test_rows = []
        for name, model, _ in iter_test_models(args, device):
            print(f"[test-pri] evaluating {name}, params={count_params(model):,}")
            test_rows.append(
                evaluate_pairs_streaming(
                    name,
                    model,
                    u0_test,
                    u1_test,
                    dt,
                    device,
                    dtype=torch.float32,
                    batch_size=args.test_batch,
                    pin_memory=pin_mem,
                )
            )
            del model
            release_eval_memory(device)
        test_csv_path = save_test_metrics_csv(args.outdir, test_rows)
        test_summary_csv_path = save_test_metrics_summary_csv(args.outdir, test_rows)
        print(f"[test-pri] saved csv: {test_csv_path}, {test_summary_csv_path}")

    print(f"Saved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
