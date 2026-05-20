"""
Evaluate Euler2D dt-step checkpoints on trajectory test data.
"""

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

from conslaw.checkpoints import compile_if_requested, load_hybrid_dt_step_2d
from conslaw.eval_reports import (
    finalize_dataset_error_row,
    init_dataset_error_stats,
    save_rollout_reports,
    save_test_metrics_csv,
    save_test_metrics_summary_csv,
    update_dataset_error_stats,
)
from conslaw.models import count_params

from common_periodic import (
    DEFAULT_TEST_NAME,
    configure_runtime,
    default_dt_ckpt_path,
    load_trajectory_split,
    rollout_step_model,
    states_to_one_step_pairs,
)


DEFAULT_HYBRID_CKPT = default_dt_ckpt_path("hybrid", "periodic")
DEFAULT_FNO_CKPT = default_dt_ckpt_path("fno", "periodic")
LINE_GRID_ALPHA = 0.3
FIELD_MATH_LABELS = {
    "rho": r"$\rho$",
    "rhou": r"$\rho u$",
    "rhov": r"$\rho v$",
    "E": r"$E$",
    "u": r"$u$",
    "v": r"$v$",
    "p": r"$p$",
}
ERROR_MATH_LABELS = {
    "rho": r"$|\rho-\rho_{\mathrm{ref}}|$",
    "rhou": r"$|\rho u-(\rho u)_{\mathrm{ref}}|$",
    "rhov": r"$|\rho v-(\rho v)_{\mathrm{ref}}|$",
    "E": r"$|E-E_{\mathrm{ref}}|$",
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


def display_model_name(name: str) -> str:
    return {
        "ref": "Ref",
        "hybrid": "Hybrid",
        "fno": "FNO",
    }.get(str(name).lower(), str(name))


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


def load_required_model(
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


def release_eval_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


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
    progress_every: int = 0,
) -> dict[str, object]:
    stats = init_dataset_error_stats()
    model.eval()
    n_total = int(u0.size(0))
    next_progress = int(progress_every) if progress_every and progress_every > 0 else 0
    last_progress = 0

    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        batch = u0[start:stop]
        target_batch = u1[start:stop]
        if pin_memory and device.type == "cuda":
            batch = batch.pin_memory()
        batch = batch.to(device=device, dtype=dtype, non_blocking=pin_memory and device.type == "cuda")
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        dt_b = torch.full((batch.size(0),), float(dt), device=device, dtype=dtype)
        pred = model(batch, dt_b)
        update_dataset_error_stats(stats, pred.detach().cpu().numpy(), target_batch.numpy())
        del pred
        del dt_b
        del batch

        if next_progress > 0:
            done = stop
            while done >= next_progress:
                print(f"[test:{model_name}] {next_progress}/{n_total}", flush=True)
                last_progress = next_progress
                next_progress += int(progress_every)

    if progress_every > 0 and last_progress < n_total:
        print(f"[test:{model_name}] {n_total}/{n_total}", flush=True)
    return finalize_dataset_error_row(model_name, stats)


def iter_test_models(
    args: argparse.Namespace,
    device: torch.device,
):
    primary_model, primary_ckpt, primary_path = load_required_model(args.ckpt, device, no_compile=args.no_compile)
    yield model_label_from_ckpt(primary_ckpt), primary_model, primary_ckpt, primary_path
    primary_model = None
    primary_ckpt = None
    primary_path = None
    release_eval_memory(device)

    if args.hybrid_only:
        return

    model_fno, ckpt_fno, path_fno = load_optional_model(
        args.ckpt_fno,
        device,
        no_compile=args.no_compile,
        default_path=DEFAULT_FNO_CKPT,
    )
    if model_fno is not None and ckpt_fno is not None and path_fno is not None:
        yield model_label_from_ckpt(ckpt_fno), model_fno, ckpt_fno, path_fno
        model_fno = None
        ckpt_fno = None
        path_fno = None
        release_eval_memory(device)


def model_label_from_ckpt(ckpt: dict) -> str:
    kind = str(ckpt.get("kind", ""))
    if "fno" in kind:
        return "fno"
    return "hybrid"


def conserved_to_primitive_2d(state: np.ndarray, gamma: float) -> dict[str, np.ndarray]:
    rho = np.asarray(state[0], dtype=np.float64)
    rho_u = np.asarray(state[1], dtype=np.float64)
    rho_v = np.asarray(state[2], dtype=np.float64)
    energy = np.asarray(state[3], dtype=np.float64)
    u_vel = rho_u / rho
    v_vel = rho_v / rho
    kinetic = 0.5 * (rho_u * rho_u + rho_v * rho_v) / rho
    pressure = (gamma - 1.0) * (energy - kinetic)
    return {
        "rho": rho,
        "u": u_vel,
        "v": v_vel,
        "p": pressure,
    }


def state_to_conserved_fields(state: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "rho": np.asarray(state[0], dtype=np.float64),
        "rhou": np.asarray(state[1], dtype=np.float64),
        "rhov": np.asarray(state[2], dtype=np.float64),
        "E": np.asarray(state[3], dtype=np.float64),
    }


def field_math_label(key: str) -> str:
    return FIELD_MATH_LABELS.get(key, str(key))


def error_math_label(key: str) -> str:
    return ERROR_MATH_LABELS.get(key, field_math_label(key))


def merged_field_limits(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    keys = set(a) & set(b)
    out: dict[str, tuple[float, float]] = {}
    for key in keys:
        lo_a, hi_a = float(np.min(a[key])), float(np.max(a[key]))
        lo_b, hi_b = float(np.min(b[key])), float(np.max(b[key]))
        out[key] = (min(lo_a, lo_b), max(hi_a, hi_b))
    return out


class _OneDecimalScalarFormatter(ScalarFormatter):
    """``ScalarFormatter`` with one decimal place on tick coefficients (ref/hybrid/fno panels share this)."""

    def _set_format(self) -> None:
        self.format = "%1.1f"
        if self._usetex or self._useMathText:
            self.format = r"$\mathdefault{%s}$" % self.format


def _style_field_colorbar(cbar) -> None:
    """Scientific-style colorbar: one decimal on coefficients, ``×10ⁿ`` at axis end; set ``cbar.formatter`` for mpl ≥3.10."""
    fmt = _OneDecimalScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 2))
    cbar.formatter = fmt
    cbar.ax.yaxis.set_major_formatter(fmt)
    cbar.update_ticks()


def save_state_panel(
    out_path: str,
    fields: dict[str, np.ndarray],
    *,
    cmap: str = "turbo",
    limits: dict[str, tuple[float, float]] | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, key in zip(axes.flat, fields.keys()):
        vmin, vmax = (limits[key] if limits is not None and key in limits else (None, None))
        im = ax.imshow(fields[key], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(field_math_label(key))
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _style_field_colorbar(cbar)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_single_field(
    out_path: str,
    field: np.ndarray,
    *,
    label: str,
    cmap: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _style_field_colorbar(cbar)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_error_map(out_path: str, err: np.ndarray, *, key: str) -> None:
    vmax = max(float(err.max()), 1e-12)
    save_single_field(
        out_path,
        err,
        label=error_math_label(key),
        cmap="turbo",
        vmin=0.0,
        vmax=vmax,
    )


def save_density_plots(
    outdir: str,
    ref_traj: np.ndarray,
    pred_trajs: dict[str, np.ndarray],
    *,
    final_time: float,
    gamma: float,
    tag: str,
    share_ref_colorbar: bool = False,
) -> None:
    ref_final = ref_traj[-1]
    final_states = {"ref": ref_final}
    for name, traj in pred_trajs.items():
        final_states[name] = traj[-1]

    conserved_fields = {name: state_to_conserved_fields(state) for name, state in final_states.items()}
    primitive_fields = {name: conserved_to_primitive_2d(state, gamma=gamma) for name, state in final_states.items()}
    conserved_shared_limits = (
        merged_field_limits(conserved_fields["ref"], conserved_fields["hybrid"])
        if share_ref_colorbar and "hybrid" in conserved_fields
        else None
    )
    primitive_shared_limits = (
        merged_field_limits(primitive_fields["ref"], primitive_fields["hybrid"])
        if share_ref_colorbar and "hybrid" in primitive_fields
        else None
    )

    for name, fields in conserved_fields.items():
        use_shared = bool(share_ref_colorbar and name in ("ref", "hybrid") and conserved_shared_limits is not None)
        panel_limits = conserved_shared_limits if use_shared else None
        out_path = os.path.join(outdir, f"conserved_{name}_{tag}.pdf")
        save_state_panel(
            out_path,
            fields,
            limits=panel_limits,
        )
        for key, values in fields.items():
            if use_shared and conserved_shared_limits is not None and key in conserved_shared_limits:
                vmin, vmax = conserved_shared_limits[key]
            else:
                vmin, vmax = None, None
            save_single_field(
                os.path.join(outdir, f"conserved_{key}_{name}_{tag}.pdf"),
                values,
                label=field_math_label(key),
                vmin=vmin,
                vmax=vmax,
            )

    for name, fields in primitive_fields.items():
        use_shared = bool(share_ref_colorbar and name in ("ref", "hybrid") and primitive_shared_limits is not None)
        panel_limits = primitive_shared_limits if use_shared else None
        out_path = os.path.join(outdir, f"primitive_{name}_{tag}.pdf")
        save_state_panel(
            out_path,
            fields,
            limits=panel_limits,
        )
        for key, values in fields.items():
            if use_shared and primitive_shared_limits is not None and key in primitive_shared_limits:
                vmin, vmax = primitive_shared_limits[key]
            else:
                vmin, vmax = None, None
            save_single_field(
                os.path.join(outdir, f"primitive_{key}_{name}_{tag}.pdf"),
                values,
                label=field_math_label(key),
                vmin=vmin,
                vmax=vmax,
            )

    ref_cons = conserved_fields["ref"]
    ref_prim = primitive_fields["ref"]
    for name in pred_trajs:
        for key, values in conserved_fields[name].items():
            err = np.abs(values - ref_cons[key])
            file_key = "cons_rho" if key == "rho" else key
            save_error_map(
                os.path.join(outdir, f"error_{file_key}_{name}_{tag}.pdf"),
                err,
                key=key,
            )
        for key, values in primitive_fields[name].items():
            err = np.abs(values - ref_prim[key])
            save_error_map(
                os.path.join(outdir, f"error_{key}_{name}_{tag}.pdf"),
                err,
                key=key,
            )


def run_rollout_eval(
    args: argparse.Namespace,
    models: dict[str, tuple[torch.nn.Module, dict]],
    test_states: torch.Tensor,
    dt: float,
    gamma: float,
    selected_idx: np.ndarray,
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

    max_rollout_steps = int(test_states.shape[1] - 1)
    rollout_steps = max_rollout_steps if args.rollout_steps <= 0 else min(int(args.rollout_steps), max_rollout_steps)
    test_states = test_states[:, : rollout_steps + 1]
    print(f"[rollout] steps={rollout_steps}/{max_rollout_steps}")

    n_eval = int(test_states.shape[0])
    times = np.arange(int(test_states.shape[1]), dtype=np.float64) * dt
    rollout_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    one_ref = None
    one_preds = None

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
        plot_tag = f"{sample_tag(int(selected_idx[0]), args.sample_seed)}_steps{rollout_steps}"
        plot_models = {k: v for k, v in one_preds.items() if k in ("hybrid", "fno")}
        save_density_plots(
            args.outdir,
            one_ref,
            plot_models,
            final_time=float(times[-1]),
            gamma=gamma,
            tag=plot_tag,
            share_ref_colorbar=args.share_ref_colorbar,
        )


def run_test_eval(
    args: argparse.Namespace,
    test_states: torch.Tensor,
    dt: float,
    device: torch.device,
    *,
    pin_memory: bool,
) -> None:
    u0_test, u1_test = states_to_one_step_pairs(test_states, dtype=torch.float32)
    print(f"[test] one-step pairs={int(u0_test.shape[0])}")
    test_rows = []
    for name, model, ckpt, path in iter_test_models(args, device):
        print(
            f"[test] evaluating {name}, params={count_params(model):,}, "
            f"path={path}, bc={ckpt.get('bc', ckpt.get('args', {}).get('bc', '?'))}"
        )
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
                pin_memory=pin_memory,
                progress_every=100,
            )
        )
        del model
        release_eval_memory(device)
    test_csv_path = save_test_metrics_csv(args.outdir, test_rows)
    test_summary_csv_path = save_test_metrics_summary_csv(args.outdir, test_rows)
    print(f"[test] saved csv: {test_csv_path}, {test_summary_csv_path}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_HYBRID_CKPT)
    ap.add_argument("--ckpt_fno", type=str, default=DEFAULT_FNO_CKPT)
    ap.add_argument("--hybrid_only", action="store_true", help="Evaluate only the primary --ckpt model.")
    ap.add_argument("--eval_mode", type=str, default="both", choices=("rollout", "test", "both"))
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--test_name", type=str, default=DEFAULT_TEST_NAME)
    ap.add_argument("--test_path", type=str, default=None)
    ap.add_argument("--test_batch", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=0, help="0 means use all test trajectories for rollout.")
    ap.add_argument("--sample_seed", type=int, default=None, help="Random seed used to choose evaluation trajectories.")
    ap.add_argument("--T", type=float, default=None, help="Optional rollout horizon; truncate test trajectories to t<=T.")
    ap.add_argument(
        "--rollout_steps",
        type=int,
        default=0,
        help="Number of rollout steps to evaluate. 0 means use the full available trajectory after any --T truncation.",
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow_tf32", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--outdir", type=str, default="eval_euler2d_periodic_dt_out")
    ap.add_argument("--plot_one", action="store_true")
    ap.add_argument(
        "--share_ref_colorbar",
        action="store_true",
        help="Use union(ref, hybrid) vmin/vmax for ref+hybrid conserved & primitive plots; fno uses per-field auto scale.",
    )
    ap.add_argument("--plot_channel", type=int, default=0, help="Conserved channel to visualize, 0=rho.")
    ap.add_argument("--yslices", type=float, nargs="+", default=[0.5], help="y-slices for x-t plots in [0, 1].")
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    configure_runtime(device, allow_tf32=args.allow_tf32)
    pin_mem = device.type == "cuda"
    run_rollout = args.eval_mode in ("rollout", "both")
    run_test = args.eval_mode in ("test", "both")

    models: dict[str, tuple[torch.nn.Module, dict]] = {}
    model_paths: dict[str, str] = {}
    model_param_counts: dict[str, int] = {}

    if run_rollout:
        primary_model, primary_ckpt, primary_path = load_model_from_ckpt(
            args.ckpt,
            device,
            no_compile=args.no_compile,
        )
        primary_name = model_label_from_ckpt(primary_ckpt)
        models[primary_name] = (primary_model, primary_ckpt)
        model_paths[primary_name] = primary_path
        model_param_counts[primary_name] = count_params(primary_model)

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

    print(f"[params] hybrid={model_param_counts.get('hybrid', 'unavailable'):,}" if "hybrid" in model_param_counts else "[params] hybrid=unavailable")
    print(f"[params] fno={model_param_counts.get('fno', 'unavailable'):,}" if "fno" in model_param_counts else "[params] fno=unavailable")

    test_path = resolve_test_split_path(args)
    test_states, test_meta = load_trajectory_split(test_path)
    dt = float(test_meta["dt"])
    gamma = float(test_meta.get("gamma", 1.4))
    total_traj = int(test_states.shape[0])
    selected_idx = select_trajectory_indices(
        total_traj,
        args.n_samples if args.n_samples > 0 else None,
        args.sample_seed,
    )
    test_states = test_states[torch.as_tensor(selected_idx, dtype=torch.long)]
    print(
        f"[data] test_path={test_path}, n_traj={int(test_states.shape[0])}/{total_traj}, "
        f"n_snaps={int(test_states.shape[1])}, dt={dt}, "
        f"nx={int(test_meta['nx'])}, ny={int(test_meta['ny'])}, boundary={test_meta.get('boundary', '?')}, "
        f"sample_seed={args.sample_seed}"
    )
    print(
        "[models] "
        + (
            ", ".join(
                f"{name}={model_paths[name]} (bc={models[name][1].get('bc', models[name][1].get('args', {}).get('bc', '?'))})"
                for name in sorted(models.keys())
            )
            if run_rollout
            else "test mode loads models sequentially"
        )
    )
    if args.plot_one and selected_idx.size > 0:
        print(f"[plot_one] selected trajectory index={int(selected_idx[0])}")

    if run_rollout:
        run_rollout_eval(args, models, test_states, dt, gamma, selected_idx, device, pin_memory=pin_mem)

    if run_test:
        if run_rollout:
            models.clear()
            primary_model = None
            primary_ckpt = None
            primary_path = None
            model_fno = None
            ckpt_fno = None
            path_fno = None
            release_eval_memory(device)
        run_test_eval(args, test_states, dt, device, pin_memory=pin_mem)

    print(f"Saved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
