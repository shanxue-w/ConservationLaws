"""Evaluate Euler2D primitive-variable dt-step checkpoints."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

# ``dataset.data`` imports the external WENO backend with Numba cache enabled.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join("/tmp", "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

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
    conserved_to_primitive_numpy,
    load_primitive_dt_step_2d,
    load_trajectory_split,
    primitive_to_conserved_numpy,
    rollout_step_model,
    states_to_one_step_pairs,
)
from dataset.data import (
    DEFAULT_P_RANGE,
    DEFAULT_RHO_RANGE,
    DEFAULT_UV_RANGE,
    build_euler2d_solver,
    downsample_conserved2d,
)


DEFAULT_HYBRID_CKPT = "checkpoints/euler2d_hybrid_pri_outflow_1e-2_dt.pt"
DEFAULT_FNO_CKPT = "checkpoints/euler2d_fno_pri_dt_24.pt"
LINE_GRID_ALPHA = 0.3
DEMO_CONFIGS = {
    "riemann_01": {
        "description": (
            "Configuration 2. Primitive states stored as rho,u,v,p; "
            "state indices map as 1=UR, 2=UL, 3=LL, 4=LR."
        ),
        "split_x": 0.5,
        "split_y": 0.5,
        "states": {
            "ul": (0.5197, -0.7259, 0.0, 0.4),
            "ur": (1.0, 0.0, 0.0, 1.0),
            "ll": (1.0, -0.7259, -0.7259, 1.0),
            "lr": (0.5197, 0.0, -0.7259, 0.4),
        },
    },
    "riemann_02": {
        "description": (
            "Configuration 7. Primitive states stored as rho,u,v,p; "
            "state indices map as 1=UR, 2=UL, 3=LL, 4=LR."
        ),
        "split_x": 0.5,
        "split_y": 0.5,
        "states": {
            "ul": (0.5197, -0.6259, 0.1, 0.4),
            "ur": (1.0, 0.1, 0.1, 1.0),
            "ll": (0.8, 0.1, 0.1, 0.4),
            "lr": (0.5197, 0.1, -0.6259, 0.4),
        },
    },
    "riemann_03": {
        "description": (
            "Configuration 8. Primitive states stored as rho,u,v,p; "
            "state indices map as 1=UR, 2=UL, 3=LL, 4=LR."
        ),
        "split_x": 0.5,
        "split_y": 0.5,
        "states": {
            "ul": (1.0, -0.6259, 0.1, 1.0),
            "ur": (0.5197, 0.1, 0.1, 0.4),
            "ll": (0.8, 0.1, 0.1, 1.0),
            "lr": (1.0, 0.1, -0.6259, 1.0),
        },
    },
    "riemann_04": {
        "description": (
            "Configuration 12. Primitive states stored as rho,u,v,p; "
            "state indices map as 1=UR, 2=UL, 3=LL, 4=LR."
        ),
        "split_x": 0.5,
        "split_y": 0.5,
        "states": {
            "ul": (1.0, 0.7276, 0.0, 1.0),
            "ur": (0.5313, 0.0, 0.0, 0.4),
            "ll": (0.8, 0.0, 0.0, 1.0),
            "lr": (1.0, 0.0, 0.7276, 1.0),
        },
    },
}
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
        "hybrid": "LGNO",
        "fno": "FNO",
        "cnn": "CNN",
        "weno256": "WENO256",
        "weno512": "WENO512",
    }.get(str(name).lower(), str(name))


def field_math_label(key: str) -> str:
    return FIELD_MATH_LABELS.get(key, str(key))


def error_math_label(key: str) -> str:
    return ERROR_MATH_LABELS.get(key, field_math_label(key))


def release_eval_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def release_loaded_rollout_models(models: dict[str, tuple[torch.nn.Module, dict]], device: torch.device) -> None:
    models.clear()
    release_eval_memory(device)


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


def field_limits(fields: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    return {key: (float(np.min(values)), float(np.max(values))) for key, values in fields.items()}


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
    shared_color_scale: bool = True,
) -> None:
    names = list(pred_fields_by_model.keys())
    if not names:
        return
    ncols = len(names)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.8), squeeze=False)
    errors = {name: np.abs(fields[key] - ref_fields[key]) for name, fields in pred_fields_by_model.items()}
    shared_vmax = max(max(float(np.max(err)), 1e-12) for err in errors.values()) if shared_color_scale else None
    for ax, name in zip(axes[0], names):
        vmax = shared_vmax if shared_vmax is not None else max(float(np.max(errors[name])), 1e-12)
        im = ax.imshow(errors[name], origin="lower", cmap="turbo", vmin=0.0, vmax=vmax)
        ax.set_title(f"{display_model_name(name)} {error_math_label(key)}")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _style_field_colorbar(cbar)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_contour_comparison_panel(
    out_path: str,
    fields_by_model: dict[str, dict[str, np.ndarray]],
    *,
    key: str,
    levels: int = 18,
) -> None:
    names = list(fields_by_model.keys())
    ncols = len(names)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.8), squeeze=False)
    vmin, vmax = merged_field_limit(fields_by_model, key)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        plt.close(fig)
        return
    if abs(vmax - vmin) < 1e-14:
        span = max(abs(vmin), 1.0) * 1e-6
        vmin -= span
        vmax += span
    contour_levels = np.linspace(vmin, vmax, int(levels))
    for ax, name in zip(axes[0], names):
        field = np.asarray(fields_by_model[name][key], dtype=np.float64)
        ax.contour(field, levels=contour_levels, colors="black", linewidths=0.55)
        ax.set_aspect("equal")
        ax.set_title(f"{display_model_name(name)} {field_math_label(key)}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_single_contour(
    out_path: str,
    field: np.ndarray,
    *,
    limits: tuple[float, float],
    levels: int = 18,
) -> None:
    vmin, vmax = float(limits[0]), float(limits[1])
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return
    if abs(vmax - vmin) < 1e-14:
        span = max(abs(vmin), 1.0) * 1e-6
        vmin -= span
        vmax += span
    contour_levels = np.linspace(vmin, vmax, int(levels))
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.contour(np.asarray(field, dtype=np.float64), levels=contour_levels, colors="black", linewidths=0.55)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_rollout_sample_plots(
    outdir: str,
    ref_traj: np.ndarray,
    pred_trajs: dict[str, np.ndarray],
    *,
    tag: str,
    shared_color_scale: bool = False,
    share_ref_colorbar: bool = False,
) -> None:
    final_fields = {"ref": primitive_fields(ref_traj[-1])}
    for name, traj in pred_trajs.items():
        final_fields[name] = primitive_fields(traj[-1])

    ref_fields = final_fields["ref"]
    pred_fields = {name: fields for name, fields in final_fields.items() if name != "ref"}
    ref_color_limits = field_limits(ref_fields) if share_ref_colorbar else None
    for key in PRIMITIVE_CHANNELS:
        contour_limits = merged_field_limit(final_fields, key)
        if ref_color_limits is not None:
            color_limits = ref_color_limits[key]
        elif shared_color_scale:
            color_limits = contour_limits
        else:
            color_limits = None
        error_vmax = None
        if shared_color_scale and pred_fields:
            error_vmax = max(
                max(float(np.max(np.abs(fields[key] - ref_fields[key]))), 1e-12)
                for fields in pred_fields.values()
            )
        save_field_comparison_panel(
            os.path.join(outdir, f"primitive_{key}_compare_{tag}.pdf"),
            final_fields,
            key=key,
            limits=color_limits,
        )
        save_contour_comparison_panel(
            os.path.join(outdir, f"contour_{key}_compare_{tag}.pdf"),
            final_fields,
            key=key,
        )
        for name, fields in final_fields.items():
            save_single_contour(
                os.path.join(outdir, f"contour_{key}_{name}_{tag}.pdf"),
                fields[key],
                limits=contour_limits,
            )
        for name, fields in final_fields.items():
            save_single_field(
                os.path.join(outdir, f"primitive_{key}_{name}_{tag}.pdf"),
                fields[key],
                vmin=color_limits[0] if color_limits is not None else None,
                vmax=color_limits[1] if color_limits is not None else None,
            )
        save_error_comparison_panel(
            os.path.join(outdir, f"error_{key}_compare_{tag}.pdf"),
            ref_fields,
            pred_fields,
            key=key,
            shared_color_scale=shared_color_scale,
        )
        for name, fields in pred_fields.items():
            err = np.abs(fields[key] - ref_fields[key])
            save_single_field(
                os.path.join(outdir, f"error_{key}_{name}_{tag}.pdf"),
                err,
                cmap="turbo",
                vmin=0.0,
                vmax=error_vmax if error_vmax is not None else max(float(np.max(err)), 1e-12),
            )


def iter_test_models(args: argparse.Namespace, device: torch.device):
    model, ckpt = load_required_model(args.ckpt, device, no_compile=args.no_compile)
    yield model_label_from_ckpt(ckpt), model, ckpt
    del model, ckpt
    release_eval_memory(device)

    model_fno, ckpt_fno = load_optional_model(args.ckpt_fno, device, no_compile=args.no_compile, default_path=DEFAULT_FNO_CKPT)
    if model_fno is not None and ckpt_fno is not None:
        yield model_label_from_ckpt(ckpt_fno), model_fno, ckpt_fno


def demo_state_override(value: list[float] | None, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if value is None:
        return fallback
    if len(value) != 4:
        raise ValueError("Each demo state override must have four values: rho u v p.")
    return tuple(float(v) for v in value)


def demo_states_from_args(args: argparse.Namespace) -> dict[str, tuple[float, float, float, float]]:
    config = DEMO_CONFIGS[str(args.demo)]
    states = config["states"]
    return {
        "ul": demo_state_override(args.demo_state_ul, states["ul"]),
        "ur": demo_state_override(args.demo_state_ur, states["ur"]),
        "ll": demo_state_override(args.demo_state_ll, states["ll"]),
        "lr": demo_state_override(args.demo_state_lr, states["lr"]),
    }


def fit_demo_rhop_scale(
    states: dict[str, tuple[float, float, float, float]],
    rho_range: tuple[float, float] = DEFAULT_RHO_RANGE,
    p_range: tuple[float, float] = DEFAULT_P_RANGE,
    lower_margin: float = 1.1,
    upper_margin: float = 0.9,
) -> float:
    values = np.asarray(list(states.values()), dtype=np.float64)
    rho_min = float(np.min(values[:, 0]))
    rho_max = float(np.max(values[:, 0]))
    p_min = float(np.min(values[:, 3]))
    p_max = float(np.max(values[:, 3]))
    if min(rho_min, rho_max, p_min, p_max) <= 0.0:
        raise ValueError("Demo scaling requires positive rho and p in every quadrant.")
    if lower_margin <= 0.0 or upper_margin <= 0.0:
        raise ValueError("Demo scale margins must be positive.")

    rho_lo = float(rho_range[0]) * float(lower_margin)
    p_lo = float(p_range[0]) * float(lower_margin)
    rho_hi = float(rho_range[1]) * float(upper_margin)
    p_hi = float(p_range[1]) * float(upper_margin)
    lower = max(rho_lo / rho_min, p_lo / p_min)
    upper = min(rho_hi / rho_max, p_hi / p_max)
    if lower <= upper:
        return float(min(max(1.0, lower), upper))
    # If one scalar cannot satisfy all bounds, balance the multiplicative violations.
    return float(np.sqrt(lower * upper))


def _inner_positive_range(
    value_range: tuple[float, float],
    *,
    lower_margin: float,
    upper_margin: float,
) -> tuple[float, float]:
    lo = float(value_range[0]) * float(lower_margin)
    hi = float(value_range[1]) * float(upper_margin)
    if lo <= 0.0 or hi <= 0.0 or lo >= hi:
        raise ValueError(f"Invalid positive inner range: {(lo, hi)}.")
    return lo, hi


def _scaled_state_violation(
    states: dict[str, tuple[float, float, float, float]],
    *,
    rhop_scale: float,
    velocity_scale: int,
    rho_range: tuple[float, float],
    p_range: tuple[float, float],
    uv_range: tuple[float, float],
    lower_margin: float,
    upper_margin: float,
) -> float:
    values = np.asarray(list(states.values()), dtype=np.float64)
    k = float(velocity_scale)
    rho_s = values[:, 0] * float(rhop_scale)
    u_s = values[:, 1] / k
    v_s = values[:, 2] / k
    p_s = values[:, 3] * float(rhop_scale) / (k * k)
    rho_lo, rho_hi = _inner_positive_range(rho_range, lower_margin=lower_margin, upper_margin=upper_margin)
    p_lo, p_hi = _inner_positive_range(p_range, lower_margin=lower_margin, upper_margin=upper_margin)
    uv_abs_hi = max(abs(float(uv_range[0])), abs(float(uv_range[1]))) * float(upper_margin)
    if uv_abs_hi <= 0.0:
        raise ValueError("Velocity range must have a positive nonzero extent.")
    violations = [
        rho_lo / max(float(np.min(rho_s)), 1e-300),
        float(np.max(rho_s)) / rho_hi,
        p_lo / max(float(np.min(p_s)), 1e-300),
        float(np.max(p_s)) / p_hi,
        max(float(np.max(np.abs(u_s))), float(np.max(np.abs(v_s)))) / uv_abs_hi,
        1.0,
    ]
    return float(max(violations))


def fit_demo_model_scales(
    states: dict[str, tuple[float, float, float, float]],
    *,
    requested_velocity_scale: int = 0,
    max_velocity_scale: int = 4,
    rho_range: tuple[float, float] = DEFAULT_RHO_RANGE,
    p_range: tuple[float, float] = DEFAULT_P_RANGE,
    uv_range: tuple[float, float] = DEFAULT_UV_RANGE,
    lower_margin: float = 1.1,
    upper_margin: float = 0.9,
) -> tuple[float, int]:
    values = np.asarray(list(states.values()), dtype=np.float64)
    rho_min = float(np.min(values[:, 0]))
    rho_max = float(np.max(values[:, 0]))
    p_min = float(np.min(values[:, 3]))
    p_max = float(np.max(values[:, 3]))
    if min(rho_min, rho_max, p_min, p_max) <= 0.0:
        raise ValueError("Demo scaling requires positive rho and p in every quadrant.")
    if requested_velocity_scale < 0:
        raise ValueError("--demo_model_velocity_scale must be nonnegative.")
    if max_velocity_scale < 1:
        raise ValueError("--demo_model_velocity_scale_max must be at least 1.")

    rho_lo, rho_hi = _inner_positive_range(rho_range, lower_margin=lower_margin, upper_margin=upper_margin)
    p_lo, p_hi = _inner_positive_range(p_range, lower_margin=lower_margin, upper_margin=upper_margin)
    candidates = [int(requested_velocity_scale)] if requested_velocity_scale > 0 else list(range(1, int(max_velocity_scale) + 1))
    best: tuple[float, int, float] | None = None
    for k_int in candidates:
        if k_int < 1:
            raise ValueError("Velocity scale candidates must be positive integers.")
        k2 = float(k_int * k_int)
        lower = max(rho_lo / rho_min, p_lo * k2 / p_min)
        upper = min(rho_hi / rho_max, p_hi * k2 / p_max)
        if lower <= upper:
            rhop_scale = float(min(max(1.0, lower), upper))
        else:
            rhop_scale = float(np.sqrt(lower * upper))
        violation = _scaled_state_violation(
            states,
            rhop_scale=rhop_scale,
            velocity_scale=k_int,
            rho_range=rho_range,
            p_range=p_range,
            uv_range=uv_range,
            lower_margin=lower_margin,
            upper_margin=upper_margin,
        )
        candidate = (violation, k_int, rhop_scale)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    _, velocity_scale, rhop_scale = best
    return float(rhop_scale), int(velocity_scale)


def scale_demo_states_rhop(
    states: dict[str, tuple[float, float, float, float]],
    scale: float,
) -> dict[str, tuple[float, float, float, float]]:
    factor = float(scale)
    return {
        key: (float(rho) * factor, float(u), float(v), float(p) * factor)
        for key, (rho, u, v, p) in states.items()
    }


def scale_demo_states_velocity(
    states: dict[str, tuple[float, float, float, float]],
    scale: int,
    *,
    inverse: bool = False,
) -> dict[str, tuple[float, float, float, float]]:
    factor = float(scale)
    if factor <= 0.0:
        raise ValueError("Velocity scale must be positive.")
    if inverse:
        return {
            key: (float(rho), float(u) * factor, float(v) * factor, float(p) * factor * factor)
            for key, (rho, u, v, p) in states.items()
        }
    return {
        key: (float(rho), float(u) / factor, float(v) / factor, float(p) / (factor * factor))
        for key, (rho, u, v, p) in states.items()
    }


def scale_primitive_rhop(prim: np.ndarray, scale: float, *, inverse: bool = False) -> np.ndarray:
    factor = 1.0 / float(scale) if inverse else float(scale)
    out = np.asarray(prim).copy()
    if out.ndim >= 4 and out.shape[1] == 4:
        out[:, 0] *= factor
        out[:, 3] *= factor
        return out
    if out.shape[0] == 4:
        out[0] *= factor
        out[3] *= factor
        return out
    if out.shape[-1] == 4:
        out[..., 0] *= factor
        out[..., 3] *= factor
        return out
    raise ValueError(f"Expected primitive state with a 4-channel axis, got shape {out.shape}.")


def scale_primitive_velocity(prim: np.ndarray, scale: int, *, inverse: bool = False) -> np.ndarray:
    factor = float(scale)
    if factor <= 0.0:
        raise ValueError("Velocity scale must be positive.")
    vel_factor = factor if inverse else 1.0 / factor
    p_factor = factor * factor if inverse else 1.0 / (factor * factor)
    out = np.asarray(prim).copy()
    if out.ndim >= 4 and out.shape[1] == 4:
        out[:, 1] *= vel_factor
        out[:, 2] *= vel_factor
        out[:, 3] *= p_factor
        return out
    if out.shape[0] == 4:
        out[1] *= vel_factor
        out[2] *= vel_factor
        out[3] *= p_factor
        return out
    if out.shape[-1] == 4:
        out[..., 1] *= vel_factor
        out[..., 2] *= vel_factor
        out[..., 3] *= p_factor
        return out
    raise ValueError(f"Expected primitive state with a 4-channel axis, got shape {out.shape}.")


def primitive_quadrant_state(
    nx: int,
    ny: int,
    *,
    split_x: float,
    split_y: float,
    states: dict[str, tuple[float, float, float, float]],
) -> np.ndarray:
    x = (np.arange(int(nx), dtype=np.float64) + 0.5) / float(nx)
    y = (np.arange(int(ny), dtype=np.float64) + 0.5) / float(ny)
    xx, yy = np.meshgrid(x, y)
    prim = np.empty((4, int(ny), int(nx)), dtype=np.float64)
    masks = {
        "ul": (xx < split_x) & (yy > split_y),
        "ur": (xx >= split_x) & (yy > split_y),
        "ll": (xx < split_x) & (yy <= split_y),
        "lr": (xx >= split_x) & (yy <= split_y),
    }
    for key, mask in masks.items():
        values = np.asarray(states[key], dtype=np.float64)
        for channel in range(4):
            prim[channel, mask] = values[channel]
    return prim


def primitive_traj_from_conserved(traj: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    return np.stack(
        [conserved_to_primitive_numpy(np.asarray(state, dtype=np.float64), gamma=gamma) for state in traj],
        axis=0,
    )


def rollout_weno256_from_primitive(
    op,
    u0_prim: torch.Tensor | np.ndarray,
    *,
    dx: float,
    dy: float,
    dt: float,
    n_steps: int,
    cfl: float,
    gamma: float,
    log_ic_state: bool = True,
) -> np.ndarray:
    if torch.is_tensor(u0_prim):
        u0_prim_np = u0_prim.detach().cpu().numpy()
    else:
        u0_prim_np = np.asarray(u0_prim)
    u0_cons = primitive_to_conserved_numpy(np.asarray(u0_prim_np, dtype=np.float64), gamma=gamma)
    cons_traj = op.solve_snapshots(
        u0_cons,
        float(dx),
        float(dy),
        dt_snap=float(dt),
        n_snaps=int(n_steps) + 1,
        cfl=float(cfl),
        log_ic_state=bool(log_ic_state),
    )
    return primitive_traj_from_conserved(cons_traj, gamma=gamma)


def downsample_conserved_traj(traj: np.ndarray, factor_y: int, factor_x: int | None = None) -> np.ndarray:
    return np.stack(
        [downsample_conserved2d(np.asarray(state, dtype=np.float64), factor_y, factor_x) for state in traj],
        axis=0,
    )


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warmup_weno_demo_rollout(
    name: str,
    op,
    u0_cons: np.ndarray,
    *,
    dx: float,
    dy: float,
    dt_snap: float,
    n_steps: int,
    cfl: float,
) -> None:
    if n_steps <= 0:
        return
    print(f"[warmup] {name}: {n_steps} CFL substep(s), not timed", flush=True)
    u = np.asarray(u0_cons, dtype=np.float64).copy()
    old_verbose = getattr(op, "verbose", None)
    if old_verbose is not None:
        op.verbose = False
    try:
        for _ in range(n_steps):
            dt_sub = op._compute_dt(u, dx, dy, cfl=cfl, t_remaining=dt_snap)
            if dt_sub <= 0.0:
                raise RuntimeError("dt_sub <= 0 in WENO warmup")
            u = op.step(u, dx, dy, dt_sub)
    finally:
        if old_verbose is not None:
            op.verbose = old_verbose


def timed_weno_demo_rollout(
    name: str,
    op,
    u0_cons: np.ndarray,
    *,
    dx: float,
    dy: float,
    dt_snap: float,
    n_steps: int,
    cfl: float,
    warmup_steps: int = 0,
) -> tuple[np.ndarray, float]:
    warmup_weno_demo_rollout(
        name,
        op,
        u0_cons,
        dx=dx,
        dy=dy,
        dt_snap=dt_snap,
        n_steps=warmup_steps,
        cfl=cfl,
    )
    t0 = time.perf_counter()
    traj = op.solve_snapshots(
        u0_cons,
        dx,
        dy,
        dt_snap=dt_snap,
        n_snaps=n_steps + 1,
        cfl=cfl,
        log_ic_state=False,
    )
    elapsed = time.perf_counter() - t0
    print(f"[demo-pri] {name}: {elapsed:.6f} s")
    return traj.astype(np.float64, copy=False), elapsed


def warmup_model_demo_rollout(
    name: str,
    model: torch.nn.Module,
    u0_prim: torch.Tensor,
    *,
    n_steps: int,
    dt: float,
    device: torch.device,
    pin_memory: bool,
) -> None:
    if n_steps <= 0:
        return
    print(f"[warmup] {name}: {n_steps} step(s), not timed", flush=True)
    model.eval()
    with torch.inference_mode():
        _ = rollout_step_model(
            model,
            u0_prim,
            n_steps=n_steps,
            dt=dt,
            device=device,
            dtype=torch.float32,
            pin_memory=pin_memory,
        )
    synchronize_if_needed(device)


def timed_model_demo_rollout(
    name: str,
    model: torch.nn.Module,
    u0_prim: torch.Tensor,
    *,
    n_steps: int,
    dt: float,
    device: torch.device,
    pin_memory: bool,
    warmup_steps: int = 0,
) -> tuple[np.ndarray, float]:
    warmup_model_demo_rollout(
        name,
        model,
        u0_prim,
        n_steps=warmup_steps,
        dt=dt,
        device=device,
        pin_memory=pin_memory,
    )
    synchronize_if_needed(device)
    model.eval()
    t0 = time.perf_counter()
    with torch.inference_mode():
        pred = rollout_step_model(
            model,
            u0_prim,
            n_steps=n_steps,
            dt=dt,
            device=device,
            dtype=torch.float32,
            pin_memory=pin_memory,
        )
    synchronize_if_needed(device)
    elapsed = time.perf_counter() - t0
    print(f"[demo-pri] {name}: {elapsed:.6f} s")
    return pred, elapsed


def run_demo_eval(
    args: argparse.Namespace,
    models: dict[str, tuple[torch.nn.Module, dict]],
    device: torch.device,
    *,
    pin_memory: bool,
) -> None:
    gamma = 1.4
    nx = int(args.demo_nx)
    ny = int(args.demo_ny)
    fine_nx = int(args.demo_weno_fine_nx)
    fine_ny = int(args.demo_weno_fine_ny)
    if nx <= 0 or ny <= 0 or fine_nx <= 0 or fine_ny <= 0:
        raise ValueError("Demo grid sizes must be positive.")
    if nx != 256 or ny != 256:
        raise ValueError("Hybrid/FNO demo rollout is fixed to the 256x256 model grid.")
    if fine_nx != 512 or fine_ny != 512:
        raise ValueError("This demo compares WENO-Z 256x256 with WENO-Z 512x512 reference.")
    if fine_nx % nx != 0 or fine_ny % ny != 0:
        raise ValueError("Fine WENO grid must be an integer multiple of the model grid.")
    frame_dt = float(args.demo_frame_dt)
    if frame_dt <= 0.0:
        raise ValueError("--demo_frame_dt must be positive.")
    rollout_steps = int(args.rollout_steps) if args.rollout_steps > 0 else int(round(float(args.demo_T) / frame_dt))
    if rollout_steps < 1:
        raise ValueError("Demo needs at least one rollout step.")

    config = DEMO_CONFIGS[str(args.demo)]
    split_x = float(args.demo_split_x if args.demo_split_x is not None else config["split_x"])
    split_y = float(args.demo_split_y if args.demo_split_y is not None else config["split_y"])
    raw_states = demo_states_from_args(args)
    if bool(getattr(args, "no_demo_scale", False)):
        rhop_scale = 1.0
        model_velocity_scale = 1
    else:
        rhop_scale, model_velocity_scale = fit_demo_model_scales(
            raw_states,
            requested_velocity_scale=int(getattr(args, "demo_model_velocity_scale", 0)),
            max_velocity_scale=int(getattr(args, "demo_model_velocity_scale_max", 4)),
        )
    states = scale_demo_states_rhop(raw_states, rhop_scale)
    model_states = scale_demo_states_velocity(states, model_velocity_scale)
    u0_prim_np = primitive_quadrant_state(nx, ny, split_x=split_x, split_y=split_y, states=states)
    u0_model_prim_np = primitive_quadrant_state(nx, ny, split_x=split_x, split_y=split_y, states=model_states)
    u0_fine_prim_np = primitive_quadrant_state(fine_nx, fine_ny, split_x=split_x, split_y=split_y, states=states)
    u0_cons = primitive_to_conserved_numpy(u0_prim_np, gamma=gamma)
    u0_fine_cons = primitive_to_conserved_numpy(u0_fine_prim_np, gamma=gamma)
    u0_prim = torch.from_numpy(u0_model_prim_np.astype(np.float32))
    dx, dy = 1.0 / float(nx), 1.0 / float(ny)
    fine_dx, fine_dy = 1.0 / float(fine_nx), 1.0 / float(fine_ny)
    final_time = float(rollout_steps) * frame_dt
    times = np.asarray([final_time], dtype=np.float64)
    warmup_steps = 0 if args.skip_warmup else min(max(int(args.warmup_steps), 0), rollout_steps)
    model_rollout_steps = rollout_steps * int(model_velocity_scale)
    model_warmup_steps = warmup_steps * int(model_velocity_scale)

    reconstruction = str(args.demo_weno_reconstruction)
    weno_verbose = not bool(args.demo_weno_quiet)
    print(
        f"[demo-pri] demo={args.demo}, model_grid={nx}x{ny}, weno_fine_grid={fine_nx}x{fine_ny}, "
        f"split=({split_x:g},{split_y:g}), steps={rollout_steps}, dt={frame_dt:g}, final_t={final_time:g}, "
        f"cfl={args.demo_weno_cfl:g}, recon={reconstruction}, verbose={weno_verbose}"
    )
    if warmup_steps > 0:
        print(
            f"[warmup] weno uses {warmup_steps} CFL substep(s); "
            f"Hybrid/FNO use {model_warmup_steps} internal step(s), immediately before timed rollout",
            flush=True,
        )
    print(f"[demo-pri] raw_states={raw_states}")
    print(
        f"[demo-pri] rhop_scale={rhop_scale:.8g} "
        f"(scaled rho,p are used for rollout; outputs are unscaled before reporting)"
    )
    print(
        f"[demo-pri] model_velocity_scale={model_velocity_scale:d} "
        f"(Hybrid/FNO use u,v/k,p/k^2 and run {model_rollout_steps} internal step(s); "
        f"only the final frame is unscaled and reported)"
    )
    print(f"[demo-pri] scaled_states={states}")
    print(f"[demo-pri] model_scaled_states={model_states}")

    op = build_euler2d_solver(
        bc="outflow",
        reconstruction=reconstruction,
        verbose=weno_verbose,
        line_batch_size=args.demo_weno_line_batch_size,
    )
    op_fine = build_euler2d_solver(
        bc="outflow",
        reconstruction=reconstruction,
        verbose=weno_verbose,
        line_batch_size=args.demo_weno_line_batch_size,
    )

    runtime_rows: list[tuple[str, float]] = []
    weno256_cons, weno256_elapsed = timed_weno_demo_rollout(
        "weno256",
        op,
        u0_cons,
        dx=dx,
        dy=dy,
        dt_snap=final_time,
        n_steps=1,
        cfl=float(args.demo_weno_cfl),
        warmup_steps=warmup_steps,
    )
    runtime_rows.append(("weno256", weno256_elapsed))
    weno512_cons, weno512_elapsed = timed_weno_demo_rollout(
        "weno512",
        op_fine,
        u0_fine_cons,
        dx=fine_dx,
        dy=fine_dy,
        dt_snap=final_time,
        n_steps=1,
        cfl=float(args.demo_weno_cfl),
        warmup_steps=warmup_steps,
    )
    runtime_rows.append(("weno512", weno512_elapsed))
    factor_y = fine_ny // ny
    factor_x = fine_nx // nx
    ref_traj_scaled = primitive_traj_from_conserved(downsample_conserved_traj(weno512_cons, factor_y, factor_x), gamma=gamma)
    weno256_prim_scaled = primitive_traj_from_conserved(weno256_cons, gamma=gamma)
    ref_traj = scale_primitive_rhop(ref_traj_scaled[-1:], rhop_scale, inverse=True)
    weno256_prim = scale_primitive_rhop(weno256_prim_scaled[-1:], rhop_scale, inverse=True)

    rollout_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: (np.zeros((1, len(times)), dtype=np.float64), np.zeros((1, len(times)), dtype=np.float64))
        for name in ("weno256", *models.keys())
    }
    pred_trajs: dict[str, np.ndarray] = {"weno256": weno256_prim}
    for j in range(len(times)):
        rollout_scores["weno256"][0][0, j] = rel_l1(weno256_prim[j], ref_traj[j])
        rollout_scores["weno256"][1][0, j] = rel_linf(weno256_prim[j], ref_traj[j])

    for name, (model, _) in models.items():
        pred, elapsed = timed_model_demo_rollout(
            name,
            model,
            u0_prim,
            n_steps=model_rollout_steps,
            dt=frame_dt,
            device=device,
            pin_memory=pin_memory,
            warmup_steps=model_warmup_steps,
        )
        runtime_rows.append((name, elapsed))
        pred_sampled = pred[-1:]
        pred_unscaled = scale_primitive_rhop(
            scale_primitive_velocity(pred_sampled, model_velocity_scale, inverse=True),
            rhop_scale,
            inverse=True,
        )
        pred_trajs[name] = pred_unscaled
        l1, linf = rollout_scores[name]
        for j in range(len(times)):
            l1[0, j] = rel_l1(pred_unscaled[j], ref_traj[j])
            linf[0, j] = rel_linf(pred_unscaled[j], ref_traj[j])

    for name, (l1, linf) in rollout_scores.items():
        print(f"[demo-pri] {name}: final relL1={l1[0, -1]:.3e}, final relLinf={linf[0, -1]:.3e}")

    npz_payload = {
        "times": times,
        "dt": np.array([frame_dt], dtype=np.float64),
        "runtime_names": np.asarray([name for name, _ in runtime_rows], dtype="<U32"),
        "runtime_seconds": np.asarray([elapsed for _, elapsed in runtime_rows], dtype=np.float64),
        "rhop_scale": np.array([rhop_scale], dtype=np.float64),
        "model_velocity_scale": np.array([model_velocity_scale], dtype=np.int64),
        "model_internal_steps": np.array([model_rollout_steps], dtype=np.int64),
    }
    report_payload = {}
    for name, (l1, linf) in rollout_scores.items():
        npz_payload[f"l1_{name}"] = l1
        npz_payload[f"linf_{name}"] = linf
        report_payload[name] = {"l1": l1, "linf": linf}
    np.savez_compressed(os.path.join(args.outdir, f"demo_{args.demo}_rollout_metrics.npz"), **npz_payload)
    runtime_path = os.path.join(args.outdir, f"demo_{args.demo}_runtime.csv")
    with open(runtime_path, "w", encoding="utf-8") as f:
        f.write("method,seconds\n")
        for name, elapsed in runtime_rows:
            f.write(f"{name},{elapsed:.9f}\n")
    summary_path, curves_path = save_rollout_reports(args.outdir, times, report_payload)
    print(f"[demo-pri] saved csv: {summary_path}, {curves_path}, {runtime_path}")

    if args.plot_one:
        tag = f"demo_{args.demo}_steps{rollout_steps}"
        save_rollout_sample_plots(
            args.outdir,
            ref_traj,
            pred_trajs,
            tag=tag,
            shared_color_scale=args.shared_color_scale,
            share_ref_colorbar=args.share_ref_colorbar,
        )


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
    ap.add_argument("--warmup_steps", type=int, default=1)
    ap.add_argument("--skip_warmup", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow_tf32", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--outdir", type=str, default="eval_euler2d_pri_dt_out")
    ap.add_argument("--plot_one", action="store_true")
    ap.add_argument(
        "--shared_color_scale",
        action="store_true",
        help="Use one union primitive-field color scale across Ref/LGNO/FNO/WENO256 plot_one colormaps.",
    )
    ap.add_argument(
        "--share_ref_colorbar",
        action="store_true",
        help="Use the ref vmin/vmax per primitive field for every plot_one colormap.",
    )
    ap.add_argument("--weno_cfl", type=float, default=None)
    ap.add_argument("--weno_reconstruction", type=str, default=None, choices=("component", "characteristic"))
    ap.add_argument("--weno_line_batch_size", type=int, default=16)
    ap.add_argument("--weno_quiet", action="store_true", help="Disable normal-rollout WENO256 CFL substep printing.")
    ap.add_argument("--demo", type=str, default=None, choices=sorted(DEMO_CONFIGS))
    ap.add_argument("--demo_nx", type=int, default=256)
    ap.add_argument("--demo_ny", type=int, default=256)
    ap.add_argument("--demo_weno_fine_nx", type=int, default=512)
    ap.add_argument("--demo_weno_fine_ny", type=int, default=512)
    ap.add_argument("--demo_frame_dt", type=float, default=1e-2)
    ap.add_argument("--demo_T", type=float, default=0.5)
    ap.add_argument("--demo_split_x", type=float, default=None)
    ap.add_argument("--demo_split_y", type=float, default=None)
    ap.add_argument("--no_demo_scale", action="store_true")
    ap.add_argument(
        "--demo_model_velocity_scale",
        type=int,
        default=0,
        help="Integer k for Hybrid/FNO demo scaling: u,v -> u/k,v/k and p -> p/k^2. Use 0 to auto-select.",
    )
    ap.add_argument(
        "--demo_model_velocity_scale_max",
        type=int,
        default=4,
        help="Maximum integer k considered when --demo_model_velocity_scale=0.",
    )
    ap.add_argument("--demo_state_ul", type=float, nargs=4, default=None, metavar=("RHO", "U", "V", "P"))
    ap.add_argument("--demo_state_ur", type=float, nargs=4, default=None, metavar=("RHO", "U", "V", "P"))
    ap.add_argument("--demo_state_ll", type=float, nargs=4, default=None, metavar=("RHO", "U", "V", "P"))
    ap.add_argument("--demo_state_lr", type=float, nargs=4, default=None, metavar=("RHO", "U", "V", "P"))
    ap.add_argument("--demo_weno_cfl", type=float, default=0.45)
    ap.add_argument("--demo_weno_reconstruction", type=str, default="component", choices=("component", "characteristic"))
    ap.add_argument("--demo_weno_line_batch_size", type=int, default=16)
    ap.add_argument("--demo_weno_quiet", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    configure_runtime(device, allow_tf32=args.allow_tf32)
    pin_mem = device.type == "cuda"
    run_rollout = args.eval_mode in ("rollout", "both")
    run_test = args.eval_mode in ("test", "both")

    models: dict[str, tuple[torch.nn.Module, dict]] = {}
    primary_model = primary_ckpt = model_fno = ckpt_fno = None
    if run_rollout:
        primary_model, primary_ckpt = load_required_model(args.ckpt, device, no_compile=args.no_compile)
        models[model_label_from_ckpt(primary_ckpt)] = (primary_model, primary_ckpt)
        model_fno, ckpt_fno = load_optional_model(args.ckpt_fno, device, no_compile=args.no_compile, default_path=DEFAULT_FNO_CKPT)
        if model_fno is not None and ckpt_fno is not None:
            models[model_label_from_ckpt(ckpt_fno)] = (model_fno, ckpt_fno)
        print("[params] " + ", ".join(f"{name}={count_params(model):,}" for name, (model, _) in models.items()))

    if args.demo is not None:
        if not run_rollout:
            raise ValueError("--demo requires --eval_mode rollout or both.")
        run_demo_eval(args, models, device, pin_memory=pin_mem)
        print(f"Saved outputs to: {args.outdir}")
        return

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
        gamma = float(test_meta.get("gamma", 1.4))
        dx = float(test_meta["dx"])
        dy = float(test_meta["dy"])
        weno_cfl = float(test_meta.get("cfl", 0.4) if args.weno_cfl is None else args.weno_cfl)
        weno_reconstruction = args.weno_reconstruction or str(test_meta.get("reconstruction", "component"))
        weno_verbose = not bool(args.weno_quiet)
        weno_op = build_euler2d_solver(
            gamma=gamma,
            bc=str(test_meta.get("boundary", "outflow")),
            reconstruction=weno_reconstruction,
            WENOtype="WENO-Z",
            verbose=weno_verbose,
            line_batch_size=int(args.weno_line_batch_size),
        )
        print(
            f"[weno256-pri] reconstruction={weno_reconstruction}, cfl={weno_cfl:g}, "
            f"line_batch_size={args.weno_line_batch_size}, verbose={weno_verbose}"
        )
        l1 = np.zeros((n_eval, len(times)), dtype=np.float64)
        linf = np.zeros((n_eval, len(times)), dtype=np.float64)
        for idx in range(n_eval):
            ref = test_states[idx, : rollout_steps + 1].cpu().numpy()
            pred = rollout_weno256_from_primitive(
                weno_op,
                test_states[idx, 0],
                dx=dx,
                dy=dy,
                dt=dt,
                n_steps=rollout_steps,
                cfl=weno_cfl,
                gamma=gamma,
                log_ic_state=True,
            )
            for j in range(len(times)):
                l1[idx, j] = rel_l1(pred[j], ref[j])
                linf[idx, j] = rel_linf(pred[j], ref[j])
            if args.plot_one and idx == 0:
                one_ref = ref
                one_preds = {} if one_preds is None else one_preds
                one_preds["weno256"] = pred
        rollout_scores["weno256"] = (l1, linf)
        print(f"[rollout-pri] weno256: final relL1={l1[:, -1].mean():.3e}, final relLinf={linf[:, -1].mean():.3e}")
        for name, (model, _) in models.items():
            model.eval()
            l1 = np.zeros((n_eval, len(times)), dtype=np.float64)
            linf = np.zeros((n_eval, len(times)), dtype=np.float64)
            for idx in range(n_eval):
                ref = test_states[idx, : rollout_steps + 1].cpu().numpy()
                with torch.inference_mode():
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
            save_rollout_sample_plots(
                args.outdir,
                one_ref,
                one_preds,
                tag=tag,
                shared_color_scale=args.shared_color_scale,
                share_ref_colorbar=args.share_ref_colorbar,
            )

    if run_test:
        if run_rollout:
            primary_model = primary_ckpt = model_fno = ckpt_fno = model = None
            release_loaded_rollout_models(models, device)
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
