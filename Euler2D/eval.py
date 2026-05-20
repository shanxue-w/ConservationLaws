"""Evaluate Euler2D dt-step checkpoints on trajectory test data."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

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

from common import (
    DEFAULT_TEST_NAME,
    configure_runtime,
    load_trajectory_split,
    rollout_step_model,
    states_to_one_step_pairs,
)


# DEFAULT_HYBRID_CKPT = "checkpoints/euler2d_hybrid_dt.pt"
# DEFAULT_HYBRID_CKPT = "checkpoints/euler2d_hybrid_outflow_cnn_dt.pt"
# DEFAULT_FNO_CKPT = "checkpoints/euler2d_fno_dt.pt"
DEFAULT_HYBRID_CKPT = "checkpoints/euler2d_hybrid_outflow_1e-2_dt.pt"
DEFAULT_FNO_CKPT = "checkpoints/euler2d_fno_outflow_1e-2_dt.pt"
FIELD_MATH_LABELS = {
    "rho": r"$\rho$",
    "rhou": r"$\rho u$",
    "rhov": r"$\rho v$",
    "E": r"$E$",
    "u": r"$u$",
    "v": r"$v$",
    "p": r"$p$",
}
CONSERVATIVE_FIELD_NAMES = ("rho", "rhou", "rhov", "E")
ERROR_MATH_LABELS = {
    "rho": r"$|\rho-\rho_{\mathrm{ref}}|$",
    "rhou": r"$|\rho u-(\rho u)_{\mathrm{ref}}|$",
    "rhov": r"$|\rho v-(\rho v)_{\mathrm{ref}}|$",
    "E": r"$|E-E_{\mathrm{ref}}|$",
    "u": r"$|u-u_{\mathrm{ref}}|$",
    "v": r"$|v-v_{\mathrm{ref}}|$",
    "p": r"$|p-p_{\mathrm{ref}}|$",
}


def rel_l1(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + eps))


def rel_linf(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + eps))


def load_optional_model(
    ckpt_path: str,
    device: torch.device,
    *,
    no_compile: bool,
    default_path: str,
) -> tuple[torch.nn.Module | None, dict | None]:
    if not ckpt_path:
        return None, None
    if not os.path.isfile(ckpt_path):
        if ckpt_path == default_path:
            print(f"[eval] checkpoint not found, skipping baseline: {ckpt_path}")
            return None, None
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model, ckpt = load_hybrid_dt_step_2d(ckpt_path, device=device)
    model = compile_if_requested(model, device, no_compile=no_compile)
    return model, ckpt


def load_required_model(
    ckpt_path: str,
    device: torch.device,
    *,
    no_compile: bool,
) -> tuple[torch.nn.Module, dict]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model, ckpt = load_hybrid_dt_step_2d(ckpt_path, device=device)
    model = compile_if_requested(model, device, no_compile=no_compile)
    return model, ckpt


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
    primary_model, primary_ckpt = load_required_model(args.ckpt, device, no_compile=args.no_compile)
    yield model_label_from_ckpt(primary_ckpt), primary_model, primary_ckpt
    primary_model = None
    primary_ckpt = None
    release_eval_memory(device)

    model_fno, ckpt_fno = load_optional_model(
        args.ckpt_fno,
        device,
        no_compile=args.no_compile,
        default_path=DEFAULT_FNO_CKPT,
    )
    if model_fno is not None and ckpt_fno is not None:
        yield model_label_from_ckpt(ckpt_fno), model_fno, ckpt_fno
        model_fno = None
        ckpt_fno = None
        release_eval_memory(device)


def model_label_from_ckpt(ckpt: dict) -> str:
    kind = str(ckpt.get("kind", ""))
    if "fno" in kind:
        return "fno"
    return "hybrid"


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


def _safe_float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        out = float(value.detach().cpu().item())
    if np.isfinite(out):
        return out
    if np.isnan(out):
        return None
    if out == float("inf"):
        return 1.0e300
    if out == float("-inf"):
        return -1.0e300
    return None


def tensor_stats(name: str, x: torch.Tensor) -> dict[str, object]:
    x_det = x.detach()
    finite = torch.isfinite(x_det)
    n_total = int(x_det.numel())
    n_finite = int(finite.sum().item())
    n_nan = int(torch.isnan(x_det).sum().item())
    n_inf = int(torch.isinf(x_det).sum().item())
    out: dict[str, object] = {
        "name": name,
        "shape": list(x_det.shape),
        "dtype": str(x_det.dtype).replace("torch.", ""),
        "n_total": n_total,
        "n_finite": n_finite,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "finite": n_finite == n_total,
    }
    if n_finite <= 0:
        return out
    vals = x_det[finite]
    abs_vals = vals.abs()
    out.update(
        {
            "min": _safe_float(vals.min()),
            "max": _safe_float(vals.max()),
            "mean": _safe_float(vals.mean()),
            "std": _safe_float(vals.std(unbiased=False)),
            "absmax": _safe_float(abs_vals.max()),
            "rms": _safe_float(torch.sqrt((vals * vals).mean())),
        }
    )
    if n_total > 0:
        safe_abs = torch.nan_to_num(x_det.abs(), nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        flat_idx = int(torch.argmax(safe_abs.reshape(-1)).item())
        out["arg_absmax"] = [int(v) for v in np.unravel_index(flat_idx, tuple(x_det.shape))]
    return out


def state_channel_stats(prefix: str, u: torch.Tensor) -> list[dict[str, object]]:
    stats = []
    if u.dim() == 4 and int(u.size(-1)) == len(CONSERVATIVE_FIELD_NAMES):
        for c, key in enumerate(CONSERVATIVE_FIELD_NAMES):
            stats.append(tensor_stats(f"{prefix}.{key}", u[..., c]))
    elif u.dim() == 3 and int(u.size(0)) == len(CONSERVATIVE_FIELD_NAMES):
        for c, key in enumerate(CONSERVATIVE_FIELD_NAMES):
            stats.append(tensor_stats(f"{prefix}.{key}", u[c]))
    return stats


def primitive_torch_stats(prefix: str, u_bhwc: torch.Tensor, gamma: float) -> dict[str, object]:
    rho = u_bhwc[..., 0]
    rhou = u_bhwc[..., 1]
    rhov = u_bhwc[..., 2]
    energy = u_bhwc[..., 3]
    tiny = torch.finfo(u_bhwc.dtype).tiny
    rho_safe = torch.where(rho.abs() > tiny, rho, torch.full_like(rho, tiny))
    vel_u = rhou / rho_safe
    vel_v = rhov / rho_safe
    kinetic = 0.5 * (rhou * rhou + rhov * rhov) / rho_safe
    pressure = (float(gamma) - 1.0) * (energy - kinetic)
    sound_arg = float(gamma) * pressure / rho_safe
    sound_speed = torch.sqrt(torch.clamp(sound_arg, min=0.0))
    return {
        "rho": tensor_stats(f"{prefix}.primitive.rho", rho),
        "u": tensor_stats(f"{prefix}.primitive.u", vel_u),
        "v": tensor_stats(f"{prefix}.primitive.v", vel_v),
        "p": tensor_stats(f"{prefix}.primitive.p", pressure),
        "sound_speed": tensor_stats(f"{prefix}.primitive.c", sound_speed),
        "rho_min": _safe_float(torch.nan_to_num(rho, nan=float("inf")).min()),
        "p_min": _safe_float(torch.nan_to_num(pressure, nan=float("inf")).min()),
        "max_abs_velocity": _safe_float(torch.maximum(vel_u.abs(), vel_v.abs()).max()),
        "max_sound_speed": _safe_float(sound_speed.max()),
    }


def rollout_triggered(
    u_next: torch.Tensor,
    prim: dict[str, object],
    *,
    abs_threshold: float,
    rho_floor: float,
    p_floor: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not bool(torch.isfinite(u_next).all().item()):
        reasons.append("nonfinite_state")
    state_absmax = tensor_stats("u_next", u_next).get("absmax")
    if isinstance(state_absmax, float) and state_absmax >= abs_threshold:
        reasons.append(f"state_absmax>={abs_threshold:g}")
    rho_min = prim.get("rho_min")
    p_min = prim.get("p_min")
    if isinstance(rho_min, float) and rho_min <= rho_floor:
        reasons.append(f"rho_min<={rho_floor:g}")
    if isinstance(p_min, float) and p_min <= p_floor:
        reasons.append(f"p_min<={p_floor:g}")
    return bool(reasons), reasons


def hybrid_dt_step_2d_debug_forward(
    model: torch.nn.Module,
    u: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, object]] | None:
    base_model = getattr(model, "_orig_mod", model)
    backbone = getattr(base_model, "backbone", None)
    rhs_projector = getattr(base_model, "rhs_projector", None)
    if backbone is None or rhs_projector is None:
        return None
    required = ("lift", "spec_ops", "mr_local_ops", "w_g", "w_l", "mix", "head")
    if not all(hasattr(backbone, name) for name in required):
        return None
    if u.dim() != 4 or int(u.size(-1)) != int(getattr(base_model, "in_channels", u.size(-1))):
        return None

    layer_stats: list[dict[str, object]] = []
    x = u.permute(0, 3, 1, 2).contiguous()
    h = backbone.lift(x).permute(0, 2, 3, 1).contiguous()
    layer_stats.append(tensor_stats("backbone.lift", h))
    h = F.silu(h)
    layer_stats.append(tensor_stats("backbone.lift_silu", h))
    for t in range(int(backbone.n_layers)):
        layer_stats.append(tensor_stats(f"backbone.layer{t}.h_in", h))
        spec_raw = backbone.spec_ops[t](h)
        layer_stats.append(tensor_stats(f"backbone.layer{t}.spec_raw", spec_raw))
        g = F.silu(spec_raw)
        layer_stats.append(tensor_stats(f"backbone.layer{t}.spec_silu", g))
        mr_raw = backbone.mr_local_ops[t](h)
        layer_stats.append(tensor_stats(f"backbone.layer{t}.mr_raw", mr_raw))
        lfeat = F.silu(mr_raw)
        layer_stats.append(tensor_stats(f"backbone.layer{t}.mr_silu", lfeat))
        gp = g.permute(0, 3, 1, 2)
        lp = lfeat.permute(0, 3, 1, 2)
        gc = backbone.w_g[t](gp)
        lc = backbone.w_l[t](lp)
        layer_stats.append(tensor_stats(f"backbone.layer{t}.gate_g", gc))
        layer_stats.append(tensor_stats(f"backbone.layer{t}.gate_l", lc))
        c = (gc * lc).permute(0, 2, 3, 1)
        layer_stats.append(tensor_stats(f"backbone.layer{t}.gate_product", c))
        z = torch.cat([h, g, lfeat, c], dim=-1).permute(0, 3, 1, 2)
        layer_stats.append(tensor_stats(f"backbone.layer{t}.mix_input", z))
        mix_delta = backbone.mix[t](z).permute(0, 2, 3, 1)
        layer_stats.append(tensor_stats(f"backbone.layer{t}.mix_delta", mix_delta))
        h = h + mix_delta
        layer_stats.append(tensor_stats(f"backbone.layer{t}.h_out", h))
    head_in = h.permute(0, 3, 1, 2)
    if isinstance(backbone.head, torch.nn.Sequential) and len(backbone.head) == 3:
        head_hidden = backbone.head[0](head_in)
        layer_stats.append(tensor_stats("backbone.head.0", head_hidden))
        head_hidden_act = backbone.head[1](head_hidden)
        layer_stats.append(tensor_stats("backbone.head.1", head_hidden_act))
        tilde_rhs = backbone.head[2](head_hidden_act).permute(0, 2, 3, 1)
    else:
        tilde_rhs = backbone.head(head_in).permute(0, 2, 3, 1)
    layer_stats.append(tensor_stats("backbone.head.out", tilde_rhs))

    _, ny, nx, _ = u.shape
    proj = rhs_projector(
        tilde_rhs,
        h=h,
        nx=nx,
        ny=ny,
        zero_mean_rhs=bool(getattr(base_model, "zero_mean_rhs", True)),
        u_ref=u,
    )
    rhs = proj["rhs"]
    u_next = base_model.evolve(u, dt, rhs)
    aux: dict[str, object] = {
        "tilde_rhs": tilde_rhs,
        "rhs": rhs,
        "h": h,
        "layer_stats": layer_stats,
    }
    for key, value in proj.items():
        if key != "rhs":
            aux[key] = value
    return u_next, aux


def write_rollout_debug_reports(outdir: str, tag: str, records: list[dict[str, object]]) -> tuple[str, str]:
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, f"rollout_debug_{tag}.json")
    csv_path = os.path.join(outdir, f"rollout_debug_{tag}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, sort_keys=True)
    rows: list[dict[str, object]] = []
    for record in records:
        base = {
            "model": record.get("model"),
            "sample_idx": record.get("sample_idx"),
            "step": record.get("step"),
            "time": record.get("time"),
            "triggered": record.get("triggered"),
            "reasons": ";".join(record.get("reasons", [])),
            "rel_l1": record.get("rel_l1"),
            "rel_linf": record.get("rel_linf"),
        }
        for stat in record.get("stats", []):
            row = dict(base)
            row.update(
                {
                    "name": stat.get("name"),
                    "finite": stat.get("finite"),
                    "n_nan": stat.get("n_nan"),
                    "n_inf": stat.get("n_inf"),
                    "min": stat.get("min"),
                    "max": stat.get("max"),
                    "mean": stat.get("mean"),
                    "std": stat.get("std"),
                    "absmax": stat.get("absmax"),
                    "rms": stat.get("rms"),
                    "arg_absmax": stat.get("arg_absmax"),
                }
            )
            rows.append(row)
    fieldnames = [
        "model",
        "sample_idx",
        "step",
        "time",
        "triggered",
        "reasons",
        "rel_l1",
        "rel_linf",
        "name",
        "finite",
        "n_nan",
        "n_inf",
        "min",
        "max",
        "mean",
        "std",
        "absmax",
        "rms",
        "arg_absmax",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


@torch.no_grad()
def rollout_step_model_debug(
    model_name: str,
    model: torch.nn.Module,
    u0: torch.Tensor,
    ref: np.ndarray,
    n_steps: int,
    dt: float,
    device: torch.device,
    *,
    dtype: torch.dtype,
    gamma: float,
    sample_idx: int,
    pin_memory: bool = False,
    abs_threshold: float = 1e3,
    rho_floor: float = 0.0,
    p_floor: float = 0.0,
    window: int = 2,
    stop_after_trigger: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    model.eval()
    if u0.ndim != 3:
        raise ValueError(f"Expected initial state (4, ny, nx), got {tuple(u0.shape)}")
    u = u0.permute(1, 2, 0).unsqueeze(0).contiguous()
    if pin_memory and device.type == "cuda":
        u = u.pin_memory()
    u = u.to(device=device, dtype=dtype, non_blocking=pin_memory and device.type == "cuda")
    traj = [u0.cpu().numpy()]
    records: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    first_trigger_step: int | None = None
    max_record_step = -1
    for step in range(1, n_steps + 1):
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        dt_b = torch.full((u.size(0),), float(dt), device=device, dtype=dtype)
        debug_result = hybrid_dt_step_2d_debug_forward(model, u, dt_b)
        if debug_result is not None:
            u_next, aux = debug_result
        elif hasattr(model, "forward"):
            try:
                result = model(u, dt_b, return_aux=True)
            except TypeError:
                result = model(u, dt_b)
            if isinstance(result, tuple):
                u_next, aux = result
            else:
                u_next = result
                aux = {}
        else:
            result = model(u, dt_b, return_aux=True)
            if isinstance(result, tuple):
                u_next, aux = result
            else:
                u_next = result
                aux = {}
        u_next = u_next.clone()
        pred_chw = u_next.squeeze(0).permute(2, 0, 1).detach().cpu().numpy()
        traj.append(pred_chw)

        prim = primitive_torch_stats("u_next", u_next, gamma)
        triggered, reasons = rollout_triggered(
            u_next,
            prim,
            abs_threshold=abs_threshold,
            rho_floor=rho_floor,
            p_floor=p_floor,
        )
        record_stats: list[dict[str, object]] = []
        record_stats.extend(state_channel_stats("u_in", u))
        record_stats.extend(state_channel_stats("u_next", u_next))
        for key in ("tilde_rhs", "rhs", "h"):
            if key in aux and torch.is_tensor(aux[key]):
                record_stats.append(tensor_stats(key, aux[key]))
                if key in ("tilde_rhs", "rhs"):
                    record_stats.extend(state_channel_stats(key, aux[key]))
        for key in ("F_w", "F_e", "F_s", "F_n"):
            if key in aux and torch.is_tensor(aux[key]):
                record_stats.append(tensor_stats(key, aux[key]))
        for stat in aux.get("layer_stats", []):
            record_stats.append(stat)
        record_stats.extend([prim["rho"], prim["u"], prim["v"], prim["p"], prim["sound_speed"]])

        record = {
            "model": model_name,
            "sample_idx": int(sample_idx),
            "step": step,
            "time": float(step * dt),
            "triggered": triggered,
            "reasons": reasons,
            "rel_l1": rel_l1(pred_chw, ref[step]) if step < len(ref) else None,
            "rel_linf": rel_linf(pred_chw, ref[step]) if step < len(ref) else None,
            "primitive_summary": {
                "rho_min": prim.get("rho_min"),
                "p_min": prim.get("p_min"),
                "max_abs_velocity": prim.get("max_abs_velocity"),
                "max_sound_speed": prim.get("max_sound_speed"),
            },
            "stats": record_stats,
        }
        history.append(record)
        history = history[-max(1, int(window)) :]
        if triggered and first_trigger_step is None:
            first_trigger_step = step
            records.extend(history)
            max_record_step = max(int(r["step"]) for r in records)
            print(
                f"[debug:{model_name}] trigger at sample={sample_idx} step={step} "
                f"t={step * dt:g}: {', '.join(reasons)}",
                flush=True,
            )
            if stop_after_trigger:
                break
        elif first_trigger_step is not None and step <= first_trigger_step + int(window):
            if step > max_record_step:
                records.append(record)
                max_record_step = step
        u = u_next
    return np.stack(traj, axis=0), records


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


def field_limits(fields: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    return {key: (float(np.min(values)), float(np.max(values))) for key, values in fields.items()}


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
    cmap: str = "viridis",
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
    cmap: str = "viridis",
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
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )


def save_rollout_sample_plots(
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_HYBRID_CKPT)
    ap.add_argument("--ckpt_fno", type=str, default=DEFAULT_FNO_CKPT)
    ap.add_argument("--eval_mode", type=str, default="both", choices=("rollout", "test", "both"))
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--test_name", type=str, default=DEFAULT_TEST_NAME)
    ap.add_argument("--test_batch", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=0, help="0 means use all test trajectories.")
    ap.add_argument("--sample_seed", type=int, default=None, help="Random seed used to choose evaluation trajectories.")
    ap.add_argument(
        "--rollout_steps",
        type=int,
        default=0,
        help="Number of rollout steps to evaluate. 0 means use the full trajectory.",
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow_tf32", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--outdir", type=str, default="eval_euler2d_dt_out")
    ap.add_argument("--plot_one", action="store_true")
    ap.add_argument(
        "--debug_rollout",
        action="store_true",
        help="Record detailed hybrid rollout diagnostics around the first unstable/nonphysical step.",
    )
    ap.add_argument(
        "--debug_model",
        type=str,
        default="hybrid",
        choices=("hybrid", "fno", "all"),
        help="Which model to instrument when --debug_rollout is set.",
    )
    ap.add_argument(
        "--debug_abs_threshold",
        type=float,
        default=1e3,
        help="Trigger rollout diagnostics when |state| reaches this value.",
    )
    ap.add_argument(
        "--debug_rho_floor",
        type=float,
        default=0.0,
        help="Trigger rollout diagnostics when predicted density falls at or below this value.",
    )
    ap.add_argument(
        "--debug_p_floor",
        type=float,
        default=0.0,
        help="Trigger rollout diagnostics when predicted pressure falls at or below this value.",
    )
    ap.add_argument(
        "--debug_window",
        type=int,
        default=2,
        help="Number of steps before/after the first trigger to keep in debug reports.",
    )
    ap.add_argument(
        "--share_ref_colorbar",
        action="store_true",
        help="Use union(ref, hybrid) vmin/vmax per field for ref & hybrid conserved/primitive plots; fno independent.",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    configure_runtime(device, allow_tf32=args.allow_tf32)
    pin_mem = device.type == "cuda"
    run_rollout = args.eval_mode in ("rollout", "both")
    run_test = args.eval_mode in ("test", "both")

    models: dict[str, tuple[torch.nn.Module, dict]] = {}
    if run_rollout:
        force_no_compile = bool(args.no_compile or args.debug_rollout)
        primary_model, primary_ckpt = load_required_model(args.ckpt, device, no_compile=force_no_compile)
        primary_name = model_label_from_ckpt(primary_ckpt)
        models[primary_name] = (primary_model, primary_ckpt)

        model_fno, ckpt_fno = load_optional_model(
            args.ckpt_fno,
            device,
            no_compile=force_no_compile,
            default_path=DEFAULT_FNO_CKPT,
        )
        if model_fno is not None and ckpt_fno is not None:
            models[model_label_from_ckpt(ckpt_fno)] = (model_fno, ckpt_fno)

        model_param_counts = {name: count_params(model) for name, (model, _) in models.items()}
        if "hybrid" in model_param_counts:
            print(f"[params] hybrid={model_param_counts['hybrid']:,}")
        else:
            print("[params] hybrid=unavailable")
        if "fno" in model_param_counts:
            print(f"[params] fno={model_param_counts['fno']:,}")
        else:
            print("[params] fno=unavailable")

    test_path = os.path.join(args.data_dir, args.test_name)
    all_test_states, test_meta = load_trajectory_split(test_path)
    dt = float(test_meta["dt"])
    gamma = float(test_meta.get("gamma", 1.4))
    total_traj = int(all_test_states.shape[0])
    selected_idx = select_trajectory_indices(
        total_traj,
        args.n_samples if args.n_samples > 0 else None,
        args.sample_seed,
    )
    test_states = all_test_states[torch.as_tensor(selected_idx, dtype=torch.long)]
    print(
        f"[data] test_path={test_path}, n_traj={int(test_states.shape[0])}/{total_traj}, "
        f"n_snaps={int(test_states.shape[1])}, dt={dt}, "
        f"nx={int(test_meta['nx'])}, ny={int(test_meta['ny'])}, boundary={test_meta.get('boundary', '?')}, "
        f"sample_seed={args.sample_seed}"
    )
    if run_rollout:
        print("[models] " + ", ".join(sorted(models.keys())))
    elif run_test:
        print("[models] test mode loads models sequentially")
    if args.plot_one and selected_idx.size > 0:
        print(f"[plot_one] selected trajectory index={int(selected_idx[0])}")

    if run_rollout:
        n_eval = int(test_states.shape[0])
        max_rollout_steps = int(test_states.shape[1] - 1)
        rollout_steps = max_rollout_steps if args.rollout_steps <= 0 else min(int(args.rollout_steps), max_rollout_steps)
        times = np.arange(rollout_steps + 1, dtype=np.float64) * dt
        rollout_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        one_ref = None
        one_preds = None
        all_debug_records: list[dict[str, object]] = []
        print(f"[rollout] steps={rollout_steps}/{max_rollout_steps}")
        for name, (model, _) in models.items():
            l1 = np.zeros((n_eval, len(times)), dtype=np.float64)
            linf = np.zeros((n_eval, len(times)), dtype=np.float64)
            debug_this_model = bool(args.debug_rollout and (args.debug_model == "all" or args.debug_model == name))
            for idx in range(n_eval):
                ref = test_states[idx, : rollout_steps + 1].cpu().numpy()
                if debug_this_model:
                    pred, debug_records = rollout_step_model_debug(
                        name,
                        model,
                        test_states[idx, 0],
                        ref,
                        n_steps=rollout_steps,
                        dt=dt,
                        device=device,
                        dtype=torch.float32,
                        gamma=gamma,
                        sample_idx=int(selected_idx[idx]),
                        pin_memory=pin_mem,
                        abs_threshold=float(args.debug_abs_threshold),
                        rho_floor=float(args.debug_rho_floor),
                        p_floor=float(args.debug_p_floor),
                        window=int(args.debug_window),
                    )
                    all_debug_records.extend(debug_records)
                else:
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
        if args.debug_rollout:
            if all_debug_records:
                debug_tag = f"{sample_tag(int(selected_idx[0]), args.sample_seed)}_steps{rollout_steps}_{args.debug_model}"
                debug_json, debug_csv = write_rollout_debug_reports(args.outdir, debug_tag, all_debug_records)
                print(f"[debug] saved rollout diagnostics: {debug_json}, {debug_csv}")
            else:
                print("[debug] no rollout trigger reached; no diagnostic report was written.")

        fig, ax = plt.subplots(figsize=(7, 5))
        for name, (l1, _) in rollout_scores.items():
            mean = l1.mean(axis=0)
            std = l1.std(axis=0)
            ax.plot(times, mean, label=f"{name} rel L1")
            ax.fill_between(times, mean - std, mean + std, alpha=0.15)
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("relative L1 error")
        ax.set_title("Euler2D rollout error vs time")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "relL1_vs_time.pdf"), bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        for name, (_, linf) in rollout_scores.items():
            ax.plot(times, linf.mean(axis=0), label=f"{name} rel Linf")
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("relative Linf error")
        ax.set_title("Euler2D rollout Linf vs time")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "relLinf_vs_time.pdf"), bbox_inches="tight")
        plt.close(fig)

        if args.plot_one and one_ref is not None and one_preds is not None:
            plot_tag = f"{sample_tag(int(selected_idx[0]), args.sample_seed)}_steps{rollout_steps}"
            plot_models = {k: v for k, v in one_preds.items() if k in ("hybrid", "fno")}
            save_rollout_sample_plots(
                args.outdir,
                one_ref,
                plot_models,
                final_time=float(times[-1]),
                gamma=gamma,
                tag=plot_tag,
                share_ref_colorbar=args.share_ref_colorbar,
            )

    if run_test:
        if run_rollout:
            models.clear()
            primary_model = None
            primary_ckpt = None
            model_fno = None
            ckpt_fno = None
            release_eval_memory(device)
        u0_test, u1_test = states_to_one_step_pairs(test_states, dtype=torch.float32)
        print(f"[test] one-step pairs={int(u0_test.shape[0])}")
        test_rows = []
        for name, model, _ in iter_test_models(args, device):
            print(f"[test] evaluating {name}, params={count_params(model):,}")
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
                    progress_every=100,
                )
            )
            del model
            release_eval_memory(device)
        test_csv_path = save_test_metrics_csv(args.outdir, test_rows)
        test_summary_csv_path = save_test_metrics_summary_csv(args.outdir, test_rows)
        print(f"[test] saved csv: {test_csv_path}, {test_summary_csv_path}")

    print(f"Saved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
