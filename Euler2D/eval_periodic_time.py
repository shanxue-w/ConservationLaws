"""
Compare Euler2D periodic rollout wall-clock cost for Hybrid, FNO, and WENO-Z.

Unlike ``eval_periodic.py``, this script does not use the saved full test
trajectory as the reference.  It reads the first frame of each selected
trajectory, advances that frame with the FD-WENOZ solver to the requested
snapshot times, and times all three rollout paths on the same coarse mesh.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time

# ``dataset.data_periodic`` imports the external ``clop.solver`` WENO backend,
# whose Numba kernels use ``cache=True``.  In this workspace that backend lives
# outside the current repo, so give Numba an explicit writable cache locator.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join("/tmp", "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

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

from conslaw.models import count_params

from common_periodic import (
    DEFAULT_TEST_NAME,
    configure_runtime,
    default_dt_ckpt_path,
    rollout_step_model,
    torch_load_cpu,
)
from dataset.data_periodic import (
    DEFAULT_P_RANGE,
    DEFAULT_RHO_RANGE,
    DEFAULT_UV_RANGE,
    build_euler2d_solver,
    downsample_conserved2d,
    enforce_physical_euler2d,
    primitives_to_conserved_2d,
)
from eval_periodic import (
    apply_line_grid,
    conserved_to_primitive_2d,
    display_model_name,
    field_math_label,
    load_model_from_ckpt,
    load_optional_model,
    model_label_from_ckpt,
    rel_l1,
    rel_linf,
    release_eval_memory,
    resolve_input_path,
    sample_tag,
    save_density_plots,
    select_trajectory_indices,
    state_to_conserved_fields,
)


DEFAULT_TIME_TEST_NAME = DEFAULT_TEST_NAME
DEFAULT_HYBRID_CKPT = default_dt_ckpt_path("hybrid", "periodic")
DEFAULT_FNO_CKPT = default_dt_ckpt_path("fno", "periodic")
RIEMANN_DEMO_CONFIGS = {
    "quad_shock": {
        "description": (
            "Primitive states stored as rho,u,v,p; converted from Lua p,rho,u,v. "
            "Source: https://ammar-hakim.org/sj/sims/s398/s398-euler-reim-ds-2d.html"
        ),
        "split_x": 0.5,
        "split_y": 0.5,
        "states": {
            "ul": (0.5197, -0.6259, -0.3, 0.4),
            "ur": (1.0, 0.1, -0.3, 1.0),
            "ll": (0.8, 0.1, -0.3, 0.4),
            "lr": (0.5313, 0.1, 0.4276, 0.4),
        },
    },
    "quad_shock_vacuum": {
        "description": (
            "Primitive states stored as rho,u,v,p; converted from Lua p,rho,u,v. "
            "Source: https://ammar-hakim.org/sj/sims/s395/s395-euler-reim-ds-2d.html"
        ),
        "split_x": 0.5,
        "split_y": 0.5,
        "states": {
            "ul": (0.5065, 0.8939, 0.0, 0.35),
            "ur": (1.1, 0.0, 0.0, 1.1),
            "ll": (1.1, 0.8939, 0.8939, 1.1),
            "lr": (0.5065, 0.0, 0.8939, 0.35),
        },
    },
    "quad_shock_s397": {
        "description": (
            "Primitive states stored as rho,u,v,p; converted from Lua p,rho,u,v. "
            "Source: https://ammar-hakim.org/sj/sims/s397/s397-euler-reim-ds-2d.html"
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
    "riemann_01": {
        "description": "Configuration 2. State indices map as 1=UR, 2=UL, 3=LL, 4=LR.",
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
        "description": "Configuration 7. State indices map as 1=UR, 2=UL, 3=LL, 4=LR.",
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
        "description": "Configuration 8. State indices map as 1=UR, 2=UL, 3=LL, 4=LR.",
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
        "description": "Configuration 12. State indices map as 1=UR, 2=UL, 3=LL, 4=LR.",
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


def resolve_test_split_path(args: argparse.Namespace) -> str:
    if args.test_path:
        return resolve_input_path(args.test_path, label="Test dataset")
    return resolve_input_path(os.path.join(args.data_dir, args.test_name), label="Test dataset")


def display_time_model_name(name: str) -> str:
    key = str(name).lower()
    if key == "weno":
        return "WENO-Z"
    if key == "weno256":
        return "WENO-Z 256"
    if key == "weno512":
        return "WENO-Z 512"
    return display_model_name(name)


def _normalize_first_frame(states: torch.Tensor, src_path: str) -> torch.Tensor:
    if not torch.is_tensor(states):
        raise ValueError(f"`states` is not a tensor in {src_path}")
    if states.ndim == 4:
        return states[0]
    if states.ndim == 5 and int(states.shape[0]) == 1:
        return states[0, 0]
    raise ValueError(
        f"Unsupported trajectory states shape in {src_path}: expected "
        f"(n_snaps, 4, ny, nx) or (1, n_snaps, 4, ny, nx), got {tuple(states.shape)}"
    )


def load_first_frames(split_path: str, selected_idx: np.ndarray) -> tuple[torch.Tensor, dict]:
    item = torch_load_cpu(split_path)
    if "trajectory_files" in item:
        meta = dict(item["meta"])
        rel_paths = list(item["trajectory_files"])
        base_dir = os.path.dirname(split_path)
        frames = []
        missing_paths = []
        for idx in selected_idx:
            if int(idx) < 0 or int(idx) >= len(rel_paths):
                raise IndexError(f"Selected trajectory index {int(idx)} is outside [0, {len(rel_paths)})")
            traj_path = os.path.join(base_dir, rel_paths[int(idx)])
            if not os.path.isfile(traj_path):
                missing_paths.append(traj_path)
                continue
            traj_item = torch_load_cpu(traj_path)
            frames.append(_normalize_first_frame(traj_item["states"], traj_path))
        if missing_paths:
            fallback_path = split_path.replace("_manifest.pt", ".pt")
            if os.path.isfile(fallback_path):
                print(
                    f"[data] manifest is incomplete ({len(missing_paths)} selected trajectory file(s) missing); "
                    f"falling back to {fallback_path}",
                    flush=True,
                )
                return load_first_frames(fallback_path, selected_idx)
            raise FileNotFoundError(
                "Missing trajectory file(s) from manifest and no monolithic fallback found. "
                f"First missing path: {missing_paths[0]}"
            )
        if not frames:
            raise ValueError("No trajectories selected.")
        return torch.stack(frames, dim=0).to(torch.float32), meta

    if "states" not in item:
        raise ValueError(f"Unsupported split format in {split_path}: keys={list(item.keys())}")
    states = item["states"]
    if not torch.is_tensor(states) or states.ndim != 5:
        raise ValueError(f"Expected split states with shape (n_traj, n_snaps, 4, ny, nx), got {type(states)}")
    frames = states[torch.as_tensor(selected_idx, dtype=torch.long), 0]
    return frames.to(torch.float32), dict(item["meta"])


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_model_rollout(
    name: str,
    model: torch.nn.Module,
    u0: torch.Tensor,
    *,
    n_steps: int,
    dt: float,
    device: torch.device,
    pin_memory: bool,
    warmup_steps: int = 0,
) -> tuple[np.ndarray, float]:
    warmup_model_rollout(
        name,
        model,
        u0,
        n_steps=warmup_steps,
        dt=dt,
        device=device,
        pin_memory=pin_memory,
    )
    synchronize_if_needed(device)
    t0 = time.perf_counter()
    traj = rollout_step_model(
        model,
        u0,
        n_steps=n_steps,
        dt=dt,
        device=device,
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    synchronize_if_needed(device)
    elapsed = time.perf_counter() - t0
    print(f"[time] {name}: {elapsed:.6f} s")
    return traj, elapsed


def timed_weno_rollout(
    name: str,
    op,
    u0: torch.Tensor,
    *,
    dx: float,
    dy: float,
    dt_snap: float,
    n_steps: int,
    cfl: float,
    gamma: float,
    warmup_steps: int = 0,
) -> tuple[np.ndarray, float]:
    warmup_weno_rollout(
        name,
        op,
        u0,
        dx=dx,
        dy=dy,
        dt_snap=dt_snap,
        n_steps=warmup_steps,
        cfl=cfl,
        gamma=gamma,
    )
    u_np = enforce_physical_euler2d(u0.cpu().numpy(), gamma=gamma)
    t0 = time.perf_counter()
    traj = op.solve_snapshots(
        u_np,
        dx,
        dy,
        dt_snap=dt_snap,
        n_snaps=n_steps + 1,
        cfl=cfl,
        log_ic_state=False,
    )
    elapsed = time.perf_counter() - t0
    print(f"[time] {name}: {elapsed:.6f} s")
    return traj.astype(np.float64, copy=False), elapsed


def warmup_model_rollout(
    name: str,
    model: torch.nn.Module,
    u0: torch.Tensor,
    *,
    n_steps: int,
    dt: float,
    device: torch.device,
    pin_memory: bool,
) -> None:
    if n_steps <= 0:
        return
    print(f"[warmup] {name}: {n_steps} step(s), not timed", flush=True)
    _ = rollout_step_model(
        model,
        u0,
        n_steps=n_steps,
        dt=dt,
        device=device,
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    synchronize_if_needed(device)


def warmup_weno_rollout(
    name: str,
    op,
    u0: torch.Tensor,
    *,
    dx: float,
    dy: float,
    dt_snap: float,
    n_steps: int,
    cfl: float,
    gamma: float,
) -> None:
    if n_steps <= 0:
        return
    print(f"[warmup] {name}: {n_steps} CFL substep(s), not timed", flush=True)
    u_np = enforce_physical_euler2d(u0.cpu().numpy(), gamma=gamma)
    old_verbose = getattr(op, "verbose", None)
    if old_verbose is not None:
        op.verbose = False
    try:
        for _ in range(n_steps):
            dt_sub = op._compute_dt(u_np, dx, dy, cfl=cfl, t_remaining=dt_snap)
            if dt_sub <= 0.0:
                raise RuntimeError("dt_sub <= 0 in WENO warmup")
            u_np = op.step(u_np, dx, dy, dt_sub)
    finally:
        if old_verbose is not None:
            op.verbose = old_verbose


def make_kelvin_helmholtz_state(
    nx: int,
    ny: int,
    *,
    seed: int,
    noise_amp: float,
    pressure: float,
    gamma: float,
) -> np.ndarray:
    y = (np.arange(int(ny), dtype=np.float64) + 0.5) / float(ny)
    inner = (y >= 0.25) & (y <= 0.75)
    rho = np.where(inner[:, None], 2.0, 1.0) * np.ones((int(ny), int(nx)), dtype=np.float64)
    u_vel = np.where(inner[:, None], 0.5, -0.5) * np.ones((int(ny), int(nx)), dtype=np.float64)
    v_vel = np.zeros((int(ny), int(nx)), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    u_vel += rng.uniform(-float(noise_amp), float(noise_amp), size=(int(ny), int(nx)))
    v_vel += rng.uniform(-float(noise_amp), float(noise_amp), size=(int(ny), int(nx)))
    p = np.full((int(ny), int(nx)), float(pressure), dtype=np.float64)
    return enforce_physical_euler2d(
        primitives_to_conserved_2d(rho, u_vel, v_vel, p, gamma=gamma),
        gamma=gamma,
    )


def primitive_state_ranges(
    states: dict[str, tuple[float, float, float, float]],
) -> dict[str, tuple[float, float]]:
    values = np.asarray(list(states.values()), dtype=np.float64)
    return {
        "rho": (float(np.min(values[:, 0])), float(np.max(values[:, 0]))),
        "u": (float(np.min(values[:, 1])), float(np.max(values[:, 1]))),
        "v": (float(np.min(values[:, 2])), float(np.max(values[:, 2]))),
        "p": (float(np.min(values[:, 3])), float(np.max(values[:, 3]))),
    }


def fit_riemann_demo_rhop_scale(
    states: dict[str, tuple[float, float, float, float]],
    rho_range: tuple[float, float] = DEFAULT_RHO_RANGE,
    p_range: tuple[float, float] = DEFAULT_P_RANGE,
) -> float:
    ranges = primitive_state_ranges(states)
    rho_min, rho_max = ranges["rho"]
    p_min, p_max = ranges["p"]
    if min(rho_min, rho_max, p_min, p_max) <= 0.0:
        raise ValueError("Riemann demo scale requires positive rho and p in every quadrant.")
    lower = max(float(rho_range[0]) / rho_min, float(p_range[0]) / p_min)
    upper = min(float(rho_range[1]) / rho_max, float(p_range[1]) / p_max)
    if lower <= 1.0 <= upper:
        return 1.0
    if upper < 1.0:
        return float(upper)
    if lower > 1.0:
        return float(lower)
    return float(np.sqrt(lower * upper))


def scale_riemann_demo_states_rhop(
    states: dict[str, tuple[float, float, float, float]],
    scale: float,
) -> dict[str, tuple[float, float, float, float]]:
    factor = float(scale)
    return {
        key: (float(rho) * factor, float(u), float(v), float(p) * factor)
        for key, (rho, u, v, p) in states.items()
    }


def validate_riemann_demo_velocity_range(
    states: dict[str, tuple[float, float, float, float]],
    uv_range: tuple[float, float] = DEFAULT_UV_RANGE,
) -> None:
    ranges = primitive_state_ranges(states)
    u_min, u_max = ranges["u"]
    v_min, v_max = ranges["v"]
    lo, hi = float(uv_range[0]), float(uv_range[1])
    if u_min < lo or v_min < lo or u_max > hi or v_max > hi:
        print(
            f"[demo] warning: velocity range u=({u_min:g},{u_max:g}), "
            f"v=({v_min:g},{v_max:g}) is outside training range ({lo:g},{hi:g}); "
            "no velocity scale is applied in periodic timing demos.",
            flush=True,
        )


def make_riemann_state(
    nx: int,
    ny: int,
    *,
    split_x: float,
    split_y: float,
    states: dict[str, tuple[float, float, float, float]],
    gamma: float,
) -> np.ndarray:
    x = (np.arange(int(nx), dtype=np.float64) + 0.5) / float(nx)
    y = (np.arange(int(ny), dtype=np.float64) + 0.5) / float(ny)
    xx, yy = np.meshgrid(x, y)
    rho = np.empty((int(ny), int(nx)), dtype=np.float64)
    u_vel = np.empty_like(rho)
    v_vel = np.empty_like(rho)
    p = np.empty_like(rho)
    masks = {
        "ul": (xx < split_x) & (yy > split_y),
        "ur": (xx >= split_x) & (yy > split_y),
        "ll": (xx < split_x) & (yy <= split_y),
        "lr": (xx >= split_x) & (yy <= split_y),
    }
    for key, mask in masks.items():
        rho_q, u_q, v_q, p_q = states[key]
        rho[mask] = float(rho_q)
        u_vel[mask] = float(u_q)
        v_vel[mask] = float(v_q)
        p[mask] = float(p_q)
    return enforce_physical_euler2d(
        primitives_to_conserved_2d(rho, u_vel, v_vel, p, gamma=gamma),
        gamma=gamma,
    )


def downsample_trajectory_conserved2d(traj: np.ndarray, factor_y: int, factor_x: int | None = None) -> np.ndarray:
    return np.stack(
        [downsample_conserved2d(np.asarray(state, dtype=np.float64), factor_y, factor_x) for state in traj],
        axis=0,
    )


def append_runtime_row(
    rows: list[dict[str, object]],
    *,
    sample_order: int,
    sample_idx: int | str,
    method: str,
    seconds: float,
    n_steps: int,
    frame_dt: float,
    weno_cfl: float,
    nx: int,
    ny: int,
    device: str,
) -> None:
    rows.append(
        {
            "sample_order": sample_order,
            "sample_idx": sample_idx,
            "method": method,
            "seconds": seconds,
            "n_steps": n_steps,
            "frame_dt": frame_dt,
            "weno_cfl": weno_cfl,
            "nx": nx,
            "ny": ny,
            "device": device,
        }
    )


def save_runtime_csvs(
    outdir: str,
    rows: list[dict[str, object]],
    summary: dict[str, dict[str, float]],
) -> None:
    per_sample_path = os.path.join(outdir, "runtime_per_sample.csv")
    with open(per_sample_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_order",
                "sample_idx",
                "method",
                "seconds",
                "n_steps",
                "frame_dt",
                "weno_cfl",
                "nx",
                "ny",
                "device",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(outdir, "runtime_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "n", "total_seconds", "mean_seconds", "std_seconds"],
        )
        writer.writeheader()
        for method in ("hybrid", "fno", "weno", "weno256", "weno512"):
            if method not in summary:
                continue
            row = dict(summary[method])
            row["method"] = method
            writer.writerow(row)
    print(f"[time] saved csv: {per_sample_path}, {summary_path}")


def save_runtime_bar(outdir: str, summary: dict[str, dict[str, float]]) -> None:
    methods = [m for m in ("hybrid", "fno", "weno", "weno256", "weno512") if m in summary]
    if not methods:
        return
    means = [summary[m]["mean_seconds"] for m in methods]
    stds = [summary[m]["std_seconds"] for m in methods]
    labels = [display_time_model_name(m) for m in methods]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = ["#2b6cb0", "#38a169", "#805ad5", "#d69e2e", "#dd6b20"]
    ax.bar(labels, means, yerr=stds, capsize=4, color=colors[: len(methods)])
    ax.set_ylabel("Seconds per trajectory")
    apply_line_grid(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "runtime_bar.pdf"), bbox_inches="tight")
    plt.close(fig)


def summarize_timings(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_method: dict[str, list[float]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(float(row["seconds"]))
    summary: dict[str, dict[str, float]] = {}
    for method, values in by_method.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[method] = {
            "n": int(arr.size),
            "total_seconds": float(arr.sum()),
            "mean_seconds": float(arr.mean()),
            "std_seconds": float(arr.std(ddof=0)),
        }
    return summary


def save_error_curves(
    outdir: str,
    times: np.ndarray,
    rollout_scores: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    if times.size > 1:
        times_plot = times[1:]
    else:
        times_plot = times

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, (l1, _) in rollout_scores.items():
        mean = l1.mean(axis=0)[1:] if l1.shape[1] > 1 else l1.mean(axis=0)
        std = l1.std(axis=0)[1:] if l1.shape[1] > 1 else l1.std(axis=0)
        ax.plot(times_plot, mean, label=display_time_model_name(name))
        ax.fill_between(times_plot, mean - std, mean + std, alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("t", fontsize=14)
    ax.set_ylabel(r"Relative $L^1$ error vs WENO-Z", fontsize=14)
    ax.legend()
    apply_line_grid(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "relL1_vs_weno_time.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, (_, linf) in rollout_scores.items():
        values = linf.mean(axis=0)[1:] if linf.shape[1] > 1 else linf.mean(axis=0)
        ax.plot(times_plot, values, label=display_time_model_name(name))
    ax.set_yscale("log")
    ax.set_xlabel("t", fontsize=14)
    ax.set_ylabel(r"Relative $L^\infty$ error vs WENO-Z", fontsize=14)
    ax.legend()
    apply_line_grid(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "relLinf_vs_weno_time.pdf"), bbox_inches="tight")
    plt.close(fig)


def merged_field_limit(fields_by_model: dict[str, dict[str, np.ndarray]], key: str) -> tuple[float, float]:
    vals = [fields[key] for fields in fields_by_model.values() if key in fields]
    return min(float(np.min(v)) for v in vals), max(float(np.max(v)) for v in vals)


def contour_levels_from_limit(limits: tuple[float, float], levels: int) -> np.ndarray | None:
    vmin, vmax = float(limits[0]), float(limits[1])
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if abs(vmax - vmin) < 1e-14:
        span = max(abs(vmin), 1.0) * 1e-6
        vmin -= span
        vmax += span
    return np.linspace(vmin, vmax, int(levels))


def save_single_contour(
    out_path: str,
    field: np.ndarray,
    *,
    limits: tuple[float, float],
    levels: int = 18,
) -> None:
    contour_levels = contour_levels_from_limit(limits, levels)
    if contour_levels is None:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.contour(np.asarray(field, dtype=np.float64), levels=contour_levels, colors="black", linewidths=0.55)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
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
    if not names:
        return
    contour_levels = contour_levels_from_limit(merged_field_limit(fields_by_model, key), levels)
    if contour_levels is None:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8), squeeze=False)
    for ax, name in zip(axes[0], names):
        field = np.asarray(fields_by_model[name][key], dtype=np.float64)
        ax.contour(field, levels=contour_levels, colors="black", linewidths=0.55)
        ax.set_aspect("equal")
        ax.set_title(f"{display_time_model_name(name)} {field_math_label(key)}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_contour_plots(
    outdir: str,
    ref_traj: np.ndarray,
    pred_trajs: dict[str, np.ndarray],
    *,
    gamma: float,
    tag: str,
) -> None:
    final_states = {"ref": ref_traj[-1]}
    for name, traj in pred_trajs.items():
        final_states[name] = traj[-1]

    field_groups = {
        "conserved": {name: state_to_conserved_fields(state) for name, state in final_states.items()},
        "primitive": {name: conserved_to_primitive_2d(state, gamma=gamma) for name, state in final_states.items()},
    }
    for group_name, fields_by_model in field_groups.items():
        keys = list(next(iter(fields_by_model.values())).keys())
        for key in keys:
            limits = merged_field_limit(fields_by_model, key)
            save_contour_comparison_panel(
                os.path.join(outdir, f"contour_{group_name}_{key}_compare_{tag}.pdf"),
                fields_by_model,
                key=key,
            )
            for name, fields in fields_by_model.items():
                save_single_contour(
                    os.path.join(outdir, f"contour_{group_name}_{key}_{name}_{tag}.pdf"),
                    fields[key],
                    limits=limits,
                )


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_HYBRID_CKPT)
    ap.add_argument("--ckpt_fno", type=str, default=DEFAULT_FNO_CKPT)
    ap.add_argument("--hybrid_only", action="store_true", help="Evaluate only the primary --ckpt model.")
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--test_name", type=str, default=DEFAULT_TIME_TEST_NAME)
    ap.add_argument("--test_path", type=str, default=None)
    ap.add_argument("--n_samples", type=int, default=1, help="Number of trajectories to time; 0 means all.")
    ap.add_argument("--sample_seed", type=int, default=None, help="Random seed used to choose trajectories.")
    ap.add_argument("--T", type=float, default=None, help="Optional horizon. Defaults to the split metadata T.")
    ap.add_argument("--rollout_steps", type=int, default=0, help="0 means use T/dt or the full metadata horizon.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow_tf32", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--outdir", type=str, default="eval_euler2d_periodic_time_out")
    ap.add_argument("--weno_cfl", type=float, default=None, help="WENO CFL. Defaults to metadata cfl or 0.4.")
    ap.add_argument("--weno_reconstruction", type=str, default=None, choices=("component", "characteristic"))
    ap.add_argument("--weno_line_batch_size", type=int, default=16)
    ap.add_argument("--weno_verbose", action="store_true", help="Print every WENO CFL substep.")
    ap.add_argument("--weno_quiet", action="store_true", help="Disable demo WENO substep printing.")
    ap.add_argument("--warmup_steps", type=int, default=1, help="Rollout steps used to warm up WENO/torch before timing.")
    ap.add_argument("--skip_warmup", action="store_true", help="Disable warmup; mainly for debugging.")
    ap.add_argument("--plot_one", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--share_ref_colorbar", action="store_true")
    ap.add_argument("--demo_kh", action="store_true", help="Run Kelvin-Helmholtz demo IC instead of loading a dataset.")
    ap.add_argument(
        "--demo_riemann",
        type=str,
        default=None,
        choices=sorted(RIEMANN_DEMO_CONFIGS),
        help="Run a quadrant Riemann demo IC instead of loading a dataset.",
    )
    ap.add_argument("--demo_nx", type=int, default=256)
    ap.add_argument("--demo_ny", type=int, default=256)
    ap.add_argument("--demo_weno_fine_nx", type=int, default=512)
    ap.add_argument("--demo_weno_fine_ny", type=int, default=512)
    ap.add_argument("--demo_frame_dt", type=float, default=1e-2)
    ap.add_argument("--demo_T", type=float, default=1.0)
    ap.add_argument("--demo_split_x", type=float, default=None)
    ap.add_argument("--demo_split_y", type=float, default=None)
    ap.add_argument("--demo_seed", type=int, default=0)
    ap.add_argument("--demo_noise_amp", type=float, default=0.005)
    ap.add_argument("--demo_pressure", type=float, default=2.5)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device)
    configure_runtime(device, allow_tf32=args.allow_tf32)
    pin_mem = device.type == "cuda"

    demo_fine_frame: torch.Tensor | None = None
    demo_label: str | None = None
    if args.demo_kh and args.demo_riemann is not None:
        raise ValueError("Use only one of --demo_kh and --demo_riemann.")
    if args.demo_kh or args.demo_riemann is not None:
        demo_label = "kh" if args.demo_kh else str(args.demo_riemann)
        test_path = "kelvin_helmholtz_demo" if args.demo_kh else f"riemann_demo_{demo_label}"
        total_traj = 1
        selected_idx = np.array([0], dtype=np.int64)
        frame_dt = float(args.demo_frame_dt)
        gamma = 1.4
        nx = int(args.demo_nx)
        ny = int(args.demo_ny)
        if nx <= 0 or ny <= 0:
            raise ValueError("--demo_nx and --demo_ny must be positive.")
        if frame_dt <= 0.0:
            raise ValueError("--demo_frame_dt must be positive.")
        if args.demo_weno_fine_nx % nx != 0 or args.demo_weno_fine_ny % ny != 0:
            raise ValueError("Fine WENO grid must be an integer multiple of the demo model grid.")
        dx = 1.0 / nx
        dy = 1.0 / ny
        if args.demo_kh:
            u0_demo = make_kelvin_helmholtz_state(
                nx,
                ny,
                seed=args.demo_seed,
                noise_amp=args.demo_noise_amp,
                pressure=args.demo_pressure,
                gamma=gamma,
            )
            u0_fine_demo = make_kelvin_helmholtz_state(
                int(args.demo_weno_fine_nx),
                int(args.demo_weno_fine_ny),
                seed=args.demo_seed,
                noise_amp=args.demo_noise_amp,
                pressure=args.demo_pressure,
                gamma=gamma,
            )
        else:
            demo_config = RIEMANN_DEMO_CONFIGS[str(args.demo_riemann)]
            split_x = float(args.demo_split_x if args.demo_split_x is not None else demo_config["split_x"])
            split_y = float(args.demo_split_y if args.demo_split_y is not None else demo_config["split_y"])
            raw_demo_states = demo_config["states"]
            validate_riemann_demo_velocity_range(raw_demo_states)
            rhop_scale = fit_riemann_demo_rhop_scale(raw_demo_states)
            demo_states = scale_riemann_demo_states_rhop(raw_demo_states, rhop_scale)
            u0_demo = make_riemann_state(
                nx,
                ny,
                split_x=split_x,
                split_y=split_y,
                states=demo_states,
                gamma=gamma,
            )
            u0_fine_demo = make_riemann_state(
                int(args.demo_weno_fine_nx),
                int(args.demo_weno_fine_ny),
                split_x=split_x,
                split_y=split_y,
                states=demo_states,
                gamma=gamma,
            )
        first_frames = torch.from_numpy(u0_demo[None].astype(np.float32))
        demo_fine_frame = torch.from_numpy(u0_fine_demo.astype(np.float32))
        test_meta = {
            "dt": frame_dt,
            "gamma": gamma,
            "dx": dx,
            "dy": dy,
            "nx": nx,
            "ny": ny,
            "n_snaps": int(round(float(args.demo_T) / frame_dt)) + 1,
            "T": float(args.demo_T),
            "cfl": 0.45,
            "boundary": "periodic",
            "reconstruction": "component",
        }
        if args.demo_riemann is not None:
            test_meta["split_x"] = split_x
            test_meta["split_y"] = split_y
            test_meta["rhop_scale"] = rhop_scale
            test_meta["raw_primitive_ranges"] = primitive_state_ranges(raw_demo_states)
            test_meta["scaled_primitive_ranges"] = primitive_state_ranges(demo_states)
    else:
        test_path = resolve_test_split_path(args)
        split_item = torch_load_cpu(test_path)
        if "meta" not in split_item:
            raise ValueError(f"Missing meta in {test_path}")
        test_meta = dict(split_item["meta"])
        if "trajectory_files" in split_item:
            total_traj = len(split_item["trajectory_files"])
        else:
            total_traj = int(split_item["states"].shape[0])
        del split_item

        selected_idx = select_trajectory_indices(
            total_traj,
            args.n_samples if args.n_samples > 0 else None,
            args.sample_seed,
        )
        first_frames, test_meta = load_first_frames(test_path, selected_idx)

    frame_dt = float(test_meta["dt"])
    gamma = float(test_meta.get("gamma", 1.4))
    dx = float(test_meta["dx"])
    dy = float(test_meta["dy"])
    nx = int(test_meta["nx"])
    ny = int(test_meta["ny"])
    meta_steps = int(test_meta.get("n_snaps", int(round(float(test_meta.get("T", 0.0)) / frame_dt)) + 1)) - 1
    if args.T is not None:
        if args.T < 0.0:
            raise ValueError("--T must be non-negative.")
        horizon_steps = int(np.floor(float(args.T) / frame_dt + 1e-12))
    else:
        horizon_steps = meta_steps
    n_steps = horizon_steps if args.rollout_steps <= 0 else min(int(args.rollout_steps), horizon_steps)
    if n_steps < 1:
        raise ValueError(f"Need at least one rollout step, got n_steps={n_steps}")

    weno_cfl = float(test_meta.get("cfl", 0.4) if args.weno_cfl is None else args.weno_cfl)
    reconstruction = args.weno_reconstruction or str(test_meta.get("reconstruction", "component"))
    print(
        f"[data] test_path={test_path}, n_traj={len(selected_idx)}/{total_traj}, "
        f"n_steps={n_steps}, frame_dt={frame_dt}, T={n_steps * frame_dt:g}, nx={nx}, ny={ny}, "
        f"boundary={test_meta.get('boundary', '?')}, sample_seed={args.sample_seed}"
    )
    print(
        f"[weno] reconstruction={reconstruction}, WENOtype=WENO-Z, cfl={weno_cfl}, "
        f"frame_dt={frame_dt}, line_batch_size={args.weno_line_batch_size}"
    )
    if args.demo_kh:
        print(
            f"[demo] Kelvin-Helmholtz: model_grid={nx}x{ny}, "
            f"weno_fine_grid={int(args.demo_weno_fine_nx)}x{int(args.demo_weno_fine_ny)}, "
            f"pressure={args.demo_pressure}, noise_amp={args.demo_noise_amp}, seed={args.demo_seed}"
        )
    elif args.demo_riemann is not None:
        demo_config = RIEMANN_DEMO_CONFIGS[str(args.demo_riemann)]
        print(
            f"[demo] Riemann {args.demo_riemann}: model_grid={nx}x{ny}, "
            f"weno_fine_grid={int(args.demo_weno_fine_nx)}x{int(args.demo_weno_fine_ny)}, "
            f"split=({float(test_meta.get('split_x', 0.5)):g},{float(test_meta.get('split_y', 0.5)):g}), "
            f"states are primitive rho,u,v,p and model/WENO inputs are conserved. "
            f"{demo_config['description']}"
        )
        print(
            f"[demo] Riemann scale: rhop_scale={float(test_meta.get('rhop_scale', 1.0)):.8g}, "
            f"raw_ranges={test_meta.get('raw_primitive_ranges')}, "
            f"scaled_ranges={test_meta.get('scaled_primitive_ranges')}"
        )

    models: dict[str, tuple[torch.nn.Module, dict]] = {}
    model_paths: dict[str, str] = {}
    primary_model, primary_ckpt, primary_path = load_model_from_ckpt(args.ckpt, device, no_compile=args.no_compile)
    primary_name = model_label_from_ckpt(primary_ckpt)
    models[primary_name] = (primary_model, primary_ckpt)
    model_paths[primary_name] = primary_path

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

    for name, (model, ckpt) in sorted(models.items()):
        print(
            f"[model] {name}: params={count_params(model):,}, path={model_paths[name]}, "
            f"bc={ckpt.get('bc', ckpt.get('args', {}).get('bc', '?'))}"
        )

    is_demo = demo_label is not None
    weno_verbose = bool(args.weno_verbose or (is_demo and not args.weno_quiet))
    weno_op = build_euler2d_solver(
        gamma=gamma,
        bc=str(test_meta.get("boundary", "periodic")),
        reconstruction=reconstruction,
        WENOtype="WENO-Z",
        verbose=weno_verbose,
        line_batch_size=args.weno_line_batch_size,
    )
    weno_fine_op = None
    if is_demo:
        weno_fine_op = build_euler2d_solver(
            gamma=gamma,
            bc="periodic",
            reconstruction=reconstruction,
            WENOtype="WENO-Z",
            verbose=weno_verbose,
            line_batch_size=args.weno_line_batch_size,
        )

    warmup_steps = 0 if args.skip_warmup else min(max(int(args.warmup_steps), 0), n_steps)
    if warmup_steps > 0:
        print(
            f"[warmup] each method runs {warmup_steps} warmup step(s) immediately before its timed rollout",
            flush=True,
        )

    times = np.arange(n_steps + 1, dtype=np.float64) * frame_dt
    score_names = list(models.keys())
    if is_demo:
        score_names = ["weno256", *score_names]
    rollout_scores = {
        name: (
            np.zeros((len(selected_idx), n_steps + 1), dtype=np.float64),
            np.zeros((len(selected_idx), n_steps + 1), dtype=np.float64),
        )
        for name in score_names
    }
    runtime_rows: list[dict[str, object]] = []
    first_ref = None
    first_preds: dict[str, np.ndarray] = {}

    for order, sample_idx in enumerate(selected_idx):
        print(f"[sample] {order + 1}/{len(selected_idx)} idx={int(sample_idx)}")
        u0 = first_frames[order]
        if is_demo:
            weno256_traj, weno256_seconds = timed_weno_rollout(
                "weno256",
                weno_op,
                u0,
                dx=dx,
                dy=dy,
                dt_snap=frame_dt,
                n_steps=n_steps,
                cfl=weno_cfl,
                gamma=gamma,
                warmup_steps=warmup_steps,
            )
            append_runtime_row(
                runtime_rows,
                sample_order=order,
                sample_idx=demo_label or int(sample_idx),
                method="weno256",
                seconds=weno256_seconds,
                n_steps=n_steps,
                frame_dt=frame_dt,
                weno_cfl=weno_cfl,
                nx=nx,
                ny=ny,
                device="cpu",
            )
            if demo_fine_frame is None or weno_fine_op is None:
                raise RuntimeError("Demo fine WENO state/operator was not initialized.")
            fine_nx = int(args.demo_weno_fine_nx)
            fine_ny = int(args.demo_weno_fine_ny)
            fine_dx = 1.0 / float(fine_nx)
            fine_dy = 1.0 / float(fine_ny)
            weno512_traj, weno512_seconds = timed_weno_rollout(
                "weno512",
                weno_fine_op,
                demo_fine_frame,
                dx=fine_dx,
                dy=fine_dy,
                dt_snap=frame_dt,
                n_steps=n_steps,
                cfl=weno_cfl,
                gamma=gamma,
                warmup_steps=warmup_steps,
            )
            append_runtime_row(
                runtime_rows,
                sample_order=order,
                sample_idx=demo_label or int(sample_idx),
                method="weno512",
                seconds=weno512_seconds,
                n_steps=n_steps,
                frame_dt=frame_dt,
                weno_cfl=weno_cfl,
                nx=fine_nx,
                ny=fine_ny,
                device="cpu",
            )
            factor_y = fine_ny // ny
            factor_x = fine_nx // nx
            ref_traj = downsample_trajectory_conserved2d(weno512_traj, factor_y, factor_x)
            l1, linf = rollout_scores["weno256"]
            for j in range(n_steps + 1):
                l1[order, j] = rel_l1(weno256_traj[j], ref_traj[j])
                linf[order, j] = rel_linf(weno256_traj[j], ref_traj[j])
            if order == 0:
                first_preds["weno256"] = weno256_traj
        else:
            ref_traj, weno_seconds = timed_weno_rollout(
                "weno",
                weno_op,
                u0,
                dx=dx,
                dy=dy,
                dt_snap=frame_dt,
                n_steps=n_steps,
                cfl=weno_cfl,
                gamma=gamma,
                warmup_steps=warmup_steps,
            )
            append_runtime_row(
                runtime_rows,
                sample_order=order,
                sample_idx=int(sample_idx),
                method="weno",
                seconds=weno_seconds,
                n_steps=n_steps,
                frame_dt=frame_dt,
                weno_cfl=weno_cfl,
                nx=nx,
                ny=ny,
                device="cpu",
            )

        if order == 0:
            first_ref = ref_traj

        for name, (model, _) in models.items():
            pred_traj, seconds = timed_model_rollout(
                name,
                model,
                u0,
                n_steps=n_steps,
                dt=frame_dt,
                device=device,
                pin_memory=pin_mem,
                warmup_steps=warmup_steps,
            )
            runtime_rows.append(
                {
                    "sample_order": order,
                    "sample_idx": demo_label or int(sample_idx),
                    "method": name,
                    "seconds": seconds,
                    "n_steps": n_steps,
                    "frame_dt": frame_dt,
                    "weno_cfl": weno_cfl,
                    "nx": nx,
                    "ny": ny,
                    "device": str(device),
                }
            )
            l1, linf = rollout_scores[name]
            for j in range(n_steps + 1):
                l1[order, j] = rel_l1(pred_traj[j], ref_traj[j])
                linf[order, j] = rel_linf(pred_traj[j], ref_traj[j])
            if order == 0:
                first_preds[name] = pred_traj

        gc.collect()

    summary = summarize_timings(runtime_rows)
    save_runtime_csvs(args.outdir, runtime_rows, summary)
    save_runtime_bar(args.outdir, summary)

    npz_payload: dict[str, np.ndarray] = {"times": times, "dt": np.array([frame_dt], dtype=np.float64)}
    for name, (l1, linf) in rollout_scores.items():
        npz_payload[f"l1_{name}"] = l1
        npz_payload[f"linf_{name}"] = linf
        print(
            f"[error] {name}: final relL1={l1[:, -1].mean():.3e}, "
            f"final relLinf={linf[:, -1].mean():.3e}"
        )
    np.savez_compressed(os.path.join(args.outdir, "rollout_metrics_vs_weno.npz"), **npz_payload)
    save_error_curves(args.outdir, times, rollout_scores)

    if args.plot_one and first_ref is not None and first_preds:
        if is_demo:
            plot_tag = f"demo_{demo_label}_steps{n_steps}_weno_ref"
        else:
            plot_tag = f"{sample_tag(int(selected_idx[0]), args.sample_seed)}_steps{n_steps}_weno_ref"
        plot_keys = ("weno256", "hybrid", "fno") if is_demo else ("hybrid", "fno")
        save_density_plots(
            args.outdir,
            first_ref,
            {k: v for k, v in first_preds.items() if k in plot_keys},
            final_time=float(times[-1]),
            gamma=gamma,
            tag=plot_tag,
            share_ref_colorbar=args.share_ref_colorbar,
        )
        save_contour_plots(
            args.outdir,
            first_ref,
            {k: v for k, v in first_preds.items() if k in plot_keys},
            gamma=gamma,
            tag=plot_tag,
        )

    models.clear()
    release_eval_memory(device)
    print(f"Saved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
