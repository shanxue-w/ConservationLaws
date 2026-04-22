from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


torch.set_default_dtype(torch.float32)


DEFAULT_TRAIN_NAME = "euler2d_quadrant_periodic_train.pt"
DEFAULT_VAL_NAME = "euler2d_quadrant_periodic_val.pt"
DEFAULT_TEST_NAME = "euler2d_quadrant_periodic_test.pt"


def default_num_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def default_dt_ckpt_path(model_name: str, bc: str) -> str:
    model_key = str(model_name).lower()
    bc_key = normalize_bc_name(bc)
    if model_key not in ("hybrid", "fno", "cnn"):
        raise ValueError(f"Unsupported Euler2D dt-step model name: {model_name}")
    if bc_key not in ("periodic", "outflow"):
        raise ValueError(f"Unsupported Euler2D boundary condition: {bc}")
    return f"checkpoints/euler2d_{model_key}_{bc_key}_dt.pt"


def torch_load_cpu(path: str | os.PathLike[str]) -> dict:
    return torch.load(path, map_location="cpu")


def configure_runtime(device: torch.device, allow_tf32: bool = True) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    if allow_tf32:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def normalize_bc_name(bc: str | None) -> str:
    bc = "periodic" if bc is None else str(bc).lower()
    if bc in ("reflect", "transmissive"):
        return "outflow"
    if bc not in ("periodic", "outflow", "zero"):
        return "periodic"
    return bc


def resolve_model_bc(meta: dict, bc_override: str | None) -> str:
    if bc_override is not None:
        bc = normalize_bc_name(bc_override)
        if bc not in ("periodic", "outflow"):
            raise ValueError("model bc must be periodic or outflow")
        return bc
    return normalize_bc_name(meta.get("boundary", meta.get("bc", "periodic")))


def resolve_zero_mean_rhs(
    model_bc: str,
    *,
    enable: bool = False,
    disable: bool = False,
) -> bool:
    if enable and disable:
        raise ValueError("Use only one of enable/disable zero_mean_rhs.")
    if enable:
        return True
    if disable:
        return False
    return model_bc == "periodic"


def resolve_split_path(data_dir: str, file_name: str) -> str:
    path = os.path.join(data_dir, file_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


def _normalize_single_traj_states(states: torch.Tensor, src_path: str) -> torch.Tensor:
    if not torch.is_tensor(states):
        raise ValueError(f"`states` is not a tensor in {src_path}")
    if states.ndim == 4:
        return states
    if states.ndim == 5 and int(states.shape[0]) == 1:
        return states[0]
    raise ValueError(
        f"Unsupported trajectory states shape in {src_path}: expected (n_snaps, 4, ny, nx) "
        f"or (1, n_snaps, 4, ny, nx), got {tuple(states.shape)}"
    )


def load_trajectory_split(split_path: str, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, dict]:
    item = torch_load_cpu(split_path)
    if "states" in item:
        states = item["states"]
        if not torch.is_tensor(states):
            raise ValueError(f"`states` is not a tensor in {split_path}")
        if states.ndim != 5:
            raise ValueError(
                f"Expected split states with shape (n_traj, n_snaps, 4, ny, nx), got {tuple(states.shape)}"
            )
        if states.dtype != dtype:
            states = states.to(dtype)
        meta = item["meta"]
        del item
        return states, dict(meta)

    if "trajectory_files" not in item:
        raise ValueError(f"Unsupported split format in {split_path}: keys={list(item.keys())}")

    base_dir = os.path.dirname(split_path)
    traj_states = []
    for rel_path in item["trajectory_files"]:
        traj_path = os.path.join(base_dir, rel_path)
        traj_item = torch_load_cpu(traj_path)
        traj_states.append(_normalize_single_traj_states(traj_item["states"], traj_path))
    if not traj_states:
        raise ValueError(f"No trajectories found in manifest: {split_path}")
    return torch.stack(traj_states, dim=0), dict(item["meta"])


def states_to_one_step_pairs(
    states: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    max_pairs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if states.ndim != 5:
        raise ValueError(f"Expected states with shape (n_traj, n_snaps, c, ny, nx), got {tuple(states.shape)}")
    if states.dtype != dtype:
        states = states.to(dtype)
    x = states[:, :-1]
    y = states[:, 1:]
    if x.size(1) <= 0:
        raise ValueError("Need at least two snapshots per trajectory to build one-step pairs.")
    x = x.reshape(-1, *x.shape[2:]).permute(0, 2, 3, 1).contiguous()
    y = y.reshape(-1, *y.shape[2:]).permute(0, 2, 3, 1).contiguous()
    if max_pairs is not None and max_pairs > 0:
        x = x[: int(max_pairs)]
        y = y[: int(max_pairs)]
    return x, y


def make_pair_loaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    *,
    dtype: torch.dtype = torch.float32,
    train_max_pairs: int | None = None,
    val_max_pairs: int | None = None,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> tuple[DataLoader, DataLoader, dict]:
    train_states, train_meta = load_trajectory_split(train_path, dtype=dtype)
    n_train_trajectories = int(train_states.shape[0])
    n_snaps_train = int(train_states.shape[1])
    u0_train, u1_train = states_to_one_step_pairs(train_states, dtype=dtype, max_pairs=train_max_pairs)
    del train_states

    val_states, val_meta = load_trajectory_split(val_path, dtype=dtype)
    n_val_trajectories = int(val_states.shape[0])
    n_snaps_val = int(val_states.shape[1])
    u0_val, u1_val = states_to_one_step_pairs(val_states, dtype=dtype, max_pairs=val_max_pairs)
    del val_states
    loader_kw = dict(
        num_workers=max(0, int(num_workers)),
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers and num_workers > 0),
    )
    train_loader = DataLoader(
        TensorDataset(u0_train, u1_train),
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        TensorDataset(u0_val, u1_val),
        batch_size=batch_size,
        shuffle=False,
        **loader_kw,
    )
    info = {
        "train_meta": train_meta,
        "val_meta": val_meta,
        "dt": float(train_meta["dt"]),
        "dx": float(train_meta["dx"]),
        "dy": float(train_meta["dy"]),
        "nx": int(train_meta["nx"]),
        "ny": int(train_meta["ny"]),
        "n_train_pairs": int(u0_train.shape[0]),
        "n_val_pairs": int(u0_val.shape[0]),
        "n_train_trajectories": n_train_trajectories,
        "n_val_trajectories": n_val_trajectories,
        "n_snaps_train": n_snaps_train,
        "n_snaps_val": n_snaps_val,
    }
    return train_loader, val_loader, info


def periodic_laplacian_sq_2d(z: torch.Tensor) -> torch.Tensor:
    zm_x = torch.roll(z, 1, dims=2)
    zp_x = torch.roll(z, -1, dims=2)
    zm_y = torch.roll(z, 1, dims=1)
    zp_y = torch.roll(z, -1, dims=1)
    lap = zp_x + zm_x + zp_y + zm_y - 4.0 * z
    return (lap**2).mean()


def outflow_laplacian_sq_2d(z: torch.Tensor) -> torch.Tensor:
    zm_x = torch.cat([z[:, :, :1], z[:, :, :-1]], dim=2)
    zp_x = torch.cat([z[:, :, 1:], z[:, :, -1:]], dim=2)
    zm_y = torch.cat([z[:, :1], z[:, :-1]], dim=1)
    zp_y = torch.cat([z[:, 1:], z[:, -1:]], dim=1)
    lap = zp_x + zm_x + zp_y + zm_y - 4.0 * z
    return (lap**2).mean()


def high_freq_error_loss_2d(pred: torch.Tensor, target: torch.Tensor, k_frac: float = 0.25) -> torch.Tensor:
    if k_frac <= 0.0:
        return pred.new_tensor(0.0)
    err = (pred - target).permute(0, 3, 1, 2).contiguous()
    err_ft = torch.fft.rfftn(err, dim=(2, 3))
    h, w_half = err_ft.shape[2], err_ft.shape[3]
    ky0 = int(h * k_frac)
    kx0 = int(w_half * k_frac)
    if ky0 >= h and kx0 >= w_half:
        return pred.new_tensor(0.0)
    mask_y = torch.arange(h, device=pred.device)[:, None] >= ky0
    mask_x = torch.arange(w_half, device=pred.device)[None, :] >= kx0
    mask = mask_y | mask_x
    return ((err_ft * mask[None, None]).abs() ** 2).mean()


def rel_l1_loss_per_channel_2d(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    abs_err = (pred - target).abs().mean(dim=(1, 2))
    abs_tgt = target.abs().mean(dim=(1, 2))
    rel = abs_err / (abs_tgt + eps)
    return rel.mean()


def total_loss_batch_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    spec_mu: float,
    spec_kfrac: float,
    lap_mu: float,
    model_bc: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_l1 = rel_l1_loss_per_channel_2d(pred, target)
    loss_spec = high_freq_error_loss_2d(pred, target, spec_kfrac) if spec_mu > 0.0 else pred.new_tensor(0.0)
    if lap_mu > 0.0:
        err = (pred - target).abs().mean(dim=-1)
        lap_fn = periodic_laplacian_sq_2d if model_bc == "periodic" else outflow_laplacian_sq_2d
        loss_lap = lap_fn(err)
    else:
        loss_lap = pred.new_tensor(0.0)
    loss = loss_l1 + spec_mu * loss_spec + lap_mu * loss_lap
    return loss, loss_l1, loss_spec, loss_lap


@torch.no_grad()
def evaluate_step_model(
    model: torch.nn.Module,
    loader: DataLoader,
    dt: float,
    device: torch.device,
    *,
    spec_mu: float,
    spec_kfrac: float,
    lap_mu: float,
    pin_memory: bool,
    model_bc: str,
) -> dict[str, float]:
    model.eval()
    tot = torch.zeros((), device=device)
    l1 = torch.zeros((), device=device)
    spec = torch.zeros((), device=device)
    lap = torch.zeros((), device=device)
    n = 0
    for u0, u1 in loader:
        u0 = u0.to(device, non_blocking=pin_memory)
        u1 = u1.to(device, non_blocking=pin_memory)
        dt_b = torch.full((u0.size(0),), float(dt), device=device, dtype=u0.dtype)
        pred = model(u0, dt_b)
        loss, loss_l1, loss_spec, loss_lap = total_loss_batch_2d(
            pred,
            u1,
            spec_mu=spec_mu,
            spec_kfrac=spec_kfrac,
            lap_mu=lap_mu,
            model_bc=model_bc,
        )
        tot += loss.detach()
        l1 += loss_l1.detach()
        spec += loss_spec.detach()
        lap += loss_lap.detach()
        n += 1
    m = max(n, 1)
    tot_v = (tot / m).item()
    l1_v = (l1 / m).item()
    spec_v = (spec / m).item()
    lap_v = (lap / m).item()
    return {
        "loss": tot_v,
        "loss_l1": l1_v,
        "loss_spec": spec_v,
        "loss_lap": lap_v,
        "spec_w": spec_mu * spec_v,
        "lap_w": lap_mu * lap_v,
    }


@torch.no_grad()
def rollout_step_model(
    model: torch.nn.Module,
    u0: torch.Tensor,
    n_steps: int,
    dt: float,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
    pin_memory: bool = False,
) -> np.ndarray:
    model.eval()
    if u0.ndim != 3:
        raise ValueError(f"Expected initial state (4, ny, nx), got {tuple(u0.shape)}")
    u = u0.permute(1, 2, 0).unsqueeze(0).contiguous()
    if pin_memory and device.type == "cuda":
        u = u.pin_memory()
    u = u.to(device=device, dtype=dtype, non_blocking=pin_memory and device.type == "cuda")
    traj = [u0.cpu().numpy()]
    for _ in range(n_steps):
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        dt_b = torch.full((u.size(0),), float(dt), device=device, dtype=dtype)
        u = model(u, dt_b).clone()
        traj.append(u.squeeze(0).permute(2, 0, 1).detach().cpu().numpy())
    return np.stack(traj, axis=0)


@torch.no_grad()
def predict_pairs(
    model: torch.nn.Module,
    u0: torch.Tensor,
    dt: float,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 32,
    pin_memory: bool = False,
) -> np.ndarray:
    preds = []
    model.eval()
    for start in range(0, u0.size(0), batch_size):
        batch = u0[start : start + batch_size]
        if pin_memory and device.type == "cuda":
            batch = batch.pin_memory()
        batch = batch.to(device=device, dtype=dtype, non_blocking=pin_memory and device.type == "cuda")
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        dt_b = torch.full((batch.size(0),), float(dt), device=device, dtype=dtype)
        preds.append(model(batch, dt_b).cpu().numpy())
    return np.concatenate(preds, axis=0)


def select_trajectories(states: torch.Tensor, n_samples: int | None) -> torch.Tensor:
    if n_samples is None or n_samples <= 0:
        return states
    return states[: int(n_samples)]


def iter_model_rows(metrics: dict[str, tuple[np.ndarray, np.ndarray]]) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    for name, values in metrics.items():
        yield name, values[0], values[1]
