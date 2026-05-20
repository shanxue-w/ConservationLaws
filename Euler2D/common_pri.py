from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import torch
import torch._dynamo as dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from conslaw.checkpoints import compile_if_requested
from conslaw.models import (
    CNNDtStep2d,
    FNODtStep2d,
    HybridBackbone2dOutflow,
    HybridDtStep2d,
    maybe_torch_compile,
)

from common import (
    DEFAULT_TEST_NAME,
    DEFAULT_TRAIN_NAME,
    DEFAULT_VAL_NAME,
    configure_runtime,
    default_num_workers,
    high_freq_error_loss_2d,
    normalize_bc_name,
    outflow_laplacian_sq_2d,
    periodic_laplacian_sq_2d,
    resolve_model_bc,
    resolve_split_path,
    torch_load_cpu,
)


PRIMITIVE_CHANNELS = ("rho", "u", "v", "p")
RHO_FLOOR_DEFAULT = 1e-6
P_FLOOR_DEFAULT = 1e-8
POSITIVE_BETA_DEFAULT = 100.0


def _channel_dim(x: torch.Tensor) -> int:
    if x.ndim >= 3 and int(x.shape[-3]) == 4:
        return -3
    if x.ndim >= 1 and int(x.shape[-1]) == 4:
        return -1
    raise ValueError(f"Expected a 4-channel Euler2D tensor, got {tuple(x.shape)}")


def conserved_to_primitive_tensor(cons: torch.Tensor, gamma: float = 1.4, eps: float = 1e-12) -> torch.Tensor:
    cdim = _channel_dim(cons)
    rho, rhou, rhov, energy = torch.unbind(cons, dim=cdim)
    rho_safe = rho.clamp_min(float(eps))
    u = rhou / rho_safe
    v = rhov / rho_safe
    kinetic = 0.5 * (rhou * rhou + rhov * rhov) / rho_safe
    p = (float(gamma) - 1.0) * (energy - kinetic)
    return torch.stack((rho, u, v, p), dim=cdim)


def primitive_to_conserved_tensor(prim: torch.Tensor, gamma: float = 1.4, eps: float = 1e-12) -> torch.Tensor:
    cdim = _channel_dim(prim)
    rho, u, v, p = torch.unbind(prim, dim=cdim)
    rho_safe = rho.clamp_min(float(eps))
    p_safe = p.clamp_min(float(eps))
    rhou = rho_safe * u
    rhov = rho_safe * v
    energy = p_safe / (float(gamma) - 1.0) + 0.5 * rho_safe * (u * u + v * v)
    return torch.stack((rho_safe, rhou, rhov, energy), dim=cdim)


def primitive_to_conserved_numpy(prim: np.ndarray, gamma: float = 1.4, eps: float = 1e-12) -> np.ndarray:
    rho = np.maximum(np.asarray(prim[..., 0] if prim.shape[-1] == 4 else prim[0], dtype=np.float64), eps)
    if prim.shape[-1] == 4:
        u = np.asarray(prim[..., 1], dtype=np.float64)
        v = np.asarray(prim[..., 2], dtype=np.float64)
        p = np.maximum(np.asarray(prim[..., 3], dtype=np.float64), eps)
        rhou = rho * u
        rhov = rho * v
        energy = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
        return np.stack((rho, rhou, rhov, energy), axis=-1)
    u = np.asarray(prim[1], dtype=np.float64)
    v = np.asarray(prim[2], dtype=np.float64)
    p = np.maximum(np.asarray(prim[3], dtype=np.float64), eps)
    rhou = rho * u
    rhov = rho * v
    energy = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return np.stack((rho, rhou, rhov, energy), axis=0)


def conserved_to_primitive_numpy(cons: np.ndarray, gamma: float = 1.4, eps: float = 1e-12) -> np.ndarray:
    rho = np.asarray(cons[0], dtype=np.float64)
    rho_safe = np.maximum(rho, eps)
    rhou = np.asarray(cons[1], dtype=np.float64)
    rhov = np.asarray(cons[2], dtype=np.float64)
    energy = np.asarray(cons[3], dtype=np.float64)
    u = rhou / rho_safe
    v = rhov / rho_safe
    p = (gamma - 1.0) * (energy - 0.5 * (rhou * rhou + rhov * rhov) / rho_safe)
    return np.stack((rho, u, v, p), axis=0)


def smooth_positive(x: torch.Tensor, floor: float, beta: float) -> torch.Tensor:
    # Strictly positive and smooth. This is not exactly identity near zero:
    # small physical rho/p values require negative raw outputs, which is the
    # unavoidable tradeoff for a smooth map from R to (floor, inf).
    return float(floor) + F.softplus(x - float(floor), beta=float(beta))


class PositivePrimitiveDtStep2d(nn.Module):
    """Wrap a dt-step model and smoothly enforce rho,p positivity on primitive outputs."""

    def __init__(
        self,
        base: nn.Module,
        *,
        rho_floor: float = RHO_FLOOR_DEFAULT,
        p_floor: float = P_FLOOR_DEFAULT,
        positive_beta: float = POSITIVE_BETA_DEFAULT,
    ):
        super().__init__()
        self.base = base
        self.rho_floor = float(rho_floor)
        self.p_floor = float(p_floor)
        self.positive_beta = float(positive_beta)

    def _modify(self, y: torch.Tensor) -> torch.Tensor:
        rho = smooth_positive(y[..., 0], self.rho_floor, self.positive_beta)
        u = y[..., 1]
        v = y[..., 2]
        p = smooth_positive(y[..., 3], self.p_floor, self.positive_beta)
        return torch.stack((rho, u, v, p), dim=-1)

    def forward(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return self._modify(self.base(u, dt))

    @dynamo.disable
    def forward_with_aux(self, u: torch.Tensor, dt: torch.Tensor):
        try:
            out = self.base(u, dt, return_aux=True)
        except TypeError:
            out = self.base(u, dt)
        if isinstance(out, tuple):
            y, aux = out
            return self._modify(y), aux
        return self._modify(out), {}


def load_trajectory_split(split_path: str, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, dict]:
    from common import load_trajectory_split as load_conserved_split

    states, meta = load_conserved_split(split_path, dtype=dtype)
    gamma = float(meta.get("gamma", 1.4))
    return conserved_to_primitive_tensor(states, gamma=gamma).to(dtype), meta


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
    val_loader = DataLoader(TensorDataset(u0_val, u1_val), batch_size=batch_size, shuffle=False, **loader_kw)
    info = {
        "train_meta": train_meta,
        "val_meta": val_meta,
        "dt": float(train_meta["dt"]),
        "dx": float(train_meta["dx"]),
        "dy": float(train_meta["dy"]),
        "nx": int(train_meta["nx"]),
        "ny": int(train_meta["ny"]),
        "gamma": float(train_meta.get("gamma", 1.4)),
        "n_train_pairs": int(u0_train.shape[0]),
        "n_val_pairs": int(u0_val.shape[0]),
        "n_train_trajectories": n_train_trajectories,
        "n_val_trajectories": n_val_trajectories,
        "n_snaps_train": n_snaps_train,
        "n_snaps_val": n_snaps_val,
    }
    return train_loader, val_loader, info


def rel_l1_loss_per_channel_2d(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    abs_err = (pred - target).abs().mean(dim=(1, 2))
    abs_tgt = target.abs().mean(dim=(1, 2))
    return (abs_err / (abs_tgt + eps)).mean()


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
    return loss_l1 + spec_mu * loss_spec + lap_mu * loss_lap, loss_l1, loss_spec, loss_lap


@torch.no_grad()
def evaluate_step_model(
    model: nn.Module,
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
    tot = l1 = spec = lap = 0.0
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
        tot += loss.item()
        l1 += loss_l1.item()
        spec += loss_spec.item()
        lap += loss_lap.item()
        n += 1
    m = max(n, 1)
    return {
        "loss": tot / m,
        "loss_l1": l1 / m,
        "loss_spec": spec / m,
        "loss_lap": lap / m,
        "spec_w": spec_mu * spec / m,
        "lap_w": lap_mu * lap / m,
    }


def build_hybrid_model_from_args(args: dict, *, dx: float, dy: float) -> PositivePrimitiveDtStep2d:
    modes = int(args.get("modes", 24))
    modes2 = int(args.get("modes2", modes))
    bc = normalize_bc_name(args.get("bc", "outflow"))
    width = int(args.get("width", 64))
    n_layers = int(args.get("n_layers", 4))
    mr_kernel = int(args.get("mr_kernel", 7))
    spectral_pad = int(args.get("spectral_pad", 4))
    octx_raw = int(args.get("outflow_ctx_width", 0) or 0)
    backbone = None
    if str(args.get("backbone2d", "outflow")) == "outflow":
        backbone = HybridBackbone2dOutflow(
            width=width,
            n_layers=n_layers,
            modes1=modes,
            modes2=modes2,
            mr_kernel=mr_kernel,
            in_channels=4,
            out_channels=4,
            bc=bc,
            spectral_pad=spectral_pad,
            outflow_ctx_width=octx_raw if octx_raw > 0 else None,
        )
    base = HybridDtStep2d(
        width=width,
        n_layers=n_layers,
        modes1=modes,
        modes2=modes2,
        mr_kernel=mr_kernel,
        in_channels=4,
        out_channels=4,
        bc=bc,
        dx=dx,
        dy=dy,
        spectral_pad=spectral_pad,
        zero_mean_rhs=bool(args.get("zero_mean_rhs", bc == "periodic")),
        project_outflow_rhs=bool(args.get("project_outflow_rhs", False)),
        backbone=backbone,
    )
    return PositivePrimitiveDtStep2d(
        base,
        rho_floor=float(args.get("rho_floor", RHO_FLOOR_DEFAULT)),
        p_floor=float(args.get("p_floor", P_FLOOR_DEFAULT)),
        positive_beta=float(args.get("positive_beta", POSITIVE_BETA_DEFAULT)),
    )


def build_baseline_model_from_args(args: dict) -> PositivePrimitiveDtStep2d:
    modes = int(args.get("modes", 32))
    modes2 = int(args.get("modes2", modes))
    bc = normalize_bc_name(args.get("bc", "periodic"))
    if str(args.get("backbone", "fno")) == "cnn":
        base = CNNDtStep2d(
            width=int(args.get("width", 64)),
            n_layers=int(args.get("n_layers", 4)),
            kernel_size=int(args.get("kernel_size", 7)),
            in_channels=4,
            out_channels=4,
            bc=bc,
            zero_mean_rhs=bool(args.get("zero_mean_rhs", bc == "periodic")),
        )
    else:
        base = FNODtStep2d(
            width=int(args.get("width", 64)),
            n_layers=int(args.get("n_layers", 4)),
            modes1=modes,
            modes2=modes2,
            in_channels=4,
            out_channels=4,
            bc=bc,
            padding=int(args.get("padding", args.get("spectral_pad", 4))),
            zero_mean_rhs=bool(args.get("zero_mean_rhs", bc == "periodic")),
        )
    return PositivePrimitiveDtStep2d(
        base,
        rho_floor=float(args.get("rho_floor", RHO_FLOOR_DEFAULT)),
        p_floor=float(args.get("p_floor", P_FLOOR_DEFAULT)),
        positive_beta=float(args.get("positive_beta", POSITIVE_BETA_DEFAULT)),
    )


def load_primitive_dt_step_2d(
    path: str,
    device: torch.device | str = "cpu",
    *,
    no_compile: bool = False,
) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device)
    args = dict(ckpt.get("args", {}))
    kind = str(ckpt.get("kind", ""))
    if "fno" in kind or "cnn" in kind:
        model = build_baseline_model_from_args(args)
    else:
        model = build_hybrid_model_from_args(args, dx=float(ckpt.get("dx", args.get("dx", 1.0))), dy=float(ckpt.get("dy", args.get("dy", 1.0))))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model = compile_if_requested(model, torch.device(device), no_compile=no_compile)
    return model, ckpt


@torch.no_grad()
def rollout_step_model(
    model: nn.Module,
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
        raise ValueError(f"Expected initial primitive state (4, ny, nx), got {tuple(u0.shape)}")
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
    model: nn.Module,
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
