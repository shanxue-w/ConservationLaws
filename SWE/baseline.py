"""
SWE baseline: FNO trunk (:class:`FNOFluxBackbone1d`, modes default 32), same map as train.py (n_cons=2).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from conslaw.models import (
    FNOFluxBackbone1d,
    HybridFixedStepMap1d,
    count_params,
    maybe_torch_compile,
    state_dict_for_ckpt,
)

torch.set_default_dtype(torch.float64)


def solver_bc_from_meta(meta: dict) -> str:
    bc = meta.get("boundary", meta.get("bc", "periodic"))
    if bc in ("reflect", "transmissive"):
        return "outflow"
    return bc if bc in ("periodic", "outflow", "zero", "reflect") else "periodic"


def model_bc_from_meta_or_arg(meta: dict, bc_override: str | None) -> str:
    if bc_override is not None:
        if bc_override not in ("periodic", "outflow"):
            raise ValueError("--bc must be periodic or outflow")
        return bc_override
    b = meta.get("boundary", "periodic")
    if b in ("reflect", "transmissive", "outflow"):
        return "outflow"
    return "periodic"


def load_conserved_pairs(
    train_pt_path,
    val_pt_path,
    batch_size,
    shuffle_train=True,
    *,
    num_workers: int = 4,
    pin_memory: bool = False,
):
    train_data = torch.load(train_pt_path)
    val_data = torch.load(val_pt_path)
    u0_train = train_data["input"].permute(0, 2, 1).to(torch.float64)
    u1_train = train_data["output"].permute(0, 2, 1).to(torch.float64)
    u0_val = val_data["input"].permute(0, 2, 1).to(torch.float64)
    u1_val = val_data["output"].permute(0, 2, 1).to(torch.float64)
    meta = train_data["meta"]
    dx = float(meta["dx"])
    N = int(meta["nx"])
    dt = float(meta["dt"])
    nw = int(max(0, num_workers))
    pw = nw > 0

    train_kw: dict = dict(
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=pin_memory,
        persistent_workers=pw,
    )
    val_kw: dict = dict(batch_size=batch_size, num_workers=nw, pin_memory=pin_memory, persistent_workers=pw)
    if nw > 0:
        train_kw["prefetch_factor"] = 4
        val_kw["prefetch_factor"] = 4

    train_loader = DataLoader(
        TensorDataset(u0_train, u1_train),
        shuffle=shuffle_train,
        drop_last=True,
        **train_kw,
    )
    val_loader = DataLoader(
        TensorDataset(u0_val, u1_val),
        shuffle=False,
        **val_kw,
    )
    return train_loader, val_loader, dx, N, dt, meta


def rel_l1_loss_per_channel(pred, target, eps: float = 1e-12):
    err = (pred - target).abs()
    mean_abs_err = err.mean(dim=(0, 1))
    mean_abs_tgt = target.abs().mean(dim=(0, 1))
    return (mean_abs_err / (mean_abs_tgt + eps)).mean()


def high_freq_error_loss(u_pred, u_target, k_frac: float = 0.25):
    if k_frac <= 0.0:
        return u_pred.new_tensor(0.0)
    err = u_pred - u_target
    err_ft = torch.fft.rfft(err, dim=1)
    K = err_ft.size(1)
    k0 = int(K * k_frac)
    if k0 >= K:
        return u_pred.new_tensor(0.0)
    spec = err_ft[:, k0:, :]
    return (spec.abs() ** 2).mean()


def high_freq_field_energy(u, k_frac: float = 0.25):
    if k_frac <= 0.0:
        return u.new_tensor(0.0)
    u_ft = torch.fft.rfft(u, dim=1)
    K = u_ft.size(1)
    k0 = int(K * k_frac)
    if k0 >= K:
        return u.new_tensor(0.0)
    spec = u_ft[:, k0:, :]
    return (spec.abs() ** 2).mean()


def periodic_laplacian_sq(z):
    z_m = torch.roll(z, 1, dims=1)
    z_p = torch.roll(z, -1, dims=1)
    lap = z_p + z_m - 2.0 * z
    return (lap**2).mean()


def transmissive_laplacian_sq(z):
    z_l = torch.cat([z[:, :1], z[:, :-1]], dim=1)
    z_r = torch.cat([z[:, 1:], z[:, -1:]], dim=1)
    lap = z_r + z_l - 2.0 * z
    return (lap**2).mean()


def total_loss_batch(pred_u, target_u, spec_mu, spec_kfrac, pred_spec_mu, lap_mu, model_bc: str):
    loss_rel = rel_l1_loss_per_channel(pred_u, target_u)
    if spec_mu > 0.0:
        loss_spec = high_freq_error_loss(pred_u, target_u, spec_kfrac)
    else:
        loss_spec = pred_u.new_tensor(0.0)
    if pred_spec_mu > 0.0:
        loss_pf = high_freq_field_energy(pred_u, spec_kfrac)
    else:
        loss_pf = pred_u.new_tensor(0.0)
    if lap_mu > 0.0:
        err = pred_u - target_u
        loss_lap = periodic_laplacian_sq(err) if model_bc == "periodic" else transmissive_laplacian_sq(err)
    else:
        loss_lap = pred_u.new_tensor(0.0)
    loss = loss_rel + spec_mu * loss_spec + pred_spec_mu * loss_pf + lap_mu * loss_lap
    return loss, loss_rel, loss_spec, loss_pf, loss_lap


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    spec_mu,
    spec_kfrac,
    pred_spec_mu,
    lap_mu,
    pin_memory: bool,
    model_bc: str,
):
    model.eval()
    tot = rel_sum = spec_sum = pfs_sum = lap_sum = 0.0
    n = 0
    for u0, u1 in loader:
        u0 = u0.to(device, non_blocking=pin_memory)
        u1 = u1.to(device, non_blocking=pin_memory)
        pred = model(u0)
        loss, lr, ls, lpf, ll = total_loss_batch(
            pred, u1, spec_mu, spec_kfrac, pred_spec_mu, lap_mu, model_bc
        )
        tot += loss.item()
        rel_sum += lr.item()
        spec_sum += ls.item()
        pfs_sum += lpf.item()
        lap_sum += ll.item()
        n += 1
    m = max(n, 1)
    spec_raw = spec_sum / m
    pfs_raw = pfs_sum / m
    lap_raw = lap_sum / m
    return {
        "loss": tot / m,
        "rel_l1": rel_sum / m,
        "spec": spec_raw,
        "spec_w": spec_mu * spec_raw,
        "pred_spec": pfs_raw,
        "pred_spec_w": pred_spec_mu * pfs_raw,
        "lap": lap_raw,
        "lap_w": lap_mu * lap_raw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--modes", type=int, default=32, help="FNO Fourier modes")
    ap.add_argument("--mr_kernel", type=int, default=5, help="unused; CLI parity with train.py")
    ap.add_argument(
        "--bc",
        type=str,
        default="auto",
        choices=("auto", "periodic", "outflow"),
    )
    ap.add_argument("--spectral_pad", type=int, default=4, help="unused; CLI parity with train.py")
    ap.add_argument("--fno_padding", type=int, default=2)
    ap.add_argument("--spec_mu", type=float, default=0.1)
    ap.add_argument("--spec_kfrac", type=float, default=0.25)
    ap.add_argument("--pred_spec_mu", type=float, default=0.0)
    ap.add_argument("--lap_mu", type=float, default=0.0)
    ap.add_argument("--save", type=str, default="checkpoints/swe_fno_flux_fixedstep.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument(
        "--compile_mode",
        type=str,
        default="auto",
        choices=("auto", "default", "reduce-overhead", "max-autotune"),
    )
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_pt = os.path.join(args.data_dir, "train_pv.pt")
    val_pt = os.path.join(args.data_dir, "val_pv.pt")

    pin_mem = device.type == "cuda"
    train_loader, val_loader, dx, N, dt, train_meta = load_conserved_pairs(
        train_pt,
        val_pt,
        args.batch,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    solver_bc = solver_bc_from_meta(train_meta)
    bc_arg = None if args.bc == "auto" else args.bc
    model_bc = model_bc_from_meta_or_arg(train_meta, bc_arg)
    print(
        f"[data] N={N}, dx={dx}, dt={dt}, dataset_boundary={train_meta.get('boundary', '?')}, "
        f"model_bc={model_bc}, solver_bc(ref)={solver_bc}"
    )

    n_cons = 2
    backbone = FNOFluxBackbone1d(
        modes=args.modes,
        width=args.width,
        n_layers=args.n_layers,
        n_cons=n_cons,
        bc=model_bc,
        padding=args.fno_padding,
    )
    model = HybridFixedStepMap1d(
        width=args.width,
        n_layers=args.n_layers,
        modes=args.modes,
        mr_kernel=args.mr_kernel,
        n_cons=n_cons,
        bc=model_bc,
        dx=dx,
        spectral_pad=args.spectral_pad,
        backbone=backbone,
    ).to(device)
    n_params = count_params(model)
    model = maybe_torch_compile(
        model,
        device,
        no_compile=args.no_compile,
        compile_mode=args.compile_mode,
        fullgraph=False,
    )
    print(f"[model] HybridFixedStepMap1d + FNOFluxBackbone1d (SWE n_cons=2), modes={args.modes}, params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

    best = 1e30
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_tot = tr_rel = tr_sp_w = tr_pf_w = tr_lap_w = 0.0
        nb = 0
        for u0, u1 in train_loader:
            u0 = u0.to(device, non_blocking=pin_mem)
            u1 = u1.to(device, non_blocking=pin_mem)
            pred = model(u0)
            loss, loss_rel, loss_spec, loss_pf, loss_lap = total_loss_batch(
                pred,
                u1,
                args.spec_mu,
                args.spec_kfrac,
                args.pred_spec_mu,
                args.lap_mu,
                model_bc,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_tot += loss.item()
            tr_rel += loss_rel.item()
            tr_sp_w += (args.spec_mu * loss_spec).item()
            tr_pf_w += (args.pred_spec_mu * loss_pf).item()
            tr_lap_w += (args.lap_mu * loss_lap).item()
            nb += 1

        vm = evaluate(
            model,
            val_loader,
            device,
            args.spec_mu,
            args.spec_kfrac,
            args.pred_spec_mu,
            args.lap_mu,
            pin_mem,
            model_bc,
        )
        print(
            f"ep {ep}: tr={tr_tot/nb:.3e} v={vm['loss']:.3e} vrel={vm['rel_l1']:.3e} "
            f"vsp={vm['spec_w']:.3e} vlap={vm['lap_w']:.3e}"
        )
        sched.step()

        if vm["loss"] < best:
            best = vm["loss"]
            save_args = dict(vars(args))
            save_args["backbone"] = "fno"
            save_args["n_cons"] = n_cons
            torch.save(
                {
                    "model": state_dict_for_ckpt(model),
                    "dt": dt,
                    "dx": dx,
                    "nx": N,
                    "bc": model_bc,
                    "solver_bc": solver_bc,
                    "kind": "swe_fno_flux_fixedstep",
                    "n_cons": n_cons,
                    "state_rep": "conserved",
                    "integrator": "u_minus_q_projected",
                    "args": save_args,
                },
                args.save,
            )
            print(f"  [best] -> {args.save}")


if __name__ == "__main__":
    main()
