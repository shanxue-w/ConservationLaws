"""
Train 2D Burgers one-step hybrid map u^{n+1} = u + dt * NN(u) (conslaw HybridDtStep2d).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from conslaw.models import (
    HybridDtStep2d,
    count_params,
    maybe_torch_compile,
    state_dict_for_ckpt,
)

torch.set_default_dtype(torch.float64)


def _load_pt_mmap(pt_path: str):
    return torch.load(pt_path, mmap=True)


def _concat_first_n(tensors, n_total):
    parts = []
    remaining = int(n_total)
    for tensor in tensors:
        if remaining <= 0:
            break
        take = min(int(tensor.shape[0]), remaining)
        if take > 0:
            parts.append(tensor[:take])
            remaining -= take
    if not parts:
        raise ValueError("No samples were selected from the dataset pool.")
    return torch.cat(parts, dim=0)


def _compute_split_counts(n_total, ratios):
    if n_total <= 0:
        raise ValueError("n_total must be positive.")
    if len(ratios) < 2:
        raise ValueError("Need at least two split ratios.")
    ratio_sum = sum(ratios)
    counts = [int(n_total * r / ratio_sum) for r in ratios]
    counts[-1] = n_total - sum(counts[:-1])
    if n_total >= len(ratios):
        for idx in range(1, len(ratios)):
            if counts[idx] == 0:
                donor = 0
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[idx] += 1
        counts[0] = n_total - sum(counts[1:])
    return counts


def make_u_pair_loader(
    data_dir,
    batch_size,
    max_samples=None,
    train_samples=None,
    val_samples=None,
    shuffle_train=True,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
):
    train_pt_path = os.path.join(data_dir, "train_pv_noavg.pt")
    val_pt_path = os.path.join(data_dir, "val_pv_noavg.pt")

    train_data = _load_pt_mmap(train_pt_path)
    val_data = _load_pt_mmap(val_pt_path)

    meta = train_data["meta"]
    dt = float(meta["dt"])
    dx = float(meta["dx"])
    dy = float(meta["dy"])
    nx = int(meta["nx"])
    ny = int(meta["ny"])

    n_available = sum(x["input"].shape[0] for x in (train_data, val_data))
    use_explicit_counts = (
        train_samples is not None and int(train_samples) > 0
    ) or (val_samples is not None and int(val_samples) > 0)

    if use_explicit_counts:
        n_train = max(0, 0 if train_samples is None else int(train_samples))
        n_val = max(0, 0 if val_samples is None else int(val_samples))
        n_requested = n_train + n_val
        if n_requested <= 0:
            raise ValueError("Requested train_samples + val_samples must be positive.")
        if n_requested > n_available:
            raise ValueError(f"Requested {n_requested} samples but only {n_available} are available.")
        n_total = n_requested
    else:
        n_total = (
            n_available
            if max_samples is None or max_samples <= 0
            else min(int(max_samples), n_available)
        )

    input_all = _concat_first_n([train_data["input"], val_data["input"]], n_total)
    output_all = _concat_first_n([train_data["output"], val_data["output"]], input_all.shape[0])

    n_total = int(input_all.shape[0])
    if use_explicit_counts:
        n_train = max(0, 0 if train_samples is None else int(train_samples))
        n_val = max(0, 0 if val_samples is None else int(val_samples))
        if n_train + n_val != n_total:
            raise ValueError("Explicit train/val sample counts do not match selected total.")
    else:
        n_train, n_val = _compute_split_counts(n_total, ratios=(5, 1))

    u0_train = input_all[:n_train].unsqueeze(-1)
    u1_train = output_all[:n_train].unsqueeze(-1)
    u0_val = input_all[n_train : n_train + n_val].unsqueeze(-1)
    u1_val = output_all[n_train : n_train + n_val].unsqueeze(-1)

    train_loader = DataLoader(
        TensorDataset(u0_train, u1_train),
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    val_loader = DataLoader(
        TensorDataset(u0_val, u1_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    split_info = {"n_total": n_total, "n_train": n_train, "n_val": n_val}
    return train_loader, val_loader, dt, dx, dy, nx, ny, split_info, meta


def high_freq_error_loss_2d(u_pred, u_target, k_frac: float = 0.25):
    if k_frac <= 0.0:
        return u_pred.new_tensor(0.0)
    err = (u_pred - u_target).squeeze(-1)
    err_ft = torch.fft.rfftn(err, dim=(1, 2))
    H, Wh = err_ft.shape[1], err_ft.shape[2]
    ky0 = int(H * k_frac)
    kx0 = int(Wh * k_frac)
    mask_y = torch.arange(H, device=err.device)[:, None] >= ky0
    mask_x = torch.arange(Wh, device=err.device)[None, :] >= kx0
    mask = mask_y | mask_x
    spec = err_ft * mask[None, :, :]
    return (spec.abs() ** 2).mean()


def periodic_laplacian_sq_2d(z):
    zm_x = torch.roll(z, 1, dims=2)
    zp_x = torch.roll(z, -1, dims=2)
    zm_y = torch.roll(z, 1, dims=1)
    zp_y = torch.roll(z, -1, dims=1)
    lap = zp_x + zm_x + zp_y + zm_y - 4.0 * z
    return (lap**2).mean()


def total_loss_batch(pred_u, target_u, spec_mu, spec_kfrac, lap_mu):
    loss_u = F.l1_loss(pred_u, target_u)
    if spec_mu > 0.0:
        loss_spec = high_freq_error_loss_2d(pred_u, target_u, spec_kfrac)
    else:
        loss_spec = pred_u.new_tensor(0.0)
    if lap_mu > 0.0:
        err = (pred_u - target_u).squeeze(-1)
        loss_lap = periodic_laplacian_sq_2d(err)
    else:
        loss_lap = pred_u.new_tensor(0.0)
    loss = loss_u + spec_mu * loss_spec + lap_mu * loss_lap
    return loss, loss_u, loss_spec, loss_lap


@torch.no_grad()
def evaluate(model, loader, dt, device, spec_mu, spec_kfrac, lap_mu, pin_memory: bool):
    model.eval()
    tot = lu = ls = ll = 0.0
    n = 0
    for u0, u1 in loader:
        u0 = u0.to(device, non_blocking=pin_memory)
        u1 = u1.to(device, non_blocking=pin_memory)
        B = u0.size(0)
        dt_b = torch.full((B,), dt, device=device, dtype=u0.dtype)
        pred = model(u0, dt_b)
        loss, a, b, c = total_loss_batch(pred, u1, spec_mu, spec_kfrac, lap_mu)
        tot += loss.item()
        lu += a.item()
        ls += b.item()
        ll += c.item()
        n += 1
    m = max(n, 1)
    return {
        "loss": tot / m,
        "loss_u": lu / m,
        "loss_spec": ls / m,
        "loss_lap": ll / m,
        "spec_w": spec_mu * ls / m,
        "lap_w": lap_mu * ll / m,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--train_samples", type=int, default=0)
    ap.add_argument("--val_samples", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--modes2", type=int, default=None)
    ap.add_argument("--mr_kernel", type=int, default=5)
    ap.add_argument(
        "--bc",
        type=str,
        default="periodic",
        choices=("periodic", "outflow"),
        help="hybrid backbone / rhs projector boundary mode",
    )
    ap.add_argument(
        "--no_zero_mean_rhs",
        action="store_true",
        help="disable spatial mean removal on rhs (see PeriodicRhs2d / OutflowAffineLearnedRhs2d)",
    )
    ap.add_argument("--spec_mu", type=float, default=0.0)
    ap.add_argument("--spec_kfrac", type=float, default=0.25)
    ap.add_argument("--lap_mu", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save", type=str, default="checkpoints/burgers2d_hybrid_dt.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument(
        "--compile_mode",
        type=str,
        default="auto",
        choices=("auto", "default", "reduce-overhead", "max-autotune"),
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, dt, dx, dy, nx, ny, split_info, meta = make_u_pair_loader(
        args.data_dir,
        args.batch,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        train_samples=args.train_samples if args.train_samples > 0 else None,
        val_samples=args.val_samples if args.val_samples > 0 else None,
        shuffle_train=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    modes2 = int(args.modes if args.modes2 is None else args.modes2)
    print(
        f"[data] dt={dt}, dx={dx}, dy={dy}, nx={nx}, ny={ny}, "
        f"train={split_info['n_train']}, val={split_info['n_val']}, "
        f"meta.boundary={meta.get('boundary', 'periodic')}, model_bc={args.bc}"
    )
    if nx % 2 != 0 or ny % 2 != 0:
        print("[warn] MR branch uses 2x pooling; even nx, ny are required.")

    model = HybridDtStep2d(
        width=args.width,
        n_layers=args.n_layers,
        modes1=args.modes,
        modes2=modes2,
        mr_kernel=args.mr_kernel,
        in_channels=1,
        out_channels=1,
        bc=args.bc,
        dx=dx,
        dy=dy,
        spectral_pad=4,
        zero_mean_rhs=not args.no_zero_mean_rhs,
    ).to(device)
    n_params = count_params(model)
    model = maybe_torch_compile(
        model,
        device,
        no_compile=args.no_compile,
        compile_mode=args.compile_mode,
        fullgraph=False,
    )
    print(f"[model] HybridDtStep2d, params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

    pin_mem = args.pin_memory and device.type == "cuda"
    best = 1e30
    save_args = vars(args).copy()
    save_args["modes2"] = modes2
    save_args["zero_mean_rhs"] = not args.no_zero_mean_rhs

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_tot = tr_u = tr_s = tr_l = 0.0
        nb = 0
        for u0, u1 in train_loader:
            u0 = u0.to(device, non_blocking=pin_mem)
            u1 = u1.to(device, non_blocking=pin_mem)
            B = u0.size(0)
            dt_b = torch.full((B,), dt, device=device, dtype=u0.dtype)
            pred = model(u0, dt_b)
            loss, a, b, c = total_loss_batch(pred, u1, args.spec_mu, args.spec_kfrac, args.lap_mu)
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_tot += loss.item()
            tr_u += a.item()
            tr_s += b.item()
            tr_l += c.item()
            nb += 1
            print(
                f"\rep={ep} batch={nb}/{len(train_loader)} loss={loss.item():.3e}",
                end="",
                flush=True,
            )
        print()

        vm = evaluate(
            model,
            val_loader,
            dt,
            device,
            args.spec_mu,
            args.spec_kfrac,
            args.lap_mu,
            pin_mem,
        )
        print(
            f"ep {ep}: tr={tr_tot/nb:.3e} v={vm['loss']:.3e} vu={vm['loss_u']:.3e} "
            f"vsp={vm['spec_w']:.3e} vlap={vm['lap_w']:.3e}"
        )
        sched.step()

        if vm["loss"] < best:
            best = vm["loss"]
            torch.save(
                {
                    "model": state_dict_for_ckpt(model),
                    "dt": dt,
                    "dx": dx,
                    "dy": dy,
                    "nx": nx,
                    "ny": ny,
                    "bc": getattr(model, "bc", args.bc),
                    "kind": "burgers2d_hybrid_dt",
                    "args": save_args,
                },
                args.save,
            )
            print(f"  [best] -> {args.save}")


if __name__ == "__main__":
    main()
