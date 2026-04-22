"""
Train Burgers2D FNO/CNN one-step residual baselines.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from conslaw.models import CNNDtStep2d, FNODtStep2d, count_params, maybe_torch_compile, state_dict_for_ckpt

from train import evaluate, make_u_pair_loader, total_loss_batch

torch.set_default_dtype(torch.float64)


def resolve_zero_mean_rhs(args: argparse.Namespace) -> bool:
    if args.zero_mean_rhs and args.no_zero_mean_rhs:
        raise ValueError("Use only one of --zero_mean_rhs or --no_zero_mean_rhs.")
    if args.zero_mean_rhs:
        return True
    if args.no_zero_mean_rhs:
        return False
    return args.bc == "periodic"


def resolve_save_path(args: argparse.Namespace) -> str:
    default_fno = "checkpoints/burgers2d_fno_dt.pt"
    if args.save != default_fno:
        return args.save
    if args.backbone == "cnn":
        return "checkpoints/burgers2d_cnn_dt.pt"
    return default_fno


def main() -> None:
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
    ap.add_argument("--backbone", type=str, default="fno", choices=("fno", "cnn"))
    ap.add_argument("--modes", type=int, default=32)
    ap.add_argument("--modes2", type=int, default=0)
    ap.add_argument("--padding", type=int, default=4, help="FNO padding used for outflow.")
    ap.add_argument("--kernel_size", type=int, default=5, help="CNN kernel size.")
    ap.add_argument("--bc", type=str, default="periodic", choices=("periodic", "outflow"))
    ap.add_argument("--zero_mean_rhs", action="store_true")
    ap.add_argument("--no_zero_mean_rhs", action="store_true")
    ap.add_argument("--spec_mu", type=float, default=0.0)
    ap.add_argument("--spec_kfrac", type=float, default=0.25)
    ap.add_argument("--lap_mu", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save", type=str, default="checkpoints/burgers2d_fno_dt.pt")
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

    pin_mem = bool(args.pin_memory and device.type == "cuda")
    zero_mean_rhs = resolve_zero_mean_rhs(args)
    save_path = resolve_save_path(args)

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
    modes2 = int(args.modes if args.modes2 <= 0 else args.modes2)
    print(
        f"[data] dt={dt}, dx={dx}, dy={dy}, nx={nx}, ny={ny}, "
        f"train={split_info['n_train']}, val={split_info['n_val']}, "
        f"dataset_bc={meta.get('boundary', 'periodic')}, model_bc={args.bc}, "
        f"backbone={args.backbone}, zero_mean_rhs={zero_mean_rhs}"
    )

    if args.backbone == "fno":
        model = FNODtStep2d(
            width=args.width,
            n_layers=args.n_layers,
            modes1=args.modes,
            modes2=modes2,
            in_channels=1,
            out_channels=1,
            bc=args.bc,
            padding=args.padding,
            zero_mean_rhs=zero_mean_rhs,
        ).to(device)
        kind = "burgers2d_fno_dt"
    else:
        model = CNNDtStep2d(
            width=args.width,
            n_layers=args.n_layers,
            kernel_size=args.kernel_size,
            in_channels=1,
            out_channels=1,
            bc=args.bc,
            zero_mean_rhs=zero_mean_rhs,
        ).to(device)
        kind = "burgers2d_cnn_dt"

    n_params = count_params(model)
    model = maybe_torch_compile(
        model,
        device,
        no_compile=args.no_compile,
        compile_mode=args.compile_mode,
        fullgraph=False,
    )
    print(f"[model] {kind}, params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    best = float("inf")
    save_args = dict(vars(args))
    save_args.update(
        {
            "modes2": modes2,
            "in_channels": 1,
            "out_channels": 1,
            "bc": args.bc,
            "backbone": args.backbone,
            "zero_mean_rhs": zero_mean_rhs,
        }
    )

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_tot = tr_u = tr_s = tr_l = 0.0
        nb = 0
        for u0, u1 in train_loader:
            u0 = u0.to(device, non_blocking=pin_mem)
            u1 = u1.to(device, non_blocking=pin_mem)
            dt_b = torch.full((u0.size(0),), dt, device=device, dtype=u0.dtype)
            pred = model(u0, dt_b)
            loss, loss_u, loss_spec, loss_lap = total_loss_batch(
                pred,
                u1,
                args.spec_mu,
                args.spec_kfrac,
                args.lap_mu,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_tot += loss.item()
            tr_u += loss_u.item()
            tr_s += loss_spec.item()
            tr_l += loss_lap.item()
            nb += 1

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
            f"ep {ep}: tr={tr_tot/max(nb,1):.3e} tu={tr_u/max(nb,1):.3e} "
            f"tsp={args.spec_mu * tr_s/max(nb,1):.3e} tlap={args.lap_mu * tr_l/max(nb,1):.3e} "
            f"v={vm['loss']:.3e} vu={vm['loss_u']:.3e} "
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
                    "bc": args.bc,
                    "kind": kind,
                    "args": save_args,
                },
                save_path,
            )
            print(f"  [best] -> {save_path}")


if __name__ == "__main__":
    main()
