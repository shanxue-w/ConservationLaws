"""
Train Euler2D FNO/CNN one-step residual baselines on trajectory datasets.
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from conslaw.models import CNNDtStep2d, FNODtStep2d, count_params, maybe_torch_compile, state_dict_for_ckpt

from common_periodic import (
    DEFAULT_TRAIN_NAME,
    DEFAULT_VAL_NAME,
    configure_runtime,
    default_dt_ckpt_path,
    default_num_workers,
    evaluate_step_model,
    make_pair_loaders,
    resolve_model_bc,
    resolve_zero_mean_rhs,
    resolve_split_path,
    total_loss_batch_2d,
    torch_load_cpu,
)


def resolve_save_path(args: argparse.Namespace) -> str:
    default_fno = default_dt_ckpt_path("fno", "periodic")
    if args.save != default_fno:
        return args.save
    bc = getattr(args, "resolved_bc", "periodic")
    if args.backbone == "cnn":
        return default_dt_ckpt_path("cnn", bc)
    return default_dt_ckpt_path("fno", bc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--train_name", type=str, default=DEFAULT_TRAIN_NAME)
    ap.add_argument("--val_name", type=str, default=DEFAULT_VAL_NAME)
    ap.add_argument("--train_max_pairs", type=int, default=0)
    ap.add_argument("--val_max_pairs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=default_num_workers())
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--backbone", type=str, default="fno", choices=("fno", "cnn"))
    ap.add_argument("--modes", type=int, default=24)
    ap.add_argument("--modes2", type=int, default=0)
    ap.add_argument("--padding", type=int, default=4, help="FNO spectral padding.")
    ap.add_argument("--kernel_size", type=int, default=5, help="CNN kernel size.")
    ap.add_argument("--bc", type=str, default="periodic", choices=("auto", "periodic", "outflow"))
    ap.add_argument("--zero_mean_rhs", action="store_true")
    ap.add_argument("--no_zero_mean_rhs", action="store_true")
    ap.add_argument("--spec_mu", type=float, default=1e-3)
    ap.add_argument("--spec_kfrac", type=float, default=0.25)
    ap.add_argument("--lap_mu", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save", type=str, default=default_dt_ckpt_path("fno", "periodic"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow_tf32", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument(
        "--compile_mode",
        type=str,
        default="auto",
        choices=("auto", "default", "reduce-overhead", "max-autotune"),
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    configure_runtime(device, allow_tf32=args.allow_tf32)
    pin_mem = bool(args.pin_memory and device.type == "cuda")
    train_path = resolve_split_path(args.data_dir, args.train_name)
    val_path = resolve_split_path(args.data_dir, args.val_name)
    train_meta = dict(torch_load_cpu(train_path)["meta"])
    model_bc = resolve_model_bc(train_meta, None if args.bc == "auto" else args.bc)
    zero_mean_rhs = resolve_zero_mean_rhs(
        model_bc,
        enable=args.zero_mean_rhs,
        disable=args.no_zero_mean_rhs,
    )
    args.resolved_bc = model_bc
    save_path = resolve_save_path(args)

    train_loader, val_loader, info = make_pair_loaders(
        train_path,
        val_path,
        args.batch,
        dtype=torch.float32,
        train_max_pairs=args.train_max_pairs if args.train_max_pairs > 0 else None,
        val_max_pairs=args.val_max_pairs if args.val_max_pairs > 0 else None,
        shuffle_train=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=args.persistent_workers,
    )
    dt = float(info["dt"])
    dx = float(info["dx"])
    dy = float(info["dy"])
    nx = int(info["nx"])
    ny = int(info["ny"])
    modes2 = int(args.modes if args.modes2 <= 0 else args.modes2)
    print(
        f"[data] dt={dt}, dx={dx}, dy={dy}, nx={nx}, ny={ny}, "
        f"train_pairs={info['n_train_pairs']}, val_pairs={info['n_val_pairs']}, "
        f"dataset_bc={train_meta.get('boundary', '?')}, model_bc={model_bc}, "
        f"backbone={args.backbone}, zero_mean_rhs={zero_mean_rhs}"
    )

    if args.backbone == "fno":
        model = FNODtStep2d(
            width=args.width,
            n_layers=args.n_layers,
            modes1=args.modes,
            modes2=modes2,
            in_channels=4,
            out_channels=4,
            bc=model_bc,
            padding=args.padding,
            zero_mean_rhs=zero_mean_rhs,
        ).to(device)
        kind = "euler2d_fno_dt"
    else:
        model = CNNDtStep2d(
            width=args.width,
            n_layers=args.n_layers,
            kernel_size=args.kernel_size,
            in_channels=4,
            out_channels=4,
            bc=model_bc,
            zero_mean_rhs=zero_mean_rhs,
        ).to(device)
        kind = "euler2d_cnn_dt"

    n_params = count_params(model)
    model = maybe_torch_compile(
        model,
        device,
        no_compile=args.no_compile,
        compile_mode=args.compile_mode,
        fullgraph=False,
    )
    print(f"[model] {kind} (bc={model_bc}), params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    best = float("inf")
    save_args = dict(vars(args))
    save_args["save"] = save_path
    save_args.update(
        {
            "modes2": modes2,
            "in_channels": 4,
            "out_channels": 4,
            "bc": model_bc,
            "zero_mean_rhs": zero_mean_rhs,
            "dtype": "float32",
        }
    )

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_tot = torch.zeros((), device=device)
        tr_l1 = torch.zeros((), device=device)
        tr_sp = torch.zeros((), device=device)
        tr_lap = torch.zeros((), device=device)
        nb = 0
        for u0, u1 in train_loader:
            u0 = u0.to(device, non_blocking=pin_mem)
            u1 = u1.to(device, non_blocking=pin_mem)
            dt_b = torch.full((u0.size(0),), dt, device=device, dtype=u0.dtype)
            pred = model(u0, dt_b)
            loss, loss_l1, loss_spec, loss_lap = total_loss_batch_2d(
                pred,
                u1,
                spec_mu=args.spec_mu,
                spec_kfrac=args.spec_kfrac,
                lap_mu=args.lap_mu,
                model_bc=model_bc,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_tot += loss.detach()
            tr_l1 += loss_l1.detach()
            tr_sp += loss_spec.detach()
            tr_lap += loss_lap.detach()
            nb += 1

        vm = evaluate_step_model(
            model,
            val_loader,
            dt,
            device,
            spec_mu=args.spec_mu,
            spec_kfrac=args.spec_kfrac,
            lap_mu=args.lap_mu,
            pin_memory=pin_mem,
            model_bc=model_bc,
        )
        tr_scale = 1.0 / max(nb, 1)
        print(
            f"ep {ep}: tr={(tr_tot * tr_scale).item():.3e} tl1={(tr_l1 * tr_scale).item():.3e} "
            f"v={vm['loss']:.3e} vl1={vm['loss_l1']:.3e} "
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
                    "bc": model_bc,
                    "kind": kind,
                    "args": save_args,
                },
                save_path,
            )
            print(f"  [best] -> {save_path}")


if __name__ == "__main__":
    main()
