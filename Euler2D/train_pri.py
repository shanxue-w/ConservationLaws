"""
Train Euler2D primitive-variable hybrid dt-step models.

The dataset stays in conservative form on disk. This script converts states to
primitive variables [rho, u, v, p] before building one-step pairs, then applies
a smooth positivity modifier to rho and p at the model output.
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

from conslaw.models import count_params, maybe_torch_compile, state_dict_for_ckpt

from common_pri import (
    DEFAULT_TRAIN_NAME,
    DEFAULT_VAL_NAME,
    P_FLOOR_DEFAULT,
    POSITIVE_BETA_DEFAULT,
    RHO_FLOOR_DEFAULT,
    build_hybrid_model_from_args,
    configure_runtime,
    evaluate_step_model,
    make_pair_loaders,
    resolve_model_bc,
    resolve_split_path,
    torch_load_cpu,
    total_loss_batch_2d,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--train_name", type=str, default=DEFAULT_TRAIN_NAME)
    ap.add_argument("--val_name", type=str, default=DEFAULT_VAL_NAME)
    ap.add_argument("--train_max_pairs", type=int, default=0)
    ap.add_argument("--val_max_pairs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--modes", type=int, default=24)
    ap.add_argument("--modes2", type=int, default=0)
    ap.add_argument("--mr_kernel", type=int, default=7)
    ap.add_argument("--spectral_pad", type=int, default=4)
    ap.add_argument("--backbone2d", type=str, default="outflow", choices=("outflow", "standard"))
    ap.add_argument("--outflow_ctx_width", type=int, default=0)
    ap.add_argument("--bc", type=str, default="outflow", choices=("auto", "periodic", "outflow"))
    ap.add_argument("--no_zero_mean_rhs", action="store_true")
    ap.add_argument("--rho_floor", type=float, default=RHO_FLOOR_DEFAULT)
    ap.add_argument("--p_floor", type=float, default=P_FLOOR_DEFAULT)
    ap.add_argument("--positive_beta", type=float, default=POSITIVE_BETA_DEFAULT)
    ap.add_argument("--spec_mu", type=float, default=1e-3)
    ap.add_argument("--spec_kfrac", type=float, default=0.25)
    ap.add_argument("--lap_mu", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save", type=str, default="checkpoints/euler2d_hybrid_pri_outflow_1e-2_dt.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow_tf32", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--compile_mode", type=str, default="auto", choices=("auto", "default", "reduce-overhead", "max-autotune"))
    args = ap.parse_args()

    device = torch.device(args.device)
    configure_runtime(device, allow_tf32=args.allow_tf32)
    pin_mem = bool(args.pin_memory and device.type == "cuda")

    train_path = resolve_split_path(args.data_dir, args.train_name)
    val_path = resolve_split_path(args.data_dir, args.val_name)
    train_meta = dict(torch_load_cpu(train_path)["meta"])
    model_bc = resolve_model_bc(train_meta, None if args.bc == "auto" else args.bc)

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
    zero_mean_rhs = False if model_bc == "outflow" else (not args.no_zero_mean_rhs)
    project_outflow_rhs = False

    print(
        f"[data-pri] dt={dt}, dx={dx}, dy={dy}, nx={nx}, ny={ny}, gamma={info['gamma']}, "
        f"train_pairs={info['n_train_pairs']}, val_pairs={info['n_val_pairs']}, "
        f"dataset_bc={train_meta.get('boundary', '?')}, model_bc={model_bc}"
    )
    print(
        f"[positive] rho_floor={args.rho_floor:g}, p_floor={args.p_floor:g}, "
        f"positive_beta={args.positive_beta:g}, output_modifier=softplus"
    )

    build_args = dict(vars(args))
    build_args.update(
        {
            "modes2": modes2,
            "bc": model_bc,
            "zero_mean_rhs": zero_mean_rhs,
            "project_outflow_rhs": project_outflow_rhs,
        }
    )
    model = build_hybrid_model_from_args(build_args, dx=dx, dy=dy).to(device)
    n_params = count_params(model)
    model = maybe_torch_compile(model, device, no_compile=args.no_compile, compile_mode=args.compile_mode, fullgraph=False)
    print(f"[model] euler2d_hybrid_pri_dt, params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

    save_args = dict(vars(args))
    save_args.update(
        {
            "primitive": True,
            "positive_output": "softplus_rho_p",
            "modes2": modes2,
            "in_channels": 4,
            "out_channels": 4,
            "bc": model_bc,
            "zero_mean_rhs": zero_mean_rhs,
            "project_outflow_rhs": project_outflow_rhs,
            "dtype": "float32",
        }
    )
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_tot = tr_l1 = tr_sp = tr_lap = 0.0
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
            tr_tot += float(loss.detach())
            tr_l1 += float(loss_l1.detach())
            tr_sp += float(loss_spec.detach())
            tr_lap += float(loss_lap.detach())
            nb += 1
            print(f"ep: {ep} batch: {nb}/{len(train_loader)}", end="\r")

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
        scale = 1.0 / max(nb, 1)
        print(
            f"ep {ep}: tr={tr_tot * scale:.3e} tl1={tr_l1 * scale:.3e} "
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
                    "kind": "euler2d_hybrid_pri_dt",
                    "args": save_args,
                },
                args.save,
            )
            print(f"  [best] -> {args.save}")


if __name__ == "__main__":
    main()
