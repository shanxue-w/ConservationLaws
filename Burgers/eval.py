"""
Rollout Burgers fixed-step map from ``train.py`` checkpoint; WENO reference via conslaw.solver.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from conslaw.checkpoints import compile_if_requested, load_hybrid_step_map_1d
from conslaw.eval_reports import dataset_error_row, save_rollout_reports, save_test_metrics_csv
from conslaw.models import count_params
from conslaw.solver import FD_WENOZ

torch.set_default_dtype(torch.float64)

DEFAULT_FNO_CKPT = "checkpoints/burgers_fno_flowmap_dt_24.pt"
LINE_GRID_ALPHA = 0.3
XT_CMAP = "turbo"


def apply_line_grid(ax) -> None:
    ax.grid(True, alpha=LINE_GRID_ALPHA, linewidth=0.8)


def build_burgers_solver(bc: str = "periodic"):
    def flux(u):
        return 0.5 * u * u

    def dflux(u):
        return u

    return FD_WENOZ(flux=flux, dflux=dflux, flux_split="local_lf", bc=bc)


def solve_burgers_weno_ref(u0_fine, times, dx_fine, bc="periodic", cfl=0.4):
    solver = build_burgers_solver(bc=bc)
    u = np.asarray(u0_fine, dtype=np.float64).copy()
    snaps = [u.copy()]
    t = 0.0
    for out_idx in range(1, len(times)):
        t_target = float(times[out_idx])
        dt_interval = t_target - t
        u = solver.solve(u, dx=dx_fine, T=dt_interval, cfl=cfl, return_all=False)
        t = t_target
        snaps.append(u.copy())
    return np.stack(snaps, axis=0)


def block_average(u_hi, r):
    return u_hi.reshape(-1, r).mean(axis=1)


def random_fourier_ic(x, kmax=5, amp_range=(0.5, 1.0), decay=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    u = np.zeros_like(x, dtype=np.float64)
    for k in range(1, kmax + 1):
        scale = 1.0 / (k**decay)
        ak = rng.normal(0.0, scale)
        bk = rng.normal(0.0, scale)
        u += ak * np.cos(2 * np.pi * k * x) + bk * np.sin(2 * np.pi * k * x)
    umax = np.max(np.abs(u)) + 1e-12
    target_amp = rng.uniform(amp_range[0], amp_range[1])
    u = u / umax * target_amp
    u += rng.uniform(-0.5, 0.5) * target_amp
    return u.astype(np.float64)


def rel_l1(a, b, eps=1e-12):
    return np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + eps)


def rel_linf(a, b, eps=1e-12):
    return np.max(np.abs(a - b)) / (np.max(np.abs(b)) + eps)


def display_model_name(name: str) -> str:
    return {
        "ref": "Ref",
        "hybrid": "LGNO",
        "fno": "FNO",
        "cnn": "CNN",
    }.get(str(name).lower(), str(name))


def add_ic_right_axis(ax, x: np.ndarray, values: np.ndarray):
    ax_ic = ax.twinx()
    (line,) = ax_ic.plot(x, values, label="IC", color="black", linestyle="--", lw=2)
    ax_ic.set_ylabel("IC", fontsize=14, color="black")
    ax_ic.tick_params(axis="y", colors="black")
    ax_ic.grid(False)
    return line


def save_metric_curves_pdf(
    outdir: str,
    times: np.ndarray,
    series: dict[str, np.ndarray],
    *,
    metric_label: str,
    filename: str,
) -> None:
    if times.size <= 1:
        return
    times_plot = times[1:]
    ylabel = r"Relative $L^\infty$ error" if "linf" in metric_label.lower() else r"Relative $L^1$ error"
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name, values in series.items():
        mean = values.mean(axis=0)[1:]
        std = values.std(axis=0)[1:]
        ax.plot(times_plot, mean, label=display_model_name(model_name))
        ax.fill_between(times_plot, mean - std, mean + std, alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("t", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(loc="best")
    apply_line_grid(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, filename), bbox_inches="tight")
    plt.close(fig)


def save_scalar_profile_pdf(
    x: np.ndarray,
    values: np.ndarray,
    *,
    ylabel: str,
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, values, lw=2)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    apply_line_grid(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_scalar_xt_pdf(
    x: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    path: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    extent = [float(x[0]), float(x[-1] + dx), float(times[0]), float(times[-1])]
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    im = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("t", fontsize=14)
    ax.set_box_aspect(1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_scalar_rollout_visuals(
    outdir: str,
    x: np.ndarray,
    times: np.ndarray,
    ref_traj: np.ndarray,
    pred_trajs: dict[str, np.ndarray],
    *,
    suffix: str,
) -> None:
    initial_ref = ref_traj[0]
    final_ref = ref_traj[-1]
    t_tag = f"T{times[-1]:g}".replace(".", "p")

    save_scalar_xt_pdf(
        x,
        times,
        ref_traj,
        title="",
        path=os.path.join(outdir, f"xt_ref_{t_tag}{suffix}.pdf"),
        cmap=XT_CMAP,
    )

    save_scalar_profile_pdf(
        x,
        initial_ref,
        ylabel="u",
        path=os.path.join(outdir, f"initial_condition_{t_tag}{suffix}.pdf"),
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, final_ref, label="Ref", lw=2)
    for name, pred_traj in pred_trajs.items():
        ax.plot(x, pred_traj[-1], label=display_model_name(name), lw=2)
    ic_line = add_ic_right_axis(ax, x, initial_ref)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("u", fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [ic_line], labels + ["IC"], loc="best")
    apply_line_grid(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"final_solution_{t_tag}{suffix}.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, pred_traj in pred_trajs.items():
        final_err = np.abs(pred_traj[-1] - final_ref)
        ax.plot(x, final_err, label=f"{display_model_name(name)} error", lw=2)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel(r"$|u - u_{\text{ref}}|$", fontsize=14)
    ax.legend(loc="best")
    apply_line_grid(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"final_error_{t_tag}{suffix}.pdf"), bbox_inches="tight")
    plt.close(fig)

    for name, pred_traj in pred_trajs.items():
        xt_err = np.abs(pred_traj - ref_traj)
        save_scalar_xt_pdf(
            x,
            times,
            pred_traj,
            title="",
            path=os.path.join(outdir, f"xt_{name}_{t_tag}{suffix}.pdf"),
            cmap=XT_CMAP,
        )
        save_scalar_xt_pdf(
            x,
            times,
            xt_err,
            title="",
            path=os.path.join(outdir, f"xt_error_{name}_{t_tag}{suffix}.pdf"),
            cmap=XT_CMAP,
            vmin=0.0,
            vmax=max(float(xt_err.max()), 1e-12),
        )


@torch.no_grad()
def rollout_hybrid_scalar(step_model, u0_low, n_steps, device, pin_memory: bool, dt_model: float | None = None):
    if hasattr(step_model, "eval") and callable(step_model.eval):
        step_model.eval()
    u = torch.from_numpy(u0_low).view(1, -1, 1)
    if pin_memory and device.type == "cuda":
        u = u.pin_memory()
    u = u.to(device=device, dtype=torch.float64, non_blocking=pin_memory and device.type == "cuda")
    traj = [u0_low.copy()]
    for _ in range(n_steps):
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        if dt_model is not None:
            dt_b = torch.full((1,), dt_model, device=u.device, dtype=u.dtype)
            u = step_model(u, dt_b).clone()
        else:
            u = step_model(u).clone()
        traj.append(u.squeeze(0).squeeze(-1).cpu().numpy())
    return np.stack(traj, axis=0)


@torch.no_grad()
def evaluate_test_dataset_scalar(
    step_model, test_pt_path, device, pin_memory: bool, batch_size: int, dt_model: float | None = None
):
    data = torch.load(test_pt_path, map_location="cpu")
    u0_all = data["input"].unsqueeze(-1).to(torch.float64)
    u1_all = data["output"].unsqueeze(-1).to(torch.float64)
    preds = []
    step_model.eval()
    for start in range(0, u0_all.size(0), batch_size):
        u0 = u0_all[start : start + batch_size]
        if pin_memory and device.type == "cuda":
            u0 = u0.pin_memory()
        u0 = u0.to(device=device, dtype=torch.float64, non_blocking=pin_memory and device.type == "cuda")
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        if dt_model is not None:
            dt_b = torch.full((u0.size(0),), dt_model, device=u0.device, dtype=u0.dtype)
            preds.append(step_model(u0, dt_b).cpu().numpy())
        else:
            preds.append(step_model(u0).cpu().numpy())
    return np.concatenate(preds, axis=0), u1_all.numpy()


def load_optional_fno(ckpt_fno: str, device: torch.device, no_compile: bool, default_path: str):
    if not ckpt_fno:
        return None, None
    if not os.path.isfile(ckpt_fno):
        if ckpt_fno == default_path:
            print(f"[eval] FNO checkpoint not found, skipping baseline: {ckpt_fno}")
            return None, None
        raise FileNotFoundError(f"FNO checkpoint not found: {ckpt_fno}")
    model_fno, ckpt_f = load_hybrid_step_map_1d(ckpt_fno, device=device)
    model_fno = compile_if_requested(model_fno, device, no_compile=no_compile)
    dt_fno = float(ckpt_f["dt"]) if str(ckpt_f.get("integrator", "")) == "u_plus_dt_rhs" else None
    print(
        f"[ckpt_fno] kind={ckpt_f.get('kind', '')}, integrator={ckpt_f.get('integrator', '')}"
        + (f", dt={dt_fno}" if dt_fno is not None else "")
    )
    return model_fno, dt_fno


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/burgers_hybrid_flowmap_dt.pt")
    ap.add_argument(
        "--ckpt_fno",
        type=str,
        default=DEFAULT_FNO_CKPT,
        help="FNO1d baseline checkpoint; ref WENO computed once for hybrid and FNO",
    )
    ap.add_argument("--eval_mode", type=str, default="rollout", choices=("rollout", "test", "both"))
    ap.add_argument("--test_path", type=str, default=None)
    ap.add_argument("--test_batch", type=int, default=256)
    ap.add_argument("--nx_low", type=int, default=256)
    ap.add_argument("--upsample", type=int, default=4)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--dt_snap", type=float, default=None)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--bc", type=str, default=None, help="reference WENO bc; default from ckpt solver_bc")
    ap.add_argument("--data_dir", type=str, default="dataset")
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--amp_min", type=float, default=0.5)
    ap.add_argument("--amp_max", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--outdir", type=str, default="eval_burgers_hybrid_out")
    ap.add_argument("--plot_one", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    model, ckpt = load_hybrid_step_map_1d(args.ckpt, device=device)
    model = compile_if_requested(model, device, no_compile=args.no_compile)
    dt_model = float(ckpt["dt"]) if str(ckpt.get("integrator", "")) == "u_plus_dt_rhs" else None
    print(
        f"[ckpt] kind={ckpt.get('kind', '')}, integrator={ckpt.get('integrator', '')}"
        + (f", dt_model={dt_model}" if dt_model is not None else "")
    )
    model_fno, dt_fno = load_optional_fno(args.ckpt_fno, device, args.no_compile, DEFAULT_FNO_CKPT)
    print(f"[params] hybrid={count_params(model):,}")
    if model_fno is not None:
        print(f"[params] fno={count_params(model_fno):,}")
    else:
        print("[params] fno=unavailable")

    run_rollout = args.eval_mode in ("rollout", "both")
    run_test = args.eval_mode in ("test", "both")
    pin_mem = device.type == "cuda"

    if run_rollout:
        dx_ckpt = float(ckpt.get("dx", 0.0))
        bc_ckpt = ckpt.get("bc", "periodic")
        solver_bc = ckpt.get("solver_bc")
        if solver_bc is None:
            solver_bc = "outflow" if bc_ckpt == "outflow" else "periodic"
        bc_eval = solver_bc if args.bc is None else args.bc

        dt_k = ckpt.get("dt")
        if args.dt_snap is not None:
            dt_snap = float(args.dt_snap)
        elif dt_k is not None:
            dt_snap = float(dt_k)
        else:
            train_pt = os.path.join(args.data_dir, "train_pv.pt")
            if os.path.isfile(train_pt):
                meta = torch.load(train_pt, map_location="cpu")["meta"]
                dt_snap = float(meta["dt"])
                print(f"[eval] dt from {train_pt}: {dt_snap}")
            else:
                raise ValueError("Pass --dt_snap or provide ckpt['dt'] or data_dir/train_pv.pt")

        times = np.arange(0.0, args.T + 1e-12, dt_snap)
        nT = len(times)
        n_steps = nT - 1

        nx_low = args.nx_low
        nx_high = args.upsample * nx_low
        x_hi = np.linspace(0.0, 1.0, nx_high, endpoint=False).astype(np.float64)
        dx_hi = 1.0 / nx_high
        dx_low = 1.0 / nx_low
        x_low = np.linspace(0.0, 1.0, nx_low, endpoint=False)

        if dx_ckpt > 0 and abs(dx_low - dx_ckpt) / max(dx_ckpt, 1e-15) > 0.05:
            print(f"[warn] dx_low={dx_low:g} vs ckpt dx={dx_ckpt:g}")

        rng = np.random.default_rng(args.seed)
        print(
            f"[eval Burgers] model_bc={bc_ckpt}, ref_bc={bc_eval}, dt={dt_snap}, "
            f"nx_low={nx_low}, upsample={args.upsample}"
        )

        l1_model = np.zeros((args.n_samples, nT))
        linf_model = np.zeros((args.n_samples, nT))
        l1_fno = None
        linf_fno = None
        if model_fno is not None:
            l1_fno = np.zeros((args.n_samples, nT))
            linf_fno = np.zeros((args.n_samples, nT))
        one_pack = None

        for s in range(args.n_samples):
            u0_hi = random_fourier_ic(
                x_hi,
                kmax=args.kmax,
                amp_range=(args.amp_min, args.amp_max),
                decay=args.decay,
                rng=rng,
            )
            snaps_hi = solve_burgers_weno_ref(u0_hi, times, dx_hi, bc=bc_eval, cfl=args.cfl)
            ref_low = np.stack([block_average(snaps_hi[j], args.upsample) for j in range(nT)], axis=0)

            u0_low = ref_low[0]
            pred = rollout_hybrid_scalar(model, u0_low, n_steps, device, pin_mem, dt_model=dt_model)
            pred_fno = (
                rollout_hybrid_scalar(model_fno, u0_low, n_steps, device, pin_mem, dt_model=dt_fno)
                if model_fno is not None
                else None
            )

            for j in range(nT):
                l1_model[s, j] = rel_l1(pred[j], ref_low[j])
                linf_model[s, j] = rel_linf(pred[j], ref_low[j])
                if pred_fno is not None:
                    l1_fno[s, j] = rel_l1(pred_fno[j], ref_low[j])
                    linf_fno[s, j] = rel_linf(pred_fno[j], ref_low[j])

            if args.plot_one and one_pack is None:
                one_pack = (
                    times.copy(),
                    ref_low.copy(),
                    pred.copy(),
                    pred_fno.copy() if pred_fno is not None else None,
                )

            print(f"[{s+1}/{args.n_samples}] done")

        save_kw = dict(
            times=times,
            dt=np.array([dt_snap]),
            dx_low=np.array([dx_low]),
            nx_low=np.array([nx_low]),
            l1_model=l1_model,
            linf_model=linf_model,
        )
        rollout_metrics = {"hybrid": {"l1": l1_model, "linf": linf_model}}
        if l1_fno is not None:
            save_kw["l1_fno"] = l1_fno
            save_kw["linf_fno"] = linf_fno
            rollout_metrics["fno"] = {"l1": l1_fno, "linf": linf_fno}
        np.savez_compressed(os.path.join(args.outdir, "metrics_hybrid_burgers.npz"), **save_kw)
        rollout_summary_path, rollout_curves_path = save_rollout_reports(args.outdir, times, rollout_metrics)

        l1_series = {"hybrid": l1_model}
        linf_series = {"hybrid": linf_model}
        if l1_fno is not None and linf_fno is not None:
            l1_series["fno"] = l1_fno
            linf_series["fno"] = linf_fno
        save_metric_curves_pdf(args.outdir, times, l1_series, metric_label="rel L1", filename="relL1_vs_time.pdf")
        save_metric_curves_pdf(args.outdir, times, linf_series, metric_label="rel Linf", filename="relLinf_vs_time.pdf")

        if one_pack is not None:
            tt, ref_traj, pred_traj, pred_fno_traj = one_pack
            pred_trajs = {"hybrid": pred_traj}
            if pred_fno_traj is not None:
                pred_trajs["fno"] = pred_fno_traj
            save_scalar_rollout_visuals(args.outdir, x_low, tt, ref_traj, pred_trajs, suffix=f"_seed{args.seed}")

        print(f"[rollout] saved csv: {rollout_summary_path}, {rollout_curves_path}")

    if run_test:
        test_path = args.test_path or os.path.join(args.data_dir, "test_pv.pt")
        if not os.path.isfile(test_path):
            raise FileNotFoundError(f"Test dataset not found: {test_path}")
        pred_h, target = evaluate_test_dataset_scalar(
            model, test_path, device, pin_mem, args.test_batch, dt_model=dt_model
        )
        test_rows = [dataset_error_row("hybrid", pred_h, target)]
        if model_fno is not None:
            pred_f, _ = evaluate_test_dataset_scalar(
                model_fno, test_path, device, pin_mem, args.test_batch, dt_model=dt_fno
            )
            test_rows.append(dataset_error_row("fno", pred_f, target))
        test_csv_path = save_test_metrics_csv(args.outdir, test_rows)
        print(f"[test] saved csv: {test_csv_path}")

    print(f"Saved under {args.outdir}")


if __name__ == "__main__":
    main()
