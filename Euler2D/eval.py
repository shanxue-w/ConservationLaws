"""Evaluate Euler2D dt-step checkpoints on trajectory test data."""

from __future__ import annotations

import argparse
import gc
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
DEFAULT_HYBRID_CKPT = "checkpoints/euler2d_hybrid_outflow_cnn_dt.pt"
DEFAULT_FNO_CKPT = "checkpoints/euler2d_fno_dt.pt"
DEFAULT_CNN_CKPT = "checkpoints/euler2d_cnn_dt.pt"
RHO_MIN = 1e-6
P_MIN = 1e-8


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

    model_cnn, ckpt_cnn = load_optional_model(
        args.ckpt_cnn,
        device,
        no_compile=args.no_compile,
        default_path=DEFAULT_CNN_CKPT,
    )
    if model_cnn is not None and ckpt_cnn is not None:
        yield model_label_from_ckpt(ckpt_cnn), model_cnn, ckpt_cnn


def model_label_from_ckpt(ckpt: dict) -> str:
    kind = str(ckpt.get("kind", ""))
    if "cnn" in kind:
        return "cnn"
    if "fno" in kind:
        return "fno"
    return "hybrid"


def display_model_name(name: str) -> str:
    return {
        "ref": "Ref",
        "hybrid": "Hybrid",
        "fno": "FNO",
        "cnn": "CNN",
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
    rho = np.maximum(np.asarray(state[0], dtype=np.float64), RHO_MIN)
    rho_u = np.asarray(state[1], dtype=np.float64)
    rho_v = np.asarray(state[2], dtype=np.float64)
    energy = np.asarray(state[3], dtype=np.float64)
    u_vel = rho_u / rho
    v_vel = rho_v / rho
    kinetic = 0.5 * (rho_u * rho_u + rho_v * rho_v) / rho
    pressure = np.maximum((gamma - 1.0) * (energy - kinetic), P_MIN)
    return {
        "rho": rho,
        "u": u_vel,
        "v": v_vel,
        "p": pressure,
    }


def state_to_conserved_fields(state: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "rho": np.asarray(state[0], dtype=np.float64),
        "rhou": np.asarray(state[1], dtype=np.float64),
        "rhov": np.asarray(state[2], dtype=np.float64),
        "E": np.asarray(state[3], dtype=np.float64),
    }


def save_state_panel(
    out_path: str,
    fields: dict[str, np.ndarray],
    *,
    title: str,
    cmap: str = "viridis",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, key in zip(axes.flat, fields.keys()):
        im = ax.imshow(fields[key], origin="lower", cmap=cmap)
        ax.set_title(key)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_error_map(out_path: str, err: np.ndarray, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    vmax = max(float(err.max()), 1e-12)
    im = ax.imshow(err, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_rollout_sample_plots(
    outdir: str,
    ref_traj: np.ndarray,
    pred_trajs: dict[str, np.ndarray],
    *,
    final_time: float,
    gamma: float,
    tag: str,
) -> None:
    ref_final = ref_traj[-1]
    final_states = {"ref": ref_final}
    for name, traj in pred_trajs.items():
        final_states[name] = traj[-1]

    conserved_fields = {name: state_to_conserved_fields(state) for name, state in final_states.items()}
    primitive_fields = {name: conserved_to_primitive_2d(state, gamma=gamma) for name, state in final_states.items()}

    for name, fields in conserved_fields.items():
        out_path = os.path.join(outdir, f"conserved_{name}_{tag}.pdf")
        save_state_panel(
            out_path,
            fields,
            title=f"{display_model_name(name)} conserved, t={final_time:g}, {tag}",
        )

    for name, fields in primitive_fields.items():
        out_path = os.path.join(outdir, f"primitive_{name}_{tag}.pdf")
        save_state_panel(
            out_path,
            fields,
            title=f"{display_model_name(name)} primitive, t={final_time:g}, {tag}",
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
                title=f"|{display_model_name(name)} - Ref| {key}, t={final_time:g}, {tag}",
            )
        for key, values in primitive_fields[name].items():
            err = np.abs(values - ref_prim[key])
            save_error_map(
                os.path.join(outdir, f"error_{key}_{name}_{tag}.pdf"),
                err,
                title=f"|{display_model_name(name)} - Ref| {key}, t={final_time:g}, {tag}",
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_HYBRID_CKPT)
    ap.add_argument("--ckpt_fno", type=str, default=DEFAULT_FNO_CKPT)
    ap.add_argument("--ckpt_cnn", type=str, default=DEFAULT_CNN_CKPT)
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
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    configure_runtime(device, allow_tf32=args.allow_tf32)
    pin_mem = device.type == "cuda"
    run_rollout = args.eval_mode in ("rollout", "both")
    run_test = args.eval_mode in ("test", "both")

    models: dict[str, tuple[torch.nn.Module, dict]] = {}
    if run_rollout:
        primary_model, primary_ckpt = load_required_model(args.ckpt, device, no_compile=args.no_compile)
        primary_name = model_label_from_ckpt(primary_ckpt)
        models[primary_name] = (primary_model, primary_ckpt)

        model_fno, ckpt_fno = load_optional_model(
            args.ckpt_fno,
            device,
            no_compile=args.no_compile,
            default_path=DEFAULT_FNO_CKPT,
        )
        if model_fno is not None and ckpt_fno is not None:
            models[model_label_from_ckpt(ckpt_fno)] = (model_fno, ckpt_fno)

        model_cnn, ckpt_cnn = load_optional_model(
            args.ckpt_cnn,
            device,
            no_compile=args.no_compile,
            default_path=DEFAULT_CNN_CKPT,
        )
        if model_cnn is not None and ckpt_cnn is not None:
            models[model_label_from_ckpt(ckpt_cnn)] = (model_cnn, ckpt_cnn)

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
        print(f"[rollout] steps={rollout_steps}/{max_rollout_steps}")
        for name, (model, _) in models.items():
            l1 = np.zeros((n_eval, len(times)), dtype=np.float64)
            linf = np.zeros((n_eval, len(times)), dtype=np.float64)
            for idx in range(n_eval):
                ref = test_states[idx, : rollout_steps + 1].cpu().numpy()
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
            plot_tag = sample_tag(int(selected_idx[0]), args.sample_seed)
            plot_models = {k: v for k, v in one_preds.items() if k in ("hybrid", "fno", "cnn")}
            save_rollout_sample_plots(
                args.outdir,
                one_ref,
                plot_models,
                final_time=float(times[-1]),
                gamma=gamma,
                tag=plot_tag,
            )

    if run_test:
        if run_rollout:
            models.clear()
            primary_model = None
            primary_ckpt = None
            model_fno = None
            ckpt_fno = None
            model_cnn = None
            ckpt_cnn = None
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
