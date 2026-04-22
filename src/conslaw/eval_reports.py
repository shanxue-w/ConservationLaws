from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np


def _write_csv(path: str | Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_rollout_reports(
    outdir: str | Path,
    times: np.ndarray,
    metrics: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[str, str]:
    times = np.asarray(times, dtype=np.float64)
    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    for model_name, values in metrics.items():
        l1 = np.asarray(values["l1"], dtype=np.float64)
        linf = np.asarray(values["linf"], dtype=np.float64)
        summary_rows.append(
            {
                "model": model_name,
                "n_samples": int(l1.shape[0]),
                "n_times": int(l1.shape[1]),
                "rel_l1_t0_mean": float(l1[:, 0].mean()),
                "rel_l1_final_mean": float(l1[:, -1].mean()),
                "rel_l1_final_std": float(l1[:, -1].std()),
                "rel_l1_timeavg_mean": float(l1.mean()),
                "rel_linf_final_mean": float(linf[:, -1].mean()),
                "rel_linf_final_std": float(linf[:, -1].std()),
                "rel_linf_timeavg_mean": float(linf.mean()),
            }
        )
        for j, t in enumerate(times):
            curve_rows.append(
                {
                    "model": model_name,
                    "time": float(t),
                    "rel_l1_mean": float(l1[:, j].mean()),
                    "rel_l1_std": float(l1[:, j].std()),
                    "rel_linf_mean": float(linf[:, j].mean()),
                    "rel_linf_std": float(linf[:, j].std()),
                }
            )

    outdir = Path(outdir)
    summary_path = outdir / "rollout_summary.csv"
    curves_path = outdir / "rollout_curves.csv"
    _write_csv(
        summary_path,
        [
            "model",
            "n_samples",
            "n_times",
            "rel_l1_t0_mean",
            "rel_l1_final_mean",
            "rel_l1_final_std",
            "rel_l1_timeavg_mean",
            "rel_linf_final_mean",
            "rel_linf_final_std",
            "rel_linf_timeavg_mean",
        ],
        summary_rows,
    )
    _write_csv(
        curves_path,
        ["model", "time", "rel_l1_mean", "rel_l1_std", "rel_linf_mean", "rel_linf_std"],
        curve_rows,
    )
    return str(summary_path), str(curves_path)


def _per_sample_dataset_errors(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> dict[str, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    if pred.ndim < 2:
        raise ValueError(f"Expected batched predictions, got shape {pred.shape}")

    reduce_axes = tuple(range(1, pred.ndim))
    abs_err = np.abs(pred - target)
    abs_tgt = np.abs(target)

    abs_l1 = abs_err.mean(axis=reduce_axes)
    rel_l1 = abs_l1 / (abs_tgt.mean(axis=reduce_axes) + eps)
    abs_linf = abs_err.max(axis=reduce_axes)
    rel_linf = abs_linf / (abs_tgt.max(axis=reduce_axes) + eps)

    return {
        "abs_l1": abs_l1,
        "rel_l1": rel_l1,
        "abs_linf": abs_linf,
        "rel_linf": rel_linf,
    }


def init_dataset_error_stats() -> dict[str, float]:
    return {
        "n_samples": 0.0,
        "abs_l1_sum": 0.0,
        "abs_l1_sumsq": 0.0,
        "rel_l1_sum": 0.0,
        "rel_l1_sumsq": 0.0,
        "abs_linf_sum": 0.0,
        "abs_linf_sumsq": 0.0,
        "rel_linf_sum": 0.0,
        "rel_linf_sumsq": 0.0,
    }


def update_dataset_error_stats(
    stats: dict[str, float],
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-12,
) -> None:
    values = _per_sample_dataset_errors(pred, target, eps=eps)
    n_batch = int(values["abs_l1"].shape[0])
    stats["n_samples"] += n_batch
    for key, arr in values.items():
        arr = np.asarray(arr, dtype=np.float64)
        stats[f"{key}_sum"] += float(arr.sum())
        stats[f"{key}_sumsq"] += float(np.square(arr).sum())


def finalize_dataset_error_row(model_name: str, stats: dict[str, float]) -> dict[str, object]:
    n_samples = int(stats["n_samples"])
    if n_samples <= 0:
        raise ValueError("No samples accumulated for dataset error statistics.")

    def mean_std(key: str) -> tuple[float, float]:
        mean = float(stats[f"{key}_sum"] / n_samples)
        mean_sq = float(stats[f"{key}_sumsq"] / n_samples)
        var = max(mean_sq - mean * mean, 0.0)
        return mean, float(np.sqrt(var))

    return {
        "model": model_name,
        "n_samples": n_samples,
        "abs_l1_mean": mean_std("abs_l1")[0],
        "abs_l1_std": mean_std("abs_l1")[1],
        "rel_l1_mean": mean_std("rel_l1")[0],
        "rel_l1_std": mean_std("rel_l1")[1],
        "abs_linf_mean": mean_std("abs_linf")[0],
        "abs_linf_std": mean_std("abs_linf")[1],
        "rel_linf_mean": mean_std("rel_linf")[0],
        "rel_linf_std": mean_std("rel_linf")[1],
    }


def dataset_error_row(model_name: str, pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> dict[str, object]:
    stats = init_dataset_error_stats()
    update_dataset_error_stats(stats, pred, target, eps=eps)
    return finalize_dataset_error_row(model_name, stats)


def save_test_metrics_csv(outdir: str | Path, rows: list[dict[str, object]]) -> str:
    outdir = Path(outdir)
    path = outdir / "test_metrics.csv"
    _write_csv(
        path,
        [
            "model",
            "n_samples",
            "abs_l1_mean",
            "abs_l1_std",
            "rel_l1_mean",
            "rel_l1_std",
            "abs_linf_mean",
            "abs_linf_std",
            "rel_linf_mean",
            "rel_linf_std",
        ],
        rows,
    )
    return str(path)


def _format_mean_pm_std(mean: object, std: object) -> str:
    return f"{float(mean):.6e} +/- {float(std):.6e}"


def save_test_metrics_summary_csv(outdir: str | Path, rows: list[dict[str, object]]) -> str:
    outdir = Path(outdir)
    path = outdir / "test_metrics_summary.csv"
    summary_rows = [
        {
            "model": row["model"],
            "n_samples": row["n_samples"],
            "abs_l1": _format_mean_pm_std(row["abs_l1_mean"], row["abs_l1_std"]),
            "rel_l1": _format_mean_pm_std(row["rel_l1_mean"], row["rel_l1_std"]),
            "abs_linf": _format_mean_pm_std(row["abs_linf_mean"], row["abs_linf_std"]),
            "rel_linf": _format_mean_pm_std(row["rel_linf_mean"], row["rel_linf_std"]),
        }
        for row in rows
    ]
    _write_csv(
        path,
        ["model", "n_samples", "abs_l1", "rel_l1", "abs_linf", "rel_linf"],
        summary_rows,
    )
    return str(path)
