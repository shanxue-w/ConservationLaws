#!/usr/bin/env python3
"""
Reassemble a partial Euler2D train split from per-trajectory `.pt` files.

This is intended for the case where data generation was interrupted after many
trajectory shards had already been written to `train_trajectories/`, but before
the final monolithic train `.pt` was assembled.

By default this script:
1. reads `<source_root>/train_trajectories/*.pt`,
2. sorts them by the numeric suffix in the filename,
3. keeps the first `--count` trajectories,
4. rewrites them into `<output_root>/train_trajectories/` with continuous names
   like `train_traj_000000.pt`,
5. writes a new manifest and a monolithic assembled train `.pt`.

The original files are left untouched unless `--rename_source_inplace` is used.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import torch


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _trajectory_sort_key(path: Path) -> tuple[int, int | str]:
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return (0, int(nums[-1]))
    return (1, path.name)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Pick the first N train trajectory shards, rename them continuously, "
            "and assemble them into a monolithic train .pt."
        )
    )
    ap.add_argument(
        "--source_root",
        type=Path,
        required=True,
        help="Directory that contains train_trajectories/ and the original split files.",
    )
    ap.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help=(
            "Directory for the rebuilt subset. Default: "
            "<source_root>/reassembled_train_first<count>"
        ),
    )
    ap.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name to rebuild. Default: train.",
    )
    ap.add_argument(
        "--count",
        type=int,
        default=180,
        help="How many trajectories to keep from the sorted shard list.",
    )
    ap.add_argument(
        "--trajectory_glob",
        type=str,
        default="*.pt",
        help="Glob used inside <source_root>/<split>_trajectories. Default: *.pt.",
    )
    ap.add_argument(
        "--assemble_name",
        type=str,
        default=None,
        help="Output assembled filename. Default: euler2d_quadrant_<split>.pt.",
    )
    ap.add_argument(
        "--manifest_name",
        type=str,
        default=None,
        help="Output manifest filename. Default: euler2d_quadrant_<split>_manifest.pt.",
    )
    ap.add_argument(
        "--rename_source_inplace",
        action="store_true",
        help="Rename the selected source shard files in-place to continuous names before assembling.",
    )
    return ap


def _strip_per_trajectory_fields(meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(meta)
    for key in (
        "trajectory_index",
        "trajectory_count_in_split",
        "seed_index",
        "storage",
        "manifest_version",
        "trajectory_dir",
        "dataset_layout",
    ):
        out.pop(key, None)
    return out


def _normalize_single_trajectory_states(states: Any, src_path: Path) -> torch.Tensor:
    if not torch.is_tensor(states):
        raise SystemExit(f"`states` is not a torch.Tensor in {src_path}")

    if states.ndim == 4:
        return states

    # Some interrupted shards were saved as (1, n_snaps, 4, ny, nx).
    if states.ndim == 5 and int(states.shape[0]) == 1:
        return states[0]

    raise SystemExit(
        "Unsupported per-trajectory states shape in "
        f"{src_path}: expected (n_snaps, 4, ny, nx) or (1, n_snaps, 4, ny, nx), "
        f"got {tuple(states.shape)}"
    )


def _rename_selected_sources_inplace(selected_paths: list[Path], split: str) -> list[Path]:
    renamed_paths: list[Path] = []
    temp_paths: list[Path] = []

    # Two-step rename avoids collisions such as 65 -> 64 while 64 already exists.
    for new_idx, old_path in enumerate(selected_paths):
        temp_path = old_path.with_name(f".__tmp__{split}_traj_{new_idx:06d}.pt")
        old_path.rename(temp_path)
        temp_paths.append(temp_path)

    for new_idx, temp_path in enumerate(temp_paths):
        final_path = temp_path.with_name(f"{split}_traj_{new_idx:06d}.pt")
        temp_path.rename(final_path)
        renamed_paths.append(final_path)

    return renamed_paths


def main() -> None:
    args = _build_parser().parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be positive.")

    source_root = args.source_root.expanduser().resolve()
    split = str(args.split)
    source_traj_dir = source_root / f"{split}_trajectories"
    if not source_traj_dir.is_dir():
        raise SystemExit(f"Trajectory directory not found: {source_traj_dir}")

    if args.output_root is None:
        output_root = source_root / f"reassembled_{split}_first{args.count}"
    else:
        output_root = args.output_root.expanduser().resolve()

    output_traj_dir = output_root / f"{split}_trajectories"
    output_traj_dir.mkdir(parents=True, exist_ok=True)

    assemble_name = args.assemble_name or f"euler2d_quadrant_{split}.pt"
    manifest_name = args.manifest_name or f"euler2d_quadrant_{split}_manifest.pt"
    assemble_path = output_root / assemble_name
    manifest_path = output_root / manifest_name

    all_paths = sorted(source_traj_dir.glob(args.trajectory_glob), key=_trajectory_sort_key)
    if not all_paths:
        raise SystemExit(f"No trajectory files found in {source_traj_dir}")
    if len(all_paths) < args.count:
        raise SystemExit(
            f"Requested {args.count} trajectories, but only found {len(all_paths)} in {source_traj_dir}"
        )

    selected_paths = all_paths[: args.count]
    print(f"[reassemble] source_root = {source_root}")
    print(f"[reassemble] source_traj_dir = {source_traj_dir}")
    print(f"[reassemble] output_root = {output_root}")
    print(f"[reassemble] selected {len(selected_paths)} / {len(all_paths)} trajectories")
    print(f"[reassemble] first file = {selected_paths[0].name}")
    print(f"[reassemble] last file  = {selected_paths[-1].name}")

    original_selected_names = [p.name for p in selected_paths]
    if args.rename_source_inplace:
        selected_paths = _rename_selected_sources_inplace(selected_paths, split)
        print(f"[reassemble] renamed selected source files in-place under {source_traj_dir}")

    first_item = _torch_load(selected_paths[0])
    first_states = _normalize_single_trajectory_states(first_item["states"], selected_paths[0])

    n_snaps, n_comp, ny, nx = map(int, first_states.shape)
    assembled = torch.empty(
        (len(selected_paths), n_snaps, n_comp, ny, nx),
        dtype=first_states.dtype,
    )

    base_meta = _strip_per_trajectory_fields(dict(first_item.get("meta", {})))
    base_meta.update(
        {
            "split": split,
            "n_ic": int(len(selected_paths)),
            "n_snaps": int(n_snaps),
        }
    )

    rel_paths: list[str] = []
    n_snaps_per_traj: list[int] = []
    selected_source_files: list[str] = []

    for new_idx, src_path in enumerate(selected_paths):
        item = first_item if new_idx == 0 else _torch_load(src_path)
        states = _normalize_single_trajectory_states(item["states"], src_path)
        if tuple(states.shape) != (n_snaps, n_comp, ny, nx):
            raise SystemExit(
                "Inconsistent trajectory shape: "
                f"{src_path.name} has {tuple(states.shape)}, expected {(n_snaps, n_comp, ny, nx)}"
            )

        assembled[new_idx] = states

        traj_meta = _strip_per_trajectory_fields(dict(item.get("meta", {})))
        traj_meta.update(
            {
                "split": split,
                "n_ic": 1,
                "n_snaps": int(states.shape[0]),
                "trajectory_index": int(new_idx),
                "trajectory_count_in_split": int(len(selected_paths)),
            }
        )

        dst_name = f"{split}_traj_{new_idx:06d}.pt"
        dst_path = output_traj_dir / dst_name
        torch.save({"states": states, "meta": traj_meta}, dst_path)

        rel_paths.append(str(dst_path.relative_to(output_root)))
        n_snaps_per_traj.append(int(states.shape[0]))
        selected_source_files.append(src_path.name)

        if (new_idx + 1) % 20 == 0 or new_idx + 1 == len(selected_paths):
            print(
                f"[reassemble] wrote {new_idx + 1}/{len(selected_paths)} trajectories "
                f"-> {dst_path.name}"
            )

    manifest = {
        "meta": {
            **base_meta,
            "storage": "per_trajectory_files",
            "manifest_version": 1,
            "trajectory_dir": f"{split}_trajectories",
            "dataset_layout": "(n_ic, n_snaps, 4, ny, nx)",
            "selected_count": int(len(selected_paths)),
            "selected_source_files": selected_source_files,
            "selected_source_files_before_rename": original_selected_names,
            "source_root": str(source_root),
        },
        "trajectory_files": rel_paths,
        "trajectory_n_snaps": n_snaps_per_traj,
    }
    torch.save(manifest, manifest_path)

    assembled_data = {
        "states": assembled,
        "meta": {
            **base_meta,
            "split": split,
            "n_ic": int(len(selected_paths)),
            "selected_count": int(len(selected_paths)),
            "selected_source_files": selected_source_files,
            "selected_source_files_before_rename": original_selected_names,
            "source_root": str(source_root),
        },
    }
    torch.save(assembled_data, assemble_path)

    print(f"[reassemble] saved manifest -> {manifest_path}")
    print(f"[reassemble] saved assemble -> {assemble_path}")


if __name__ == "__main__":
    main()
