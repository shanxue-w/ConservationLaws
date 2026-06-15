#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

run_in_dir() {
    local dir="$1"
    shift
    echo "========================================"
    echo "Running in ${dir}: $*"
    echo "========================================"
    (
        cd "$dir"
        "$@"
    )
}

# PureConvection
run_in_dir PureConvection python eval.py --eval_mode rollout --plot_one --seed 2 --T 3
run_in_dir PureConvection python eval.py --eval_mode test

# Burgers
run_in_dir Burgers python eval.py --eval_mode rollout --plot_one --seed 2 --T 4
run_in_dir Burgers python eval.py --eval_mode test

# SWE
run_in_dir SWE python eval.py --eval_mode rollout --plot_one --seed 1 --T 3
run_in_dir SWE python eval.py --eval_mode rollout --plot_one --seed 2 --T 3
run_in_dir SWE python eval.py --eval_mode test

# Euler
run_in_dir Euler python eval.py --eval_mode rollout --plot_one --seed 2 --T 1
run_in_dir Euler python eval.py --eval_mode rollout --plot_one --seed 2 --T 5
run_in_dir Euler python eval.py --eval_mode test

# Burgers2D
run_in_dir Burgers2D python eval.py --eval_mode rollout --plot_one --seed 2 --T 3
run_in_dir Burgers2D python eval.py --eval_mode test

# Euler2D
run_in_dir Euler2D python eval_pri.py --allow_tf32 --eval_mode test
run_in_dir Euler2D python eval_pri.py --allow_tf32 --eval_mode rollout --n_samples 1 --plot_one --rollout_steps 40 --sample_seed 4 --share_ref_colorbar
run_in_dir Euler2D python eval_pri.py --demo riemann_01 --eval_mode rollout --rollout_steps 25 --allow_tf32 --plot_one
run_in_dir Euler2D python eval_pri.py --demo riemann_02 --eval_mode rollout --rollout_steps 25 --allow_tf32 --plot_one
run_in_dir Euler2D python eval_pri.py --demo riemann_03 --eval_mode rollout --rollout_steps 25 --allow_tf32 --plot_one
run_in_dir Euler2D python eval_pri.py --demo riemann_04 --eval_mode rollout --rollout_steps 25 --allow_tf32 --plot_one
run_in_dir Euler2D python eval_periodic.py --allow_tf32 --eval_mode test
run_in_dir Euler2D python eval_periodic.py --allow_tf32 --eval_mode rollout --n_samples 1 --plot_one --rollout_steps 50 --sample_seed 2 --share_ref_colorbar
run_in_dir Euler2D python eval_periodic_time.py --demo_riemann riemann_01 --rollout_steps 50 --plot_one
run_in_dir Euler2D python eval_periodic_time.py --demo_riemann riemann_02 --rollout_steps 50 --plot_one
run_in_dir Euler2D python eval_periodic_time.py --demo_riemann riemann_03 --rollout_steps 50 --plot_one
run_in_dir Euler2D python eval_periodic_time.py --demo_riemann riemann_04 --rollout_steps 50 --plot_one

# Time evaluations
run_in_dir PureConvection python eval_time.py --seed 2 --T 3 --solution_plot_format pdf
run_in_dir SWE python eval_time.py --seed 1 --T 3 --solution_plot_format pdf
run_in_dir Burgers python eval_time.py --seed 2 --T 4 --solution_plot_format pdf
run_in_dir Euler python eval_time.py --seed 2 --T 1 --solution_plot_format pdf
run_in_dir Burgers2D python eval_time.py --seed 2 --T 3 --solution_plot_format pdf
run_in_dir Euler2D python eval_time.py --seed 4 --T 0.4 --solution_plot_format pdf --case pri --allow_tf32
run_in_dir Euler2D python eval_time.py --seed 2 --T 0.5 --solution_plot_format pdf --case periodic --allow_tf32
