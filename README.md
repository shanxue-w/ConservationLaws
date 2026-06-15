# LGNO: A Local--Global Neural Operator for Hyperbolic Conservation Laws

This repository contains the code for **LGNO**, a Local--Global Neural Operator for
learning one-step discrete flow maps for hyperbolic conservation laws. LGNO
combines a global spectral branch with a local multiresolution branch so that the
model can capture both large-scale smooth dynamics and localized nonsmooth
features such as shocks, contact discontinuities, and steep fronts.

The repository includes one- and two-dimensional benchmark problems, FNO
baselines, training scripts, rollout/evaluation scripts, and the manuscript
source.

## Repository Layout

- `src/conslaw/`: shared model, solver, checkpoint, and evaluation utilities.
- `PureConvection/`: one-dimensional pure convection benchmark.
- `Burgers/`: one-dimensional Burgers benchmark.
- `SWE/`: one-dimensional shallow water benchmark.
- `Euler/`: one-dimensional Euler benchmark.
- `Burgers2D/`: two-dimensional Burgers benchmark.
- `Euler2D/`: two-dimensional Euler benchmarks with periodic and outflow cases.
- `test.sh`: evaluation script for trained checkpoints.

## Installation

From the repository root, install the package in editable mode and then install
the Python dependencies:

```bash
pip install -e .
pip install -r requirements.txt
```

## Training

The commands below assume a SLURM environment and should be run from the
repository root. The `train*.py` scripts train LGNO models, while the
`baseline*.py` scripts train parameter-matched FNO baselines.

### Pure Convection

```bash
cd PureConvection
python -u train.py --integrator dt --save checkpoints/pureconvection_hybrid_flowmap_dt.pt
python -u baseline.py --integrator dt --modes 24 --save checkpoints/pureconvection_fno_flowmap_dt_24.pt
cd ..
```

### Shallow Water Equations

```bash
cd SWE
python -u train.py --integrator dt --save checkpoints/swe_hybrid_flowmap_dt.pt
python -u baseline.py --integrator dt --modes 24 --save checkpoints/swe_fno_flowmap_dt_24.pt
cd ..
```

### Burgers

```bash
cd Burgers
python -u train.py --integrator dt --save checkpoints/burgers_hybrid_flowmap_dt.pt
python -u baseline.py --integrator dt --modes 24 --save checkpoints/burgers_fno_flowmap_dt_24.pt
cd ..
```

### Euler

```bash
cd Euler
python -u train.py --integrator dt --save checkpoints/euler_hybrid_flowmap_dt.pt --epochs 1000
python -u baseline.py --integrator dt --modes 24 --save checkpoints/euler_fno_flowmap_dt_24.pt --epochs 1000
cd ..
```

### Burgers 2D

```bash
cd Burgers2D
python -u train.py --epochs 200 --train_samples 10000 --val_samples 2000 --num_workers 16 --pin_memory --spec_mu 1e-3 --modes 16 --save checkpoints/burgers2d_hybrid_dt.pt
python -u baseline.py --epochs 200 --train_samples 10000 --val_samples 2000 --num_workers 16 --pin_memory --spec_mu 1e-3 --modes 24 --save checkpoints/burgers2d_fno_dt_24.pt
cd ..
```

### Euler 2D

```bash
cd Euler2D
python -u train_periodic.py --allow_tf32 --num_workers 16 --epochs 200 --batch 16 --lr 5e-4
python -u baseline_periodic.py --allow_tf32 --num_workers 16 --epochs 200 --batch 16 --lr 5e-4
python -u train_pri.py --allow_tf32 --num_workers 16 --epochs 500 --batch 16 --lr 5e-4
python -u baseline_pri.py --allow_tf32 --num_workers 16 --epochs 500 --batch 16 --lr 5e-4
cd ..
```

## Evaluation

Pretrained checkpoint files are available on the
[release page](https://github.com/shanxue-w/ConservationLaws/releases). Download
them before running evaluation and place them under the corresponding
`checkpoints/` directories.

After the corresponding checkpoints are available, run the bundled evaluation
script from the repository root:

```bash
bash test.sh
```

The core evaluation commands are listed below. All one-dimensional and Burgers2D
benchmarks run `python eval.py --eval_mode test`; Euler2D uses `eval_pri.py` for
the outflow case and `eval_periodic.py` for the periodic case.

### Test-set Evaluation

```bash
cd PureConvection
python eval.py --eval_mode test
cd ../Burgers
python eval.py --eval_mode test
cd ../SWE
python eval.py --eval_mode test
cd ../Euler
python eval.py --eval_mode test
cd ../Burgers2D
python eval.py --eval_mode test
cd ../Euler2D
python eval_pri.py --allow_tf32 --eval_mode test
python eval_periodic.py --allow_tf32 --eval_mode test
cd ..
```

### Rollout Visualization

```bash
cd PureConvection
python eval.py --eval_mode rollout --plot_one --seed 2 --T 3
cd ../Burgers
python eval.py --eval_mode rollout --plot_one --seed 2 --T 4
cd ../SWE
python eval.py --eval_mode rollout --plot_one --seed 1 --T 3
python eval.py --eval_mode rollout --plot_one --seed 2 --T 3
cd ../Euler
python eval.py --eval_mode rollout --plot_one --seed 2 --T 1
python eval.py --eval_mode rollout --plot_one --seed 2 --T 5
cd ../Burgers2D
python eval.py --eval_mode rollout --plot_one --seed 2 --T 3
cd ../Euler2D
python eval_periodic.py --allow_tf32 --eval_mode rollout --n_samples 1 --plot_one --rollout_steps 50 --sample_seed 2 --share_ref_colorbar
python eval_pri.py --allow_tf32 --eval_mode rollout --n_samples 1 --plot_one --rollout_steps 40 --sample_seed 4 --share_ref_colorbar
python eval_pri.py --demo riemann_01 --eval_mode rollout --rollout_steps 25 --allow_tf32 --plot_one
python eval_pri.py --demo riemann_02 --eval_mode rollout --rollout_steps 25 --allow_tf32 --plot_one
python eval_pri.py --demo riemann_03 --eval_mode rollout --rollout_steps 25 --allow_tf32 --plot_one
python eval_pri.py --demo riemann_04 --eval_mode rollout --rollout_steps 25 --allow_tf32 --plot_one
python eval_periodic_time.py --demo_riemann riemann_01 --rollout_steps 50 --allow_tf32 --plot_one
python eval_periodic_time.py --demo_riemann riemann_02 --rollout_steps 50 --allow_tf32 --plot_one
python eval_periodic_time.py --demo_riemann riemann_03 --rollout_steps 50 --allow_tf32 --plot_one
python eval_periodic_time.py --demo_riemann riemann_04 --rollout_steps 50 --allow_tf32 --plot_one
cd ..
```

Additional Euler2D periodic Riemann visualization demos can be run with:

```bash
cd Euler2D

cd ..
```

Individual benchmark directories also provide `eval_time.py` scripts for timing
experiments.
