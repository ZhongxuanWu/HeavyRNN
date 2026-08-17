# HeavyRNN

HeavyRNN is a focused reimplementation of the flagship autonomous experiment
from [*Slow Transition to Low-Dimensional Chaos in Heavy-Tailed Recurrent
Neural Networks*](https://arxiv.org/abs/2505.09816). It simulates a random,
untrained, one-layer tanh RNN,

$$
h_{t+1} = \tanh(W h_t), \qquad
W_{ij} = \frac{g}{N^{1/\alpha}} z_{ij}
$$

where the independent \(z_{ij}\) follow a symmetric Lévy
\(\alpha\)-stable distribution. The recurrent gain \(g\) is swept to locate
the transition from non-chaotic to chaotic dynamics. There is no time-varying
input or noise: after the weights and initial state are sampled, each
trajectory is deterministic.

This repository provides efficient CPU/CUDA simulation, recorded neural
activity, Lyapunov diagnostics, reproducible condition seeds, and resumable
experiment artifacts. Training, plotting, noisy-input experiments, and the
paper's broader analyses are outside its current scope.

## Requirements

- Python 3.10 or newer (Python 3.11 or 3.12 is recommended)
- A CPU for the smoke configuration
- A CUDA-capable GPU with a CUDA-enabled PyTorch installation for the full
  flagship sweep; CPU execution is supported but is not practical at the
  largest configured network sizes
- Sufficient output storage: the default flagship configuration is estimated
  to produce about **8.78 GB** (**8.17 GiB**) before filesystem and
  NPZ-container overhead

The runtime dependencies are NumPy, OmegaConf, and PyTorch. Pytest is included
in `requirements.txt` for development and verification.

## Installation

Clone the repository and enter its root directory:

```bash
git clone https://github.com/ZhongxuanWu/HeavyRNN.git
cd HeavyRNN
```

Create an isolated Python environment. For example, with `venv` on Linux or
macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
Conda users can create an equivalent environment with:

```bash
conda create --name heavyrnn python=3.11 -y
conda activate heavyrnn
python -m pip install -r requirements.txt
```

The standard requirements command installs PyTorch from the package index
configured for `pip`. For GPU execution, install the PyTorch build appropriate
for your operating system, GPU, and CUDA driver using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/),
then install the remaining requirements. Confirm the installation with:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

HeavyRNN is run directly from the repository root; it is not an installable
Python package and does not expose a supported library API.

## Quick start

`run_experiment.py` is the supported command-line interface. Begin with the
small CPU smoke configuration:

```bash
python run_experiment.py inspect --config configs/smoke.yaml
python run_experiment.py run --config configs/smoke.yaml
```

`inspect` validates the resolved configuration and reports the number of
conditions plus estimated output and device-memory requirements without
starting a simulation or creating the output directory. The smoke run contains
12 small conditions and writes its results to `runs/smoke/` by default.

Run `python run_experiment.py --help` for the top-level CLI help. Both `inspect`
and `run` require a YAML configuration and accept optional OmegaConf dot-list
overrides after the configuration path.

## Run the flagship sweep

Inspect the full configuration before allocating compute or storage:

```bash
python run_experiment.py inspect --config configs/flagship.yaml
```

The default flagship configuration uses `runtime.device=cuda:0`. To run it on
the first visible GPU:

```bash
python run_experiment.py run --config configs/flagship.yaml
```

On a multi-GPU host, this program still uses one GPU process. Select a physical
GPU before Python starts and continue to address it as `cuda:0` inside the
process:

```bash
CUDA_VISIBLE_DEVICES=1 python run_experiment.py run --config configs/flagship.yaml
```

In PowerShell, the equivalent selection is:

```powershell
$env:CUDA_VISIBLE_DEVICES = "1"
python run_experiment.py run --config configs/flagship.yaml
```

The flagship defaults cover three network sizes, three tail indices, 50 gains,
and 10 trials, for 4,500 conditions. The run is computationally demanding;
check the `inspect` report and available disk space first.

A CPU run can be requested explicitly, although the full default sweep will be
very slow:

```bash
python run_experiment.py run --config configs/flagship.yaml \
  runtime.device=cpu output.directory=runs/flagship-cpu
```

## Configuration and common overrides

Sweep values are under `sweep`, numerical settings under `simulation`, device
and batching settings under `runtime`, and artifact behavior under `output`.
See [`configs/smoke.yaml`](configs/smoke.yaml) and
[`configs/flagship.yaml`](configs/flagship.yaml) for all available fields.

Examples:

```bash
# Run two CPU trials of the smoke configuration in a separate directory.
python run_experiment.py run --config configs/smoke.yaml \
  runtime.device=cpu sweep.trials=2 output.directory=runs/cpu-smoke

# Restrict a flagship-derived run to one size, two tail indices, and three gains.
python run_experiment.py run --config configs/flagship.yaml \
  'sweep.network_sizes=[1000]' \
  'sweep.alphas=[1.0,2.0]' \
  sweep.gains.mode=explicit \
  'sweep.gains.values=[0.5,0.8,1.1]' \
  output.directory=runs/selected

# Disable full activity storage when only summary and Lyapunov diagnostics are needed.
python run_experiment.py run --config configs/flagship.yaml \
  simulation.record_activity=false output.directory=runs/no-activity

# Use a fixed trial batch size instead of automatic GPU-memory-based selection.
python run_experiment.py run --config configs/flagship.yaml \
  runtime.batch_size=2 output.directory=runs/batch-two
```

Unknown configuration keys and incompatible values are rejected. Use a new
`output.directory` whenever the modeled sweep or simulation settings change.

## Resume an interrupted run

`output.resume=true` is the default. Re-run the same command with the same
output directory to validate existing condition files and continue from the
remaining conditions:

```bash
python run_experiment.py run --config configs/flagship.yaml
```

Batch size, memory fraction, compression, and the literal output path may
change safely when resuming; changes to the modeled sweep or numerical settings
require a new output directory. Resume validation checks the resolved
configuration, artifact payload CRCs, numerical-source fingerprint, PyTorch and
CUDA builds, device class, and artifact schema. It refuses to mix incompatible
or damaged results.

## Output files

Each output directory contains:

- `resolved_config.yaml`: the complete resolved configuration;
- `environment.json`: Python, PyTorch, CUDA, device, Git, and implementation
  fingerprint metadata;
- `manifest.json`: run status and progress;
- `conditions/*.npz`: atomic per-condition activity, Lyapunov spectra, local
  stretches, diagnostics, and seeds;
- `metrics.csv`: one row per trial and condition;
- `aggregates.csv`: mean and standard deviation across trials; and
- `transitions.csv`: the first sampled gain where the mean maximum Lyapunov
  exponent changes from negative to nonnegative.

Weight matrices are not stored because they can be regenerated from the saved
condition seeds. Generated output belongs under `runs/` or `artifacts/`, both
of which are ignored by Git.

Chaotic trajectories are not promised to be bitwise identical when the CUDA
backend, PyTorch version, or effective batch shape changes: tiny floating-point
differences are exponentially amplified. Compare aggregate statistics rather
than individual time samples across different execution layouts.

If a QR stretch underflows to zero, it is floored at the smallest positive
normal value of the configured dynamics dtype. Per-exponent and total floor
counts are stored so strongly contracting directions are not hidden.

## Lyapunov implementation

For each measured transition, the implementation evaluates the Jacobian at the
state that generated that transition,

\[
J_t = \operatorname{diag}\!\left(\operatorname{sech}^2(W h_t)\right)W,
\]

and performs the canonical forward tangent update

\[
Q_{t+1}R_{t+1}=J_tQ_t.
\]

The same preactivation is used for both the state update and its derivative.
The implementation does not use the transposed-Jacobian update.

## Development and tests

Run all tests that do not require CUDA:

```bash
python -m pytest -m "not cuda"
```

With a working CUDA-enabled PyTorch installation, run the complete suite with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest
```

## Original study code

[`original/`](original/) is retained as an archival copy of the study's public
source and supporting material. The reimplementation does not modify or import
it.
