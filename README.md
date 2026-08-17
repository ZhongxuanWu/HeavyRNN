# HeavyRNN

HeavyRNN is a research-script reimplementation of the flagship experiment from
*Slow Transition to Low-Dimensional Chaos in Heavy-Tailed Recurrent Neural
Networks*. It simulates a random, untrained, one-layer tanh RNN,

\[
h_{t+1} = \tanh(W h_t), \qquad
W_{ij} = \frac{g}{N^{1/\alpha}} z_{ij},
\]

where the independent \(z_{ij}\) follow a symmetric Lévy
\(\alpha\)-stable distribution. The recurrent gain \(g\), rather than elapsed
time, is swept to locate the transition from non-chaotic to chaotic dynamics.
The autonomous flagship experiment has no time-varying input or noise: after
the weights and initial state are sampled, each trajectory is deterministic.

This implementation focuses on efficient simulation, recorded neural activity,
and Lyapunov diagnostics. Independent network trials are batched on a selected
PyTorch device. Training, plotting, noisy-input experiments, and the paper's
broader analyses are intentionally out of scope.

## Setup

The provided `agent-codex` conda environment already contains a CUDA-enabled
PyTorch stack. To install the script's runtime and test dependencies in another
environment, run from the repository root:

```bash
python -m pip install -r requirements.txt
```

CUDA is optional; use `runtime.device=cpu` for a CPU run. HeavyRNN is not an
installable Python package and does not expose a supported import API.

## Run the experiment

`run_experiment.py` is the supported interface. Start with the small smoke
configuration:

```bash
conda run -n agent-codex python run_experiment.py inspect --config configs/smoke.yaml
conda run -n agent-codex python run_experiment.py run --config configs/smoke.yaml
```

`inspect` validates the configuration and reports the expanded condition count
and estimated storage and device-memory requirements without simulating. The
paper-scale sweep is configured separately:

```bash
conda run -n agent-codex python run_experiment.py inspect --config configs/flagship.yaml
conda run -n agent-codex python run_experiment.py run --config configs/flagship.yaml
```

This implementation intentionally uses one GPU process. On a host exposing
multiple physical GPUs, select one before Python starts while continuing to use
`runtime.device=cuda:0` inside the process:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n agent-codex \
  python run_experiment.py run --config configs/flagship.yaml
```

The flagship defaults cover three network sizes, three tail indices, 50 gains,
and 10 trials, for 4,500 conditions. The activity tensors alone occupy
**8.4 GB** (decimal); the full configured array payload, including Lyapunov
diagnostics, is about **8.78 GB** (**8.17 GiB**) before NPZ-container and
filesystem overhead. Run `inspect` before launching it and confirm that the
output filesystem has sufficient free space.

Configuration uses YAML with OmegaConf dot-list overrides appended to either
command. For example:

```bash
conda run -n agent-codex python run_experiment.py run \
  --config configs/smoke.yaml \
  runtime.device=cpu sweep.trials=2 output.directory=runs/cpu-smoke

conda run -n agent-codex python run_experiment.py run \
  --config configs/flagship.yaml \
  runtime.device=cuda:0 runtime.batch_size=auto output.resume=true
```

Sweep values under `sweep`, simulation lengths and dtype under `simulation`,
device and batching under `runtime`, and artifact/resume behavior under `output`
are configurable. See the supplied YAML files for the exact keys and defaults.
The top-level support modules are internal implementation details.

## Results

Each run records enough information to inspect and reproduce every condition:

- the fully resolved configuration plus environment, Git, and numerical-source
  fingerprint metadata;
- atomic per-condition NPZ artifacts with measured activity, Lyapunov
  exponents, local stretch values, summary metrics, and random seeds;
- per-trial tabular metrics and aggregate mean/standard-deviation curves; and
- a transition table identifying the first sampled gain at which the mean
  maximum Lyapunov exponent crosses from negative to nonnegative.

Weight matrices are not saved because they can be regenerated from the recorded
condition seeds. Interrupted runs can resume without recomputing completed
conditions. Resume validation rejects artifacts produced by a different
numerical source tree, PyTorch build, CUDA build, device class, or artifact
schema, verifies NPZ payload CRCs, and rejects foreign condition files. This
prevents a long sweep from silently mixing implementations or accepting damaged
activity. Because relocating the numerical modules changes their source
fingerprint, runs made by an older package-based checkout must use a new output
directory.

Generated output belongs under `runs/` or `artifacts/`, both of which are
excluded from version control. Seeds make weights and initial states independent
of sweep ordering and trial chunking. Chaotic trajectories are not promised to
be bitwise identical when the CUDA backend, PyTorch version, or effective batch
shape changes: tiny kernel roundoff differences are exponentially amplified.
Each artifact therefore records its effective batch identifier, ordered member
list, size, and position. Compare aggregate statistics, not individual time
samples, across different execution layouts.

If a QR stretch underflows to zero, it is floored at the smallest positive
normal value of the dynamics dtype. Per-exponent and total floor counts are
stored with every condition so strongly contracting directions are never hidden.

## Development

Run the CPU test suite with:

```bash
conda run -n agent-codex pytest -m 'not cuda'
```

On a CUDA-capable machine, include the GPU sampling, simulation, and artifact
smoke tests with:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n agent-codex pytest
```

## Original study code

[`original/`](original/) is retained as an archival copy of the study's source
and supporting material. The research scripts do not modify or import it; they
are a clean, focused reimplementation of the autonomous flagship experiment.
