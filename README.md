# EchoSAT

EchoSAT is a research codebase for **event-conditioned slow-fast neural guidance for CDCL SAT solving**. It extends RLAF-style one-shot GNN guidance with solver event feedback, enhanced variable-level search state, strict total-time accounting, and a lightweight event adapter that updates solver guidance without repeatedly running a full GNN.

## Core Idea

```text
CNF formula
  -> Literal-Clause Graph
  -> Slow GNN: one-shot structural guidance
  -> CDCL rollout: collect real solver events
  -> Fast event adapter: residual weight / polarity refinement
  -> Guided CDCL solver
```

The main research direction is:

**Event-Trace Distilled Slow-Fast Guidance for CDCL SAT Solving**

The design is documented in:

- `docs/event_trace_slow_fast_guidance_scheme.md`

## What Is Included

This repository contains the necessary project files for EchoSAT:

- `src/`: data loading, GNN model, policy, solver wrappers, feedback state encoding, and training utilities
- `configs/`: Hydra configs for RLAF baselines, global feedback, and event-var adapter experiments
- `solvers/glucose/`: base Glucose source
- `solvers/glucose_weighted/`: modified Glucose source with variable weights, polarities, event tracing, and conflict-budget rollout
- `train_rlaf.py`: solver-in-the-loop GRPO/DPO training
- `evaluate_guided_solver.py`: guided solver evaluation with strict total-time accounting
- `evaluate_base_solver.py`: base solver evaluation
- `generate_data.py`: dataset generation entry point
- `build_solvers.sh`: builds the included Glucose solvers

Large local artifacts are intentionally excluded: datasets, checkpoints, runs, wandb logs, compiled objects, generated PDFs, and local test files.

## Main Features

### 1. Slow-Fast Guidance

The GNN acts as a slow structural encoder. It can cache per-variable embeddings and base guidance logits. The lightweight event adapter then refines the base guidance from solver event state:

```text
guidance_t = guidance_0 + adapter(variable_embedding, event_memory, global_state)
```

### 2. Variable-Level Event State

The modified Glucose solver can emit:

- `event_var_decisions`
- `event_var_propagations`
- `event_var_conflict_lits`
- `event_var_learnt_lits`
- `event_var_activity`

EchoSAT supports:

- legacy 5-dimensional cumulative event state
- enhanced 20-dimensional event state with cumulative, delta, rate, and rank features
- EMA event memory for multi-round refinement

### 3. Strict Total-Time Accounting

Evaluation includes:

- GNN or adapter inference time
- refinement rollout solver time
- final solver time
- total wall-clock proxy per instance

This avoids reporting a method that reduces decisions while hiding guidance overhead.

### 4. Conflict-Budget Rollouts

The weighted Glucose binary supports:

```bash
-conf-lim=<N>
```

This enables reproducible low-frequency rollouts such as 500 conflicts per feedback round.

## Installation

Create a Python environment, then install the package:

```bash
conda create -n echosat python=3.12
conda activate echosat
pip install -e .
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

Adjust the PyG wheel URL for your CUDA or CPU environment.

## Build Solvers

```bash
bash build_solvers.sh
```

This builds:

- `solvers/glucose/simp/glucose_static`
- `solvers/glucose_weighted/simp/glucose_release`

If `zlib.h` is not found but you use conda, build with:

```bash
CPATH="$CONDA_PREFIX/include" \
LIBRARY_PATH="$CONDA_PREFIX/lib" \
LD_LIBRARY_PATH="$CONDA_PREFIX/lib" \
bash build_solvers.sh
```

## Data

EchoSAT follows the RLAF data layout:

```text
data/
  training/
  validation/
  test/
```

Example paths used by configs:

```text
data/training/3sat/*/*.cnf
data/validation/3sat/*/*.cnf
data/test/3sat/350/*.cnf
```

Datasets are not committed to this repository.

## Training

### RLAF baseline

```bash
python train_rlaf.py \
  model_name=GNN_Glucose_3SAT \
  solver.solver=glucose \
  dataset.train_path='data/training/3sat/*/*.cnf' \
  dataset.val_path='data/validation/3sat/*/*.cnf'
```

### Event-var slow-fast adapter

```bash
python train_rlaf.py --config-name config_train_rlaf_event_var \
  from_checkpoint='runs/GNN_Glucose_3SAT/best.pt' \
  dataset.train_path='data/training/3sat/*/*.cnf' \
  dataset.val_path='data/validation/3sat/*/*.cnf'
```

The event-var config enables:

```yaml
training:
  feedback_state_type: event_var
  feedback_event_state_features: enhanced
  feedback_state_momentum: 0.5
  feedback_warmup_budget_type: conflicts
  feedback_warmup_conflicts: 500

model:
  var_state_dim: 20
  event_adapter:
    enabled: true
```

## Evaluation

### One-shot guided solver

```bash
python evaluate_guided_solver.py \
  checkpoint='runs/GNN_Glucose_3SAT/best.pt' \
  dataset.eval_path='data/test/3sat/350/*.cnf'
```

### Event-var feedback refinement

```bash
python evaluate_guided_solver.py --config-name config_eval_guided_solver_event_var \
  checkpoint='runs/GNN_Glucose_3SAT_EventVarFeedback/best.pt' \
  dataset.eval_path='data/test/3sat/350/*.cnf'
```

The event-var evaluation config runs two conflict-budget feedback rounds and reports strict total time.

## Suggested Experiments

Main baselines:

- Glucose
- RLAF one-shot
- global feedback refinement
- event-var multi-round GNN refinement
- one-shot GNN + event adapter
- trace-pretrained event adapter
- trace-pretrained + GRPO event adapter

Key ablations:

- shuffled event state
- random event state
- adapter without GNN embedding
- adapter without EMA memory
- activity-only / conflict-only / propagation-only / learnt-only
- different rollout budgets and numbers of refinement rounds

Recommended metrics:

- total wall-clock time
- PAR-2
- solved count
- SAT/UNSAT split
- cactus plot
- GNN / adapter / rollout / final solver time breakdown

## Repository Notes

This repository is intentionally trimmed for EchoSAT development. It does not include:

- local datasets
- checkpoints and run outputs
- wandb logs
- compiled solver binaries and object files
- paper PDFs
- local unit tests
- unrelated experiment scratch files

## References

- Learning from Algorithm Feedback: One-Shot SAT Solver Guidance with GNNs, arXiv 2025
- NeuroBack: Improving CDCL SAT Solving using Graph Neural Networks, ICLR 2024
- NeuroCore: Guiding High-Performance SAT Solvers with Unsat-Core Predictions, SAT 2019
- Graph-Q-SAT, NeurIPS 2020
- RDC-SAT, ICLR 2025
- ImitSAT, ICLR 2026 submission
- NeuroSelect, DAC 2024
