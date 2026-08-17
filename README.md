# Gradient-Balanced Federated Continual Learning

A research implementation of **Federated Continual Learning (FCL)** for studying catastrophic forgetting under
heterogeneous client data, with a gradient-based adaptive retention mechanism.

The project evaluates standard federated continual learning, replay, Learning without Forgetting (LwF), fixed
Replay+LwF, an adapted Fed-A-GEM gradient-projection baseline, and a proposed **Gradient-Balanced Replay+LwF** approach
on class-incremental CIFAR-10.

The proposed method dynamically adjusts retention strength using gradient-balance information rather than relying on a
single fixed distillation weight.

---

## Overview

Federated Learning (FL) enables multiple clients to collaboratively train a global model without directly sharing their
local datasets.

Continual Learning (CL) considers systems that learn a sequence of tasks over time.

Combining the two produces **Federated Continual Learning**, where distributed clients must learn new tasks while
retaining knowledge acquired from previous tasks.

A major challenge is **catastrophic forgetting**: learning a new task can substantially degrade performance on
previously learned tasks.

This repository investigates that problem using:

- FedAvg-based Federated Continual Learning
- Experience Replay
- Learning without Forgetting (LwF)
- Replay + LwF
- Adapted Fed-A-GEM gradient projection
- Gradient diagnostics
- Gradient-Balanced adaptive retention

---

## Proposed Method

### Gradient-Balanced Replay+LwF

The main method extends Replay+LwF by replacing the fixed retention strength with an adaptive weight.

During training, gradient information associated with old and new knowledge is used to estimate the balance between:

- retaining previously learned information, and
- adapting to the current task.

The resulting signal controls the strength of the retention objective within a bounded interval.

For the main experiments:

```text
Minimum retention weight = 0.5
Maximum retention weight = 1.5
```

The observed adaptive weights across the five main experimental seeds ranged from:

```text
1.0619 to 1.4903
```

with:

```text
Mean = 1.2963
Standard deviation = 0.0951
```

This shows that the mechanism does not collapse to the fixed Replay+LwF coefficient of 1.0.

---

## Experimental Setting

The main experiments use CIFAR-10 in a class-incremental federated continual-learning setting.

### Dataset

```text
Dataset: CIFAR-10
Number of classes: 10
```

The classes are divided into five sequential tasks:

```text
Task 1: Classes 0 and 1
Task 2: Classes 2 and 3
Task 3: Classes 4 and 5
Task 4: Classes 6 and 7
Task 5: Classes 8 and 9
```

### Federated Configuration

```text
Clients: 5
Tasks: 5
Classes per task: 2
Federated rounds per task: 3
Local epochs: 1
Batch size: 64
Replay memory size: 500
LwF temperature: 2.0
Fixed LwF weight: 1.0
```

Non-IID client distributions are generated using Dirichlet partitioning.

The primary setting uses:

```text
Dirichlet alpha = 0.5
```

Additional heterogeneity experiments evaluate:

```text
alpha = 0.1
alpha = 0.5
alpha = 1.0
```

---

## Methods Compared

### FedAvg-FCL

Federated continual-learning baseline without an explicit continual-learning retention mechanism.

### Replay

Stores examples from previous tasks and reuses them during subsequent training.

### Learning without Forgetting (LwF)

Uses knowledge distillation to encourage the current model to preserve predictions associated with previously learned
knowledge.

### Replay + LwF

Combines replay memory with a fixed LwF distillation weight.

### Adapted Fed-A-GEM

Implements an adapted gradient-projection baseline inspired by Fed-A-GEM.

Clients maintain replay buffers and compute reference gradients from remembered samples. When the gradient for
current-task learning conflicts with the global reference gradient, the update is projected to reduce destructive
interference.

The baseline is adapted to the same fixed protocol used in this repository:

```text
Clients: 5
Tasks: 5
Rounds per task: 3
Local epochs: 1
Memory size: 500
Dirichlet alpha: 0.5
```

A learning-rate sweep was performed at Seed 42:

```text
0.001
0.01
0.1
```

The best tested configuration was:

```text
Learning rate = 0.001
```

The implementation should therefore be interpreted as an **adapted Fed-A-GEM baseline**, not as a bit-for-bit
reproduction of the original implementation.

### Gradient-Balanced Replay + LwF

Uses Replay+LwF while dynamically adapting retention strength using gradient-balance information.

---

# Results

## Ablation Study

Three seeds were used for the initial ablation study.

| Method       | Final Average Accuracy | Average Forgetting |
|:-------------|-----------------------:|-------------------:|
| FedAvg-FCL   |          16.42 ± 1.04% |    68.75 ± 2.84 pp |
| Replay       |          19.33 ± 4.72% |    65.55 ± 2.79 pp |
| LwF          |          19.89 ± 1.84% |    43.66 ± 5.93 pp |
| Replay + LwF |      **46.33 ± 4.08%** | **5.23 ± 0.56 pp** |

Replay and LwF individually provide limited protection in this setting, whereas their combination produces a large
improvement over the FedAvg-FCL baseline.

---

## Five-Seed Main Comparison

The primary comparison uses five seeds:

```text
42
123
2026
777
1001
```

| Method             | Final Average Accuracy | Average Forgetting |
|:-------------------|-----------------------:|-------------------:|
| Adapted Fed-A-GEM  |          13.15 ± 0.45% |    60.00 ± 4.46 pp |
| Fixed Replay + LwF |      **46.94 ± 3.40%** |     4.76 ± 0.76 pp |
| Gradient-Balanced  |          44.76 ± 3.50% | **1.77 ± 0.34 pp** |

The adapted Fed-A-GEM baseline reduces forgetting somewhat relative to the FedAvg-FCL baseline, but severe catastrophic
forgetting remains under this experimental protocol.

Gradient-Balanced FCL reduces average forgetting from:

```text
4.76 pp -> 1.77 pp
```

relative to fixed Replay+LwF.

This corresponds to:

```text
62.89% relative reduction in forgetting
```

The improved retention is accompanied by a reduction of:

```text
2.18 percentage points
```

in final average accuracy.

The results therefore indicate a **stability-plasticity trade-off** rather than uniform superiority across all metrics.

---

## Per-Seed Fixed vs. Gradient-Balanced Comparison

| Seed | Fixed Accuracy | GB Accuracy | Fixed Forgetting | GB Forgetting |
|-----:|---------------:|------------:|-----------------:|--------------:|
|   42 |         46.57% |      45.77% |          4.80 pp |       2.01 pp |
|  123 |         42.14% |      40.22% |          5.03 pp |       1.85 pp |
| 2026 |         50.28% |      46.60% |          5.86 pp |       1.68 pp |
|  777 |         50.12% |      48.97% |          3.90 pp |       2.07 pp |
| 1001 |         45.60% |      42.26% |          4.23 pp |       1.23 pp |

Gradient-Balanced reduces average forgetting relative to the fixed method for all five tested seeds.

---

## Adapted Fed-A-GEM Five-Seed Results

|          Seed | Final Average Accuracy |  Average Forgetting |
|--------------:|-----------------------:|--------------------:|
|            42 |                 13.52% |            59.29 pp |
|           123 |                 13.71% |            56.06 pp |
|          2026 |                 12.92% |            56.58 pp |
|           777 |                 12.64% |            67.15 pp |
|          1001 |                 12.98% |            60.93 pp |
| **Mean ± SD** |      **13.15 ± 0.45%** | **60.00 ± 4.46 pp** |

The adapted Fed-A-GEM result should be interpreted in the context of this repository's fixed task-based protocol and
architecture.

---

## Statistical Analysis

For final average accuracy:

```text
Mean paired difference (GB - Fixed): -2.18 pp
95% CI: [-3.78, -0.58] pp
Cohen's dz: -1.692
Exact two-sided sign-flip p-value: 0.0625
```

For average forgetting:

```text
Mean reduction (Fixed - GB): +3.00 pp
95% CI: [+1.94, +4.05] pp
Cohen's dz: +3.538
Exact two-sided sign-flip p-value: 0.0625
```

Only five paired seeds are available, so inferential power is limited. The paired differences, confidence intervals,
consistency across seeds, and effect sizes should therefore be interpreted together with the p-values.

---

# Heterogeneity Robustness

Additional experiments examine different levels of client heterogeneity using Seed 42.

| Dirichlet α | Fixed Accuracy | GB Accuracy | Accuracy Difference | Fixed Forgetting | GB Forgetting | Forgetting Reduction |
|------------:|---------------:|------------:|--------------------:|-----------------:|--------------:|---------------------:|
|         0.1 |         32.19% |      28.28% |            -3.91 pp |          7.56 pp |       3.99 pp |             +3.58 pp |
|         0.5 |         46.57% |      45.77% |            -0.80 pp |          4.80 pp |       2.01 pp |             +2.79 pp |
|         1.0 |         48.29% |      44.46% |            -3.83 pp |          5.15 pp |       1.47 pp |             +3.68 pp |

Relative forgetting reductions are:

```text
alpha = 0.1: 47.27%
alpha = 0.5: 58.07%
alpha = 1.0: 71.36%
```

Across these Seed-42 experiments, Gradient-Balanced FCL consistently reduces forgetting while sacrificing some final
average accuracy.

---

# Adaptive Retention Analysis

Across five seeds, 300 adaptive-weight observations were analyzed.

```text
Minimum weight: 1.0619
Maximum weight: 1.4903
Mean weight: 1.2963
Standard deviation: 0.0951
```

## By Task

| Task | Mean Weight | Standard Deviation |
|-----:|------------:|-------------------:|
|    2 |      1.3892 |             0.0701 |
|    3 |      1.2129 |             0.0807 |
|    4 |      1.2528 |             0.0571 |
|    5 |      1.3302 |             0.0548 |

## By Federated Round

| Round | Mean Weight | Standard Deviation |
|------:|------------:|-------------------:|
|     1 |      1.3594 |             0.0729 |
|     2 |      1.2735 |             0.0863 |
|     3 |      1.2560 |             0.0916 |

The results indicate stronger task- and round-dependent adaptation than persistent client-specific differences.

---

# Gradient Diagnostics

The development of the adaptive mechanism included several diagnostic experiments.

Update-conflict analysis across 60 observations produced:

```text
Pearson correlation with forgetting: +0.4228
Spearman correlation with forgetting: +0.4175
```

Gradient-signal analysis found the strongest investigated signal to be the old/new gradient magnitude ratio:

```text
Pearson correlation: -0.4731
Spearman correlation: -0.4421
Mean absolute correlation: 0.4576
```

These diagnostics suggest that gradient information contains useful but incomplete information about future forgetting,
motivating adaptive rather than purely fixed retention.

---

# Figures

Publication-oriented figures are stored in:

```text
figures/
```

The repository includes figures for:

- Five-seed final average accuracy
- Five-seed average forgetting
- Heterogeneity robustness
- Relative forgetting reduction
- Task-wise retention
- Stage-wise continual performance
- Final per-task retention
- Adaptive retention weights by task
- Adaptive retention weights by round
- Adaptive retention weights by client
- Adaptive-weight heatmaps
- Gradient magnitude ratio versus retention weight

Both PNG and PDF versions are available for the main publication figures.

---

# Project Structure

```text
federated-continual-learning/
│
├── src/
│   ├── algorithms/
│   ├── clients/
│   │   ├── client.py
│   │   ├── continual_client.py
│   │   ├── replay_lwf_client.py
│   │   ├── gradient_balanced_client.py
│   │   └── fed_agem_client.py
│   ├── data/
│   ├── models/
│   ├── server/
│   └── utils/
│
├── scripts/
│   ├── analysis/
│   │   ├── comparison scripts
│   │   ├── diagnostic scripts
│   │   ├── statistical tests
│   │   ├── Fed-A-GEM summary
│   │   ├── experiment manifest generation
│   │   └── paper-table generation
│   │
│   └── plots/
│       ├── result plotting
│       ├── task-retention plotting
│       └── adaptive-weight plotting
│
├── data/
├── experiments/
├── figures/
├── results/
│
├── main.py
├── run_fcl_baseline.py
├── run_fcl_replay.py
├── run_fcl_lwf.py
├── run_fcl_replay_lwf.py
├── run_fcl_gradient_balanced.py
├── run_fcl_fed_agem.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Installation

Python 3 is required.

Clone the repository and create a virtual environment.

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The main Python dependencies are:

```text
torch==2.8.0
torchvision==0.23.0
numpy
pandas
matplotlib
```

The experiments were developed using PyTorch 2.8.0 and torchvision 0.23.0.

---

# Running Experiments

Run commands from the repository root so that relative paths to `results/`, `figures/`, and `src/` resolve correctly.

## FedAvg-FCL Baseline

```bash
python run_fcl_baseline.py
```

## Replay

```bash
python run_fcl_replay.py
```

## LwF

```bash
python run_fcl_lwf.py
```

## Fixed Replay + LwF

```bash
python run_fcl_replay_lwf.py
```

## Gradient-Balanced FCL

```bash
python run_fcl_gradient_balanced.py
```

## Adapted Fed-A-GEM

```bash
python run_fcl_fed_agem.py
```

Experimental outputs are written to:

```text
results/
```

---

# Analysis Scripts

Generate the Fed-A-GEM five-seed summary:

```bash
python scripts/analysis/summarize_fed_agem.py
```

Generate paper-oriented tables:

```bash
python scripts/analysis/create_fcl_paper_tables.py
```

Regenerate and verify the experiment manifest:

```bash
python scripts/analysis/create_experiment_manifest.py
```

---

# Reproducibility

The repository contains an experiment manifest:

```text
results/experiment_manifest.csv
```

The manifest records the experimental configuration and corresponding result/model artifact for the experiments used
during the study.

At the current reproducibility checkpoint:

```text
Experiments recorded: 28
Missing result files: 0
Missing model files: 0
```

Model checkpoints are excluded from Git tracking by default because of their size. Numerical result CSV files and
publication figures remain tracked.

The adapted Fed-A-GEM summary is available as:

```text
results/fcl_fed_agem_5seed_summary.csv
```

Paper-oriented summary tables are available as:

```text
results/paper_table_ablation.csv
results/paper_table_main_5seeds.csv
results/paper_table_heterogeneity.csv
results/paper_table_adaptive_weights.csv
results/paper_table_statistics.csv
```

Fed-A-GEM learning-rate diagnostics and five-seed experiment outputs are also stored under:

```text
results/
```

---

# Main Findings

The current experiments support the following observations:

1. Vanilla FedAvg-FCL experiences severe catastrophic forgetting in the evaluated class-incremental setting.
2. Replay and LwF individually provide limited protection against severe forgetting, while combining them produces a
   much stronger baseline.
3. The adapted Fed-A-GEM gradient-projection baseline reduces forgetting somewhat relative to FedAvg-FCL but still
   exhibits severe forgetting under this protocol.
4. Gradient-Balanced FCL reduces average forgetting relative to fixed Replay+LwF across all five tested seeds.
5. Gradient-Balanced reduces average forgetting from 4.76 pp to 1.77 pp, corresponding to a 62.89% relative reduction.
6. This retention improvement comes with a 2.18 percentage-point decrease in final average accuracy.
7. Adaptive retention weights vary substantially across tasks and federated rounds, indicating that the mechanism does
   not collapse to a constant coefficient.

The proposed approach should therefore be interpreted as a mechanism for **strengthening retention adaptively**, rather
than as a method that dominates every baseline on every performance metric.

---

# Limitations

The current experimental study has several limitations:

- CIFAR-10 is the primary dataset.
- The main experiments use five clients.
- Five seeds are used for the primary Gradient-Balanced comparison.
- Five seeds are used for the adapted Fed-A-GEM comparison after a three-point Seed-42 learning-rate sweep.
- The initial ablation study uses three seeds.
- Heterogeneity robustness experiments across multiple Dirichlet alpha values currently use Seed 42.
- Only three federated rounds are performed per continual task.
- The adaptive mechanism improves retention while reducing final average accuracy.
- The Fed-A-GEM implementation is adapted to this repository's task-based protocol and is not a bit-for-bit reproduction
  of the original method.
- Larger datasets, additional architectures, more clients, additional FCL baselines, and broader hyperparameter studies
  remain future work.

These limitations should be considered when interpreting the generality of the results.

---

# Research Status

This repository contains an experimental research implementation.

The current results are intended to support a research manuscript on gradient-aware adaptive retention for federated
continual learning.

The project is under active research development, and the current experiments should not be interpreted as establishing
state-of-the-art performance.

---

# License

A license has not yet been selected for this research repository.