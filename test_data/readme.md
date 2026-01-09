# Test Data

This folder (`test_data/`) contains two datasets for evaluating models on classic Operations Research (OR) optimization problems:

- **canonical/**: a canonical benchmark containing *classic* instances from **10** OR problem types.
- **perturbation/**: a derived benchmark created by applying controlled perturbations to the canonical instances.

---

## Directory Overview

```text
test_data/
├── canonical/        # Classic OR problem instances (10 types)
└── perturbation/     # Modified versions of canonical instances
    ├── ConstraintPertubations/
    ├── LinguisticComplexity/
    └── ObjectivePertubation/
```

## Perturbation Types

The `perturbation/` dataset contains three categories of controlled perturbations applied to the canonical instances:

### 1) ConstraintPertubations
Adds **one additional simple constraint** on top of the original canonical formulation, while keeping the rest of the problem unchanged.

- Purpose: evaluate whether a model can correctly incorporate minor constraint changes.
- Effect: the feasible region is slightly modified (often tightened), which may change feasibility and/or the optimal solution.

### 2) LinguisticComplexity
Modifies the **natural-language description** of the problem without changing its underlying mathematical formulation.

- Purpose: test robustness to paraphrasing and increased linguistic complexity.
- Effect: the optimization structure (variables/constraints/objective) is intended to remain equivalent to the canonical version.

### 3) ObjectivePertubation
Modifies the **objective function** of the canonical problem while keeping the rest of the formulation largely the same.

- Purpose: evaluate whether a model truly follows the optimization goal rather than relying on a memorized canonical objective.
- Effect: the optimal solution may change due to the objective shift.