# Modules

The `modules/` directory contains **instance generators** for constructing optimization problems used in the benchmark.
Each module corresponds to a specific *problem variant* and is responsible for generating **fully validated, natural-language optimization problems**.

All generators follow a unified three-stage pipeline:
1. **Random parameter sampling**
2. **Exact solution and feasibility verification via Gurobi**
3. **Natural-language mapping using structured templates**

---

## Directory Overview

```text
modules/
├── modules_canonical/   # Canonical problem generators
├── modules_con/         # Canonical problems with extra constraints
├── modules_obj/         # Objective-perturbed problem generators
├── modules_ppl/         # Linguistic-complexity-perturbed generators
└── base_generator.py    # Shared base class and utilities
```

## Generator Pipeline (Common to All Modules)

Each generator implements the same core workflow:

### 1) Parameter Generation

Problem-specific parameters (e.g., costs, capacities, demands, graph weights) are **randomly sampled** from predefined ranges.
This step controls the **instance diversity** while ensuring numerical stability.

### 2) Exact Solving and Validation

For each sampled parameter set:
- A corresponding **Gurobi model** is constructed.
- The model is solved to optimality.
- Instances are **filtered** to ensure:
  - feasibility (or well-defined infeasibility/unboundedness),
  - existence of a valid optimal solution when required.

This guarantees that every generated instance has a **ground-truth solution**.

### 3) Natural-Language Mapping

Validated parameters are mapped into a **natural-language optimization problem** using structured mapping templates.
This step converts mathematical formulations into readable problem descriptions while preserving:
- variables,
- constraints,
- objective structure.

## Module Descriptions

### `modules_canonical/`

Generators for **canonical optimization problems**.

- Produces standard formulations for each OR problem type.
- Serves as the foundation for all downstream perturbations.
- Mathematical structure and objective follow the textbook definition.

### `modules_con/`

Generators for **constraint-augmented canonical problems**.

- Adds **one additional simple constraint** on top of the canonical formulation.
- Keeps the original objective and variables unchanged.
- Used to test sensitivity to minor constraint modifications.

### `modules_obj/`

Generators for **objective-perturbed problems**.

- Modifies the **objective function** while largely preserving the original constraints.
- Forces models to correctly re-interpret the optimization goal.
- The optimal solution may differ from the canonical version.

### `modules_ppl/`

Generators for **linguistic-complexity-perturbed problems**.

- Changes only the **natural-language description**.
- Increases lexical variety, syntactic depth, or narrative complexity.
- The underlying mathematical formulation remains equivalent to the canonical problem.