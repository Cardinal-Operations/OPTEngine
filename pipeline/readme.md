# Pipeline

This directory contains the end-to-end pipeline used to generate, rephrase, and evaluate
optimization problems in OPT-Engine.

Each script corresponds to one stage of the pipeline. Together, they transform numeric
optimization instances into natural-language problems with verified ground-truth solutions
and support systematic evaluation of LLMs.

---

## Files Overview

```text
pipeline/
├── generation.py        # Generate numeric instances and canonical problems
├── rephrase.py          # Rephrase canonical problems using LLMs
├── evaluation_ptr.py    # Evaluation with ground-truth numeric reference
├── evaluation_tir.py    # Evaluation focusing on text-induced reasoning
```

## Pipeline Flow

The typical pipeline runs in the following order:

1. `generation.py`
2. `rephrase.py`
3. `evaluation_ptr.py` or `evaluation_tir.py`

Each stage reads the output of the previous stage and writes structured JSONL files for
the next step.

## generation.py

This script generates the base optimization problems.

Main responsibilities:
- Randomly sample problem parameters for each problem type
- Solve the generated instance using Gurobi
- Filter out invalid or infeasible instances
- Convert valid numeric instances into canonical natural-language problem descriptions

The output serves as the ground-truth reference for all later stages.

## rephrase.py

This script augments canonical problems using LLM-based rephrasing.

Main responsibilities:
- Take canonical problem descriptions as input
- Rewrite them into alternative natural-language formulations
- Preserve all numerical values and optimization structure
- Increase linguistic diversity and narrative variation

The output contains multiple rephrased versions of the same underlying problem.

## evaluation_ptr.py

This script evaluates model outputs against known ground-truth solutions.

Main responsibilities:
- Compare model predictions with the true numeric solution
- Check feasibility, optimality, and numerical correctness
- Report accuracy under controlled perturbations

This evaluation focuses on whether the model recovers the correct solution.

## evaluation_tir.py

This script evaluates model behavior under text-induced reasoning.

Main responsibilities:
- Evaluate performance across different rephrasings
- Measure robustness to linguistic variation
- Analyze reasoning consistency across descriptions

This evaluation focuses on how language affects model reasoning.

## evaluation_ptr.py

Evaluates model solutions against ground-truth numeric answers.

This script runs an LLM to solve each problem, extracts the final answer, compares it with
the reference numeric solution, and reports accuracy.

## evaluation_tir.py

Evaluates model performance on rephrased problem descriptions.

This script checks whether the model produces correct solutions under different
natural-language formulations of the same problem.