benchmark_gurobi_prompts= {
"zeroshot_q2mc_en": f"""
You are an operation research and gurobi solver expert. Below is an operations research question:
{{Question}}
Carefully analyze the problem to identify the core elements such as Decision Variables, Objective Function, and Constraints,
determine whether the variable is an integer or a continuous variable; 
Build a mathematical model and corresponding gurobi python code start with:
```python\n
import gurobipy as gp\n
from gurobipy import GRB\n```
- Make sure the model variable is named 'model'.
- Avoid using "<" and ">" in Gurobi constraints; instead, use "<=" or ">=" as appropriate.
Think step by step to ensure flawless translation from math to code.
""",
"zeroshot_verbal_en": f"""
You are a Mathematical Modeling and Optimization Consultant specializing in analytical solutions.
Below is an operations research question.
{{Question}}
Your task is to rigorously formulate and solve the following optimization problem.

Follow this structured approach: Begin with understanding the problem -> Extract the set and parameters -> Identify the variables -> Provide the objective function -> Analyze the constraints -> Develop the mathematical model -> Solve it step-by-step, output the corresponding decision variables

Instruct:
 1.) All equations and mathematical definitions must be presented clearly.
 2.) Provide a clear, step-by-step solution process. Absolutely do not use or reference any external OR solvers or software (e.g., Gurobi, PuLP, Solver). The solution must be purely analytical.
Output the final optimal objective function value in markdown with the exact tag: <answer> optimal value here </answer>
"""
}

