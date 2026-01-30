import os
import re
import json
import time
import subprocess
from copy import deepcopy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

class GurobiCodeGenerator:
    def __init__(self, input_path, output_path, max_workers=16):
        self.input_path = input_path
        self.output_path = output_path
        self.max_workers = max_workers

        self.api_key = "sk-3e4407cc0d404e3282ed99ec348de370"
        self.api_base = "https://api.deepseek.com"
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        self.completion_kwargs = {
            "temperature": 0.5,
            "top_p": 0.95,
            "n": 1,
            "stop": [],
            "max_tokens": None
        }

        self.prompt_template = """
You are an operation research and Gurobi solver expert. Below is an operations research question. 
{question}
Build a mathematical model and corresponding Gurobi code in Python that appropriately addresses the question.
Please output Python code starting with the following lines:
```python

import gurobipy as gp
from gurobipy import GRB
```
- Make sure the model variable is named 'model'.
- Avoid using "<" and ">" in Gurobi constraints; instead, use "<=" or ">=" as appropriate.
- Carefully determine whether each variable should be integer or continuous.
- At the end of the code, after solving the model, print the objective value in this exact format (only if a solution is found): `print("Optimal value:", model.ObjVal)`
"""

    def call_llm(self, prompt, retry_limit=5, retry_sleep=5):
        messages = [{"role": "user", "content": prompt}]
        for _ in range(retry_limit):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model="deepseek-chat",
                    **self.completion_kwargs
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print("Error:", e)
                time.sleep(retry_sleep)
        return None

    def generate_code_for_one(self, problem_obj):
        question = problem_obj["augmented"]
        prompt = self.prompt_template.format(question=question)
        response = self.call_llm(prompt)

        if response:
            problem_obj.pop("original", None)
            problem_obj["response"] = response
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(problem_obj, ensure_ascii=False) + "\n")
        else:
            print(f"Failed to generate code for {problem_obj.get('type', 'unknown')}")

    def run(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            problems = [json.loads(line) for line in f]

        print(f"开始生成 Gurobi 代码，共 {len(problems)} 个问题。")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self.generate_code_for_one, problems), total=len(problems)))
        print(f"Gurobi 代码生成完成，保存在：{self.output_path}")

def extract_code_block(text):
    """从 LLM response 中提取 Gurobi Python 代码块"""
    if not text:
        return None
    code_blocks = []
    in_block = False
    current_block = []
    for line in text.splitlines():
        if line.strip().startswith("```python"):
            in_block = True
            current_block = []
            continue
        elif line.strip().startswith("```") and in_block:
            in_block = False
            code_blocks.append("\n".join(current_block))
            continue
        if in_block:
            current_block.append(line)
    return code_blocks[0] if code_blocks else None


def extract_obj_and_sol(output_str):
    """提取 print("Optimal value:", xxx) 的结果"""
    if not output_str:
        return None, None
    if any(x in output_str.lower() for x in ["infeasible", "unbounded", "no optimal solution"]):
        return None, None
    match = re.search(r"Optimal value:\s*([-+]?[0-9]*\.?[0-9]+)", output_str)
    if match:
        return float(match.group(1)), None
    return None, None

class GurobiCodeValidator:
    def __init__(self, input_path, output_path, max_workers=16):
        self.input_path = input_path
        self.output_path = output_path
        self.max_workers = max_workers

        # 标签编码定义
        self.result_key = {
            0: "wrong",
            1: "correct",
            2: "code formulation failed",
            3: "execution error",
            4: "other error"
        }

    def solve_and_tag(self, problem_obj):
        item = deepcopy(problem_obj)
        true_answer = item.get("true_answer")
        response = item.get("response")

        code = extract_code_block(response)
        if code is None:
            return self._build_output(item, tag=2, answer=None)  # code formulation failed

        try:
            result = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return self._build_output(item, tag=3, answer=None)  # execution error
        except Exception:
            return self._build_output(item, tag=3, answer=None)

        if result.returncode != 0:
            return self._build_output(item, tag=3, answer=None)  # execution error

        solver_result, _ = extract_obj_and_sol(result.stdout)

        if true_answer == "No Best Solution" and "infeasible" in result.stdout.lower():
            tag = 1  # correct
        elif isinstance(true_answer, (int, float)) and isinstance(solver_result, (int, float)):
            tag = 1 if abs(true_answer - solver_result) <= 0.002 * true_answer else 0
        elif solver_result == true_answer:
            tag = 1
        elif solver_result is None:
            tag = 4  # other error (no output, no value)
        else:
            tag = 0  # wrong

        return self._build_output(item, tag=tag, answer=solver_result)

    def _build_output(self, item, tag, answer):
        """统一输出格式"""
        return {
            "type": item.get("type"),
            "size": item.get("size"),
            "true_answer": item.get("true_answer"),
            "predicted_answer": answer,
            "tag": tag,
            "tag_description": self.result_key.get(tag, "unknown"),
            "augmented": item.get("augmented"),
            "response": item.get("response")
        }

    def run(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            problems = [json.loads(line) for line in f]

        print(f"开始验证求解，共 {len(problems)} 个问题。")
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.solve_and_tag, p) for p in problems]
            for future in tqdm(as_completed(futures), total=len(problems), desc="Validating"):
                results.append(future.result())

        with open(self.output_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"验证完成，结果保存在：{self.output_path}")

        total = len(results)
        correct = sum(1 for r in results if r["tag"] == 1)
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"Accuracy: {accuracy:.2f}%")

def run(problem_type, max_workers=16):
    print(f"开始 Evaluation Pipeline for [{problem_type}]")

    input_path = f"/data1/LLMOptChall/LLMs-OPT/results_try/{problem_type}/{problem_type}_rephrase_poor.jsonl"
    mid_path = f"/data1/LLMOptChall/LLMs-OPT/results_try/{problem_type}/{problem_type}_response_tool_poor.jsonl"
    os.makedirs(os.path.dirname(mid_path), exist_ok=True)

    codegen = GurobiCodeGenerator(input_path, mid_path, max_workers=max_workers)
    codegen.run()

    output_path = f"/data1/LLMOptChall/LLMs-OPT/results_try/{problem_type}/{problem_type}_tool_poor.jsonl"
    validator = GurobiCodeValidator(mid_path, output_path, max_workers=max_workers)
    validator.run()

    print(f"Evaluation 全流程完成 for [{problem_type}]\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gurobi Code Generation + Validation Pipeline")
    parser.add_argument("--problem_type", type=str, required=True, help="e.g., tsp, knapsack, jobshop")
    parser.add_argument("--max_workers", type=int, default=16)
    args = parser.parse_args()

    run(
        problem_type=args.problem_type,
        max_workers=args.max_workers,
    )