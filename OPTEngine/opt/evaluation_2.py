import os
import re
import json
import time
import argparse
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


class AnalyticalEvaluator:
    def __init__(self, input_path, output_path, max_workers=16):
        """
        Evaluates LLM analytical answers by comparing extracted <answer> with true answers.
        """
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
You are a Mathematical Modeling and Optimization Consultant specializing in analytical solutions. 
Below is an operations research question:

{question}

Your task is to rigorously formulate and solve the optimization problem **analytically** and report **only the final optimal objective value** as the answer (no decision-variable values in the answer tag).

Follow this structure in your reasoning (outside the answer tag):
1) Understand the problem and restate it succinctly.  
2) Extract sets and parameters.  
3) Define decision variables (names, domains, meanings).  
4) Write the objective function (state Max or Min).  
5) List all constraints and any assumptions.  
6) Present the complete mathematical model.  
7) Solve analytically (e.g., algebraic manipulation, graphical intuition, complementary slackness, KKT, simple Simplex reasoning).  
   - **Do not** use or reference external OR solvers or software (e.g., Gurobi, PuLP, Solver).  
   - If multiple optimal solutions exist, still compute the **optimal objective value**.  
   - If the problem is **infeasible**, state this in the reasoning and set the answer tag to `infeasible`.  
   - If the problem is **unbounded**, state this in the reasoning and set the answer tag to `unbounded`.

Formatting rules:
- Provide a clear, step-by-step solution in markdown.  
- Do not include decision-variable values, code, or units inside the final answer tag.  
- The **only** content inside the final tag must be the optimal objective value (a single number), or the token `infeasible`, or `unbounded`.

Output the final result with the exact tag on a separate line:
<answer> 123.4567 <\\answer>
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

    def extract_answer(self, text):
        """
        extract the value in <answer> ... </answer> from the LLM response
        """
        if not text:
            return None
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()

    def evaluate_one(self, problem_obj):
        item = deepcopy(problem_obj)
        true_answer = item.get("true_answer")
        question = item.get("augmented") or item.get("nl_problem")  

        if question is None:
            return None

        prompt = self.prompt_template.format(question=question)
        response = self.call_llm(prompt)
        extracted_answer = self.extract_answer(response)


        def parse_num(val):
            try:
                return float(val)
            except:
                return val.lower() if isinstance(val, str) else None

        pred_value = parse_num(extracted_answer)
        true_value = parse_num(true_answer)

        if pred_value is None:
            tag = 0
        elif pred_value in ["infeasible", "unbounded"] and pred_value == true_value:
            tag = 1
        elif isinstance(pred_value, float) and isinstance(true_value, float):
            tag = 1 if abs(pred_value - true_value) <= 0.002 * abs(true_value) else 0
        else:
            tag = 0

        return {
            "type": item.get("type"),
            "size": item.get("size"),
            "true_answer": true_answer,
            "predicted_answer": pred_value,
            "tag": tag,
            "response": response,
            "question": question
        }

    def run(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            problems = [json.loads(line) for line in f]

        print(f"Start Analytical Evaluation, {len(problems)} questions in total")
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.evaluate_one, p) for p in problems]
            for future in tqdm(as_completed(futures), total=len(problems), desc="Evaluating"):
                res = future.result()
                if res:
                    results.append(res)

        with open(self.output_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        acc = sum([r["tag"] for r in results]) / len(results) if results else 0
        print(f"Evaluation finished | accuracy rate: {acc:.2%}")
        print(f"results saved to：{self.output_path}")


def run(problem_type, base_input_dir="results_try", base_output_dir="results_try", max_workers=16):
    input_path = f"/data1/LLMOptChall/LLMs-OPT/{base_input_dir}/{problem_type}/{problem_type}_rephrase_poor_1.jsonl"
    output_path = f"/data1/LLMOptChall/LLMs-OPT/{base_output_dir}/{problem_type}/{problem_type}_notool_poor_1.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    evaluator = AnalyticalEvaluator(input_path, output_path, max_workers=max_workers)
    evaluator.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analytical Solution Evaluation Pipeline")
    parser.add_argument("--problem_type", type=str, required=True)
    parser.add_argument("--base_input_dir", type=str, default="results_try")
    parser.add_argument("--base_output_dir", type=str, default="results_try")
    parser.add_argument("--max_workers", type=int, default=16)
    args = parser.parse_args()

    run(
        problem_type=args.problem_type,
        base_input_dir=args.base_input_dir,
        base_output_dir=args.base_output_dir,
        max_workers=args.max_workers
    )