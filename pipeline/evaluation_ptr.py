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
    def __init__(
        self,
        input_path: str,
        output_path: str,
        api_key: str,
        api_base: str,
        model: str = "deepseek-chat",
        max_workers: int = 16,
    ):
        """
        Evaluates LLM analytical answers by comparing extracted <answer> with true answers.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.max_workers = max_workers

        self.api_key = api_key
        self.api_base = api_base
        self.model = model
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
Below is an operations research question.
{question}
Your task is to rigorously formulate and solve the following optimization problem.

Follow this structured approach: Begin with understanding the problem -> Extract the set and parameters -> Identify the variables -> Provide the objective function -> Analyze the constraints -> Develop the mathematical model -> Solve it step-by-step, output the corresponding decision variables

Instruct:
 1.) All equations and mathematical definitions must be presented clearly.
 2.) Provide a clear, step-by-step solution process. Absolutely do not use or reference any external OR solvers or software (e.g., Gurobi, PuLP, Solver). The solution must be purely analytical.
Output the final optimal objective function value in markdown with the exact tag: <answer> optimal value here <\\answer>
"""

    def call_llm(self, prompt, retry_limit=5, retry_sleep=5):
        messages = [{"role": "user", "content": prompt}]
        for _ in range(retry_limit):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
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
        true_answer = item.get("en_answer")
        question = item.get("augmented") or item.get("en_question")

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

        try:
            true_answer = float(true_answer)
        except (ValueError, TypeError):
            pass

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

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        acc = sum([r["tag"] for r in results]) / len(results) if results else 0
        print(f"Evaluation finished | accuracy rate: {acc:.2%}")
        print(f"results saved to: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analytical Solution Evaluation Pipeline")

    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL")

    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="API key (or set env OPENAI_API_KEY)")
    parser.add_argument("--api_base", type=str, default="https://api.deepseek.com", help="Base URL, e.g. https://api.deepseek.com")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name")

    parser.add_argument("--max_workers", type=int, default=16)

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("api_key is empty. Pass --api_key or set env OPENAI_API_KEY")

    evaluator = AnalyticalEvaluator(
        input_path=args.input_path,
        output_path=args.output_path,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        max_workers=args.max_workers,
    )
    evaluator.run()


if __name__ == "__main__":
    main()