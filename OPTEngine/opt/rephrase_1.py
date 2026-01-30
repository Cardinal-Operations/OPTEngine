import json
import time
import os
import argparse
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class RephraseAugmentor:
    def __init__(self, input_path, output_path, max_workers=16):
        self.input_path = input_path
        self.output_path = output_path
        self.max_workers = max_workers

        self.api_key = "sk-3e4407cc0d404e3282ed99ec348de370"
        self.api_base = "https://api.deepseek.com"
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        self.completion_kwargs = {
            "temperature": 1.2,
            "top_p": 0.95,
            "n": 1,
            "stop": [],
            "max_tokens": None
        }

        self.prompt_template = """You are an expert in operations research problem design and NLP data augmentation.
Your task is to take the following optimization problem and rewrite it according to the instructions.
### Original Problem:
\"\"\"{original_problem}\"\"\"
### Instructions:
- Rewrite the problem in a **different real-world scenario or application context**, while preserving its **mathematical structure, optimization goal, and logical constraints**.
- All **numerical values, quantities, and parameter relationships must remain exactly the same**.
- Use **different terminology, phrasing, and narrative style** to describe the problem, but ensure that the underlying model and relationships are identical.
- Do not add or remove any mathematical constraints, variables, or objectives.
- The rewritten problem should read naturally and clearly as a self-contained description in the new scenario.
- Do not include any explanations, reasoning, or headers.
- Output only the rewritten problem description, without commentary.
- Slightly increase the perplexity of description the question.
### Output:
[Start your output below. No headers, no comments.]
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

    def rephrase_one_problem(self, problem_obj):
        original_problem = problem_obj["nl_problem"]
        index = problem_obj["index"]
        problem_type = problem_obj["problem_type"]
        problem_size = problem_obj["problem_size"]
        answer = problem_obj["answer"]

        prompt = self.prompt_template.format(original_problem=original_problem)
        result = self.call_llm(prompt)

        if result:
            output = {
                "type": problem_type,
                "size": problem_size,
                "original": original_problem,
                "augmented": result,
                "true_answer": answer
            }
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
        else:
            print(f"Failed to rephrase problem at index {index}")

    def run(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            problems = [json.loads(line) for line in f]

        print(f"Start rephrasing, {len(problems)} problems in total")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self.rephrase_one_problem, problems), total=len(problems)))
        print(f"Rephrase completed, output file saved to {self.output_path}")

def run(problem_type, base_input_dir="results_try", base_output_dir="results_try", max_workers=16):
    input_path = f"/data1/LLMOptChall/LLMs-OPT/{base_input_dir}/{problem_type}/{problem_type}_nl_poor_1.jsonl"
    output_path = f"/data1/LLMOptChall/LLMs-OPT/{base_output_dir}/{problem_type}/{problem_type}_rephrase_poor_1.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rephraser = RephraseAugmentor(input_path, output_path, max_workers=max_workers)
    rephraser.run()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
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