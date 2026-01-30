import json
import time
import os
import argparse
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from threading import Lock


class RephraseAugmentor:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        api_key: str,
        api_base: str,
        model: str = "deepseek-chat",
        max_workers: int = 16,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.max_workers = max_workers

        self.api_key = api_key
        self.api_base = api_base
        self.model = model
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

        # 用于多线程写文件避免交错
        self._write_lock = Lock()

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

    def rephrase_one_problem(self, problem_obj):
        original_problem = problem_obj["nl_problem"]
        index = problem_obj.get("index")
        problem_type = problem_obj.get("problem_type")
        problem_size = problem_obj.get("problem_size")
        answer = problem_obj.get("answer")

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
            with self._write_lock:
                with open(self.output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output, ensure_ascii=False) + "\n")
        else:
            print(f"Failed to rephrase problem at index {index}")

    def run(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            problems = [json.loads(line) for line in f]

        print(f"Start rephrasing, {len(problems)} problems in total")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as _:
            pass

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self.rephrase_one_problem, problems), total=len(problems)))

        print(f"Rephrase completed, output file saved to {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Rephrase Augmentation Pipeline")

    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL")

    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key (or set env OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="https://api.deepseek.com",
        help="Base URL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="Model name"
    )

    parser.add_argument("--max_workers", type=int, default=16)

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("api_key is empty. Pass --api_key or set env OPENAI_API_KEY")

    rephraser = RephraseAugmentor(
        input_path=args.input_path,
        output_path=args.output_path,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        max_workers=args.max_workers,
    )
    rephraser.run()


if __name__ == "__main__":
    main()