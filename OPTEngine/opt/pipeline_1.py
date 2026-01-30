import argparse
import os
import sys
from generation import generate_instances
from rephrase_1 import run as rephrase_run
from evaluation_2 import run as eval_run 


def run_pipeline(stage, problem_type):
    """
    stage: str, ["generate", "rephrase", "eval", "all"]
    problem_type: str, ["tsp", "knapsack", "jobshop", "netflow", "binpacking", "portfolio", "transportation", "production", "inventory", "pollution"]
    """

    print(f"\nStarting pipeline for [{problem_type}] | Stage: {stage}\n")
    
    if stage in ["generate", "all"]:
        print("[Stage 1] Generating instances & natural language mapping...")
        generate_instances(problem_type)
        print("Instance generation finished.\n")

    if stage in ["rephrase", "all"]:
        print("[Stage 2] Rephrasing problems with LLM...")
        rephrase_run(problem_type)
        print("Rephrase stage finished.\n")

    if stage in ["evaluation", "all"]:
        print("[Stage 3] Generating and validating Gurobi code...")
        eval_run(problem_type)  
        print("Evaluation stage finished.\n")

    print("Pipeline completed successfully!\n")


def main():
    parser = argparse.ArgumentParser(description="Unified LLM-OPT Data Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["generate", "rephrase", "evaluation", "all"],
        help="Which stage to run: generate | rephrase | eval | all"
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        required=True,
        choices=["tsp", "knapsack", "jobshop", "netflow", "binpacking", "portfolio", "transportation", "production", "inventory", "pollution"],
        help="Problem type to process"
    )

    args = parser.parse_args()
    run_pipeline(args.stage, args.problem_type)


if __name__ == "__main__":
    main()