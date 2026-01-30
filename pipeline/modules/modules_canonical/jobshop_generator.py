import random
import json
import re
import os
import gurobipy as gp
from gurobipy import GRB
from OPTEngine.modules.base_generator import BaseGenerator

class JobShopGenerator(BaseGenerator):
    def __init__(self, job_range=(3, 10), time_range=(1, 10), samples_per_type=10, seed=42):
        super().__init__(samples_per_type=samples_per_type, seed=seed)
        self.job_range = job_range
        self.time_range = time_range
        random.seed(seed)

    def generate_instance(self, index, n_jobs):
        n_machines = n_jobs
        jobs = {}
        for j in range(1, n_jobs + 1):
            machines_order = random.sample(range(1, n_machines + 1), n_machines)
            operations = [(f"M{m}", random.randint(*self.time_range)) for m in machines_order]
            jobs[f"J{j}"] = operations
        return {
            "index": index,
            "n_jobs": n_jobs,
            "n_machines": n_machines,
            "jobs": jobs
        }

    def solve_jobshop(self, instance, time_limit=30):
        """
        Return:
          (makespan, solution_dict)
        where solution_dict contains start times for each (job, op).
        """
        n_jobs = instance["n_jobs"]
        n_machines = instance["n_machines"]
        jobs = instance["jobs"]

        job_keys = sorted(jobs.keys(), key=lambda x: int(x[1:]))

        jobs_ops = []
        for j_name in job_keys:
            ops = []
            for mstr, t in jobs[j_name]:
                machine = int(re.search(r"(\d+)", mstr).group(1)) - 1  # 0-based
                ops.append((machine, int(t)))
            jobs_ops.append(ops)

        bigM = sum(t for ops in jobs_ops for (_, t) in ops)

        try:
            m = gp.Model("JSP")
            m.Params.OutputFlag = 0
            m.Params.TimeLimit = time_limit

            S = {}
            Y = {}
            Cmax = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C_max")
            mach_to_ops = {i: [] for i in range(n_machines)}

            for j in range(n_jobs):
                for k, (mach, p) in enumerate(jobs_ops[j]):
                    S[j, k] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"S_{j}_{k}")
                    mach_to_ops[mach].append((j, k))

            for j in range(n_jobs):
                for k in range(len(jobs_ops[j]) - 1):
                    p = jobs_ops[j][k][1]
                    m.addConstr(S[j, k+1] >= S[j, k] + p)

            for mach in range(n_machines):
                ops = mach_to_ops[mach]
                for i in range(len(ops)):
                    j1, k1 = ops[i]
                    p1 = jobs_ops[j1][k1][1]
                    for j in range(i + 1, len(ops)):
                        j2, k2 = ops[j]
                        p2 = jobs_ops[j2][k2][1]
                        y = m.addVar(vtype=GRB.BINARY, name=f"Y_{j1}_{k1}_{j2}_{k2}")
                        Y[(j1, k1, j2, k2)] = y
                        m.addConstr(S[j1, k1] + p1 <= S[j2, k2] + bigM * (1 - y))
                        m.addConstr(S[j2, k2] + p2 <= S[j1, k1] + bigM * y)

            for j in range(n_jobs):
                last = len(jobs_ops[j]) - 1
                p_last = jobs_ops[j][last][1]
                m.addConstr(Cmax >= S[j, last] + p_last)

            m.setObjective(Cmax, GRB.MINIMIZE)
            m.optimize()

            if m.Status == GRB.OPTIMAL:
                makespan = float(Cmax.X)

                start_times = {}
                for j in range(n_jobs):
                    for k in range(len(jobs_ops[j])):
                        start_times[f"{j},{k}"] = float(S[j, k].X)

                solution = {
                    "S": start_times,
                    "Cmax": makespan
                }

                return round(makespan), solution
            else:
                return None
        except:
            return None

    def generate_instances(self, output_path="/data1/LLMOptChall/LLMs-OPT/poor_grounding_hard/jobshop/jobshop_instances_original.jsonl"):
        index = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fout:
            for n_jobs in range(self.job_range[0], self.job_range[1] + 1):
                count = 0
                while count < self.samples_per_type:
                    instance = self.generate_instance(index, n_jobs)
                    res = self.solve_jobshop(instance)
                    if res is not None:
                        opt, sol = res
                        instance["answer"] = opt
                        instance["solution"] = sol  # ✅ 新增
                        fout.write(json.dumps(instance) + "\n")
                        count += 1
                        index += 1
        print(f"Generation completed: {index} valid jobshop instances saved to {output_path}")

    def make_nl_example(self, n_jobs, n_machines, jobs):
        job_lines = []
        for job_name in sorted(jobs.keys(), key=lambda x: int(x[1:])):
            ops = jobs[job_name]
            ops_str = " → ".join([f"({machine}, {time})" for machine, time in ops])
            job_lines.append(f"* {job_name} requires the sequence: {ops_str}")
        job_text = "\n".join(job_lines)

        template = f"""A manufacturing plant needs to schedule {n_jobs} production orders across {n_machines} machines.
Each order consists of a sequence of processing steps, represented as (Machine, Processing time) pairs, in the order they must be processed.

Job details:
{job_text}

Each machine can handle only one task at a time, and once a task begins, it must run without interruption.
The goal is to determine a processing schedule for all orders so that the overall completion time (the makespan) is minimized."""
        return template

    def map_to_nl(self, input_path="/data1/LLMOptChall/LLMs-OPT/poor_grounding_hard/jobshop/jobshop_instances_original.jsonl",
                  output_path="/data1/LLMOptChall/LLMs-OPT/poor_grounding_hard/jobshop/jobshop_nl_original.jsonl"):
        total = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            for line in fin:
                data = json.loads(line)
                nl = self.make_nl_example(data["n_jobs"], data["n_machines"], data["jobs"])
                out = {
                    "index": data["index"],
                    "problem_type": "JobShop",
                    "problem_size": data["n_machines"],
                    "nl_problem": nl,
                    "answer": data["answer"],
                    "solution": data.get("solution")  
                }
                fout.write(json.dumps(out) + "\n")
                total += 1
        print(f"Natural language problem generation completed: {total} instances saved to {output_path}")