import random
import json
import re
import os
import gurobipy as gp
from gurobipy import GRB
from OPTEngine.modules.base_generator import BaseGenerator

class JobShopGenerator(BaseGenerator):
    def __init__(
        self,
        job_range=(3, 10),
        time_range=(1, 10),
        samples_per_type=10,
        seed=42,
        startup_time=10,
    ):
        super().__init__(samples_per_type=samples_per_type, seed=seed)
        self.job_range = job_range
        self.time_range = time_range
        self.startup_time = startup_time
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
            "jobs": jobs,
            "startup_time": self.startup_time,
        }

    def solve_jobshop(self, instance, time_limit=30):
        n_jobs = instance["n_jobs"]
        n_machines = instance["n_machines"]
        jobs = instance["jobs"]
        startup_time = instance["startup_time"]

        job_keys = sorted(jobs.keys(), key=lambda x: int(x[1:]))
        jobs_ops = []
        for j_name in job_keys:
            ops = []
            for mstr, t in jobs[j_name]:
                match = re.search(r"(\d+)", mstr)
                machine = int(match.group(1)) - 1 if match else 0
                ops.append((machine, int(t)))
            jobs_ops.append(ops)

        bigM = sum(t for ops in jobs_ops for (_, t) in ops) + startup_time

        try:
            with gp.Model("JSP") as m:
                m.Params.OutputFlag = 0
                m.Params.TimeLimit = time_limit

                S = {}
                Y = {}

                Cmax = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C_max")
                Obj = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Total_Time")

                mach_to_ops = {i: [] for i in range(n_machines)}

                for j in range(n_jobs):
                    for k, (mach, _) in enumerate(jobs_ops[j]):
                        S[j, k] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"S_{j}_{k}")
                        mach_to_ops[mach].append((j, k))

                for j in range(n_jobs):
                    for k in range(len(jobs_ops[j]) - 1):
                        p = jobs_ops[j][k][1]
                        m.addConstr(S[j, k + 1] >= S[j, k] + p)

                for mach in range(n_machines):
                    ops = mach_to_ops[mach]
                    for i in range(len(ops)):
                        for jj in range(i + 1, len(ops)):
                            j1, k1 = ops[i]
                            j2, k2 = ops[jj]
                            p1 = jobs_ops[j1][k1][1]
                            p2 = jobs_ops[j2][k2][1]
                            
                            y = m.addVar(vtype=GRB.BINARY)
                            Y[(j1, k1, j2, k2)] = y
                            m.addConstr(S[j1, k1] + p1 <= S[j2, k2] + bigM * (1 - y))
                            m.addConstr(S[j2, k2] + p2 <= S[j1, k1] + bigM * y)

                for j in range(n_jobs):
                    last = len(jobs_ops[j]) - 1
                    p_last = jobs_ops[j][last][1]
                    m.addConstr(Cmax >= S[j, last] + p_last)

                m.addConstr(Obj == Cmax + startup_time)
                m.setObjective(Obj, GRB.MINIMIZE)

                m.optimize()

                if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
                    total_time = float(Obj.X)
                    start_times = {
                        f"{j},{k}": float(S[j, k].X)
                        for j in range(n_jobs)
                        for k in range(len(jobs_ops[j]))
                    }
                    solution = {
                        "S": start_times,
                        "Cmax": float(Cmax.X),
                        "startup_time": startup_time,
                        "objective": total_time,
                    }
                    return round(total_time), solution
                else:
                    return None
        except Exception as e:
            print(f"Error solving JSP: {e}")
            return None

    def generate_instances(
        self,
        output_path="/data1/LLMOptChall/LLMs-OPT/results_new/jobshop/jobshop_instances_obj.jsonl",
    ):
        index = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

        with open(output_path, "w") as fout:
            for n_jobs in range(self.job_range[0], self.job_range[1] + 1):
                count = 0
                while count < self.samples_per_type:
                    instance = self.generate_instance(index, n_jobs)
                    res = self.solve_jobshop(instance)
                    if res is not None:
                        opt, sol = res
                        instance["answer"] = opt
                        instance["solution"] = sol
                        fout.write(json.dumps(instance) + "\n")
                        count += 1
                        index += 1

        print(f"Generation completed: {index} instances saved to {output_path}")

    def make_nl_example(self, n_jobs, n_machines, jobs, startup_time):
        job_lines = []
        for job_name in sorted(jobs.keys(), key=lambda x: int(x[1:])):
            ops = jobs[job_name]
            ops_str = " → ".join([f"({m}, {t})" for m, t in ops])
            job_lines.append(f"* {job_name} requires the sequence: {ops_str}")

        job_text = "\n".join(job_lines)

        template = f"""A manufacturing plant needs to schedule {n_jobs} production orders across {n_machines} machines.
Each order consists of a sequence of processing steps, represented as (Machine, Processing time) pairs, in the order they must be processed.

Job details:
{job_text}

Each machine can handle only one task at a time, and once a task begins, it must run without interruption.

Before any production can start, the machines require a fixed initial startup time of {startup_time} time units.
As a result, the total completion time equals the makespan plus this startup time.

The goal is to determine a processing schedule that minimizes this total completion time.
"""
        return template

    def map_to_nl(self, input_path="/data1/LLMOptChall/LLMs-OPT/results_new/jobshop/jobshop_instances_obj.jsonl",
                  output_path="/data1/LLMOptChall/LLMs-OPT/results_new/jobshop/jobshop_nl_obj.jsonl"):
        total = 0
        if not os.path.exists(input_path):
            print(f"Input file {input_path} not found.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        
        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            for line in fin:
                data = json.loads(line)
                nl = self.make_nl_example(
                    data["n_jobs"], 
                    data["n_machines"], 
                    data["jobs"], 
                    data.get("startup_time", self.startup_time) 
                )
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