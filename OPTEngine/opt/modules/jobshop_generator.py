import random
import json
import re
import gurobipy as gp
from gurobipy import GRB
from modules.base_generator import BaseGenerator

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

    def _parse_machine(self, mstr):
        return int(re.search(r'(\d+)', mstr).group(1)) - 1

    def solve_jobshop(self, instance, time_limit=30):
        n_jobs = instance["n_jobs"]
        n_machines = instance["n_machines"]
        jobs = instance["jobs"]

        job_keys = sorted(jobs.keys(), key=lambda x: int(x[1:]))
        jobs_ops = [
            [(self._parse_machine(m), int(t)) for m, t in jobs[j]]
            for j in job_keys
        ]
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
                    S[j, k] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS)
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
                        y = m.addVar(vtype=GRB.BINARY)
                        Y[(j1,k1,j2,k2)] = y
                        m.addConstr(S[j1,k1] + p1 <= S[j2,k2] + bigM * (1 - y))
                        m.addConstr(S[j2,k2] + p2 <= S[j1,k1] + bigM * y)

            for j in range(n_jobs):
                last = len(jobs_ops[j]) - 1
                p_last = jobs_ops[j][last][1]
                m.addConstr(Cmax >= S[j, last] + p_last)

            m.setObjective(Cmax, GRB.MINIMIZE)
            m.optimize()

            if m.Status == GRB.OPTIMAL:
                return round(Cmax.X)
            else:
                return None
        except:
            return None

    def generate_instances(self, output_path="/data1/LLMOptChall/LLMs-OPT/results/jobshop/jsp_instances.jsonl"):
        index = 0
        with open(output_path, "w") as fout:
            for n_jobs in range(self.job_range[0], self.job_range[1] + 1):
                count = 0
                while count < self.samples_per_type:
                    instance = self.generate_instance(index, n_jobs)
                    opt = self.solve_jobshop(instance)
                    if opt is not None:
                        instance["answer"] = opt
                        fout.write(json.dumps(instance) + "\n")
                        count += 1
                        index += 1
        print(f"Generation completed: {index} valid BinPacking instances saved to {output_path}")

    def make_nl_example(self, n_jobs, n_machines, jobs):
        job_lines = []
        for job_name, ops in jobs.items():
            ops_str = " → ".join([f"({machine}, {time})" for machine, time in ops])
            job_lines.append(f"* {job_name} requires the sequence: {ops_str}")
        job_text = "\n".join(job_lines)

        template_1 = f"""Suppose there are {n_jobs} jobs that need to be processed on {n_machines} machines.
        \nEach job consists of a sequence of operations represented as pairs (Machine, Processing time),
        \nwhere each pair specifies the machine on which the operation must run and the amount of time it requires.
        \nThe order of pairs indicates the required sequence in which the operations must be performed.
        \n\nJob details:\n{job_text}\n
        \nEach operation must run continuously once it starts and cannot be interrupted,
        \nand each machine can only process one operation at a time.\nThe objective is to determine the processing order of all operations on the machines
        \nso that the makespan (i.e., the total completion time of all jobs) is minimized."""

        template = f"""Consider {n_jobs} jobs to be processed on {n_machines} machines.
        \nEach job is a sequence of operations given as (Machine, Processing time) pairs, in the order they must be executed.
        \nJob details:\n{job_text}\n
        \nEach operation must run continuously, and each machine can process only one operation at a time.
        \nThe goal is to schedule all operations on the machines to minimize the makespan (the total completion time)."""

        return template
        


    def map_to_nl(self, input_path="/data1/LLMOptChall/LLMs-OPT/results/jobshop/jsp_instances.jsonl", output_path="/data1/LLMOptChall/LLMs-OPT/results/jobshop/jsp_nl.jsonl"):
        total = 0
        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            for line in fin:
                data = json.loads(line)
                nl = self.make_nl_example(data["n_jobs"], data["n_machines"], data["jobs"])
                out = {
                    "index": data["index"],
                    "problem_type": "JobShop",
                    "problem_size": data["n_machines"],
                    "nl_problem": nl,
                    "answer": data["answer"]
                }
                fout.write(json.dumps(out) + "\n")
                total += 1
        print(f"Natural language problem generation completed: {total} instances saved to {output_path}")