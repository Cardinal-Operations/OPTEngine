import random
import json
import gurobipy as gp
from gurobipy import GRB
from OPTEngine.modules.base_generator import BaseGenerator

class PollutionGenerator(BaseGenerator):
    def __init__(self,
                 T_range=(3, 11),
                 K_range=(2, 6),
                 w_range=(0.5, 3.0), 
                 p_range=(50.0, 300.0), 
                 s_range=(0.10, 0.90),
                 P_range=(0.20, 0.70),
                 cost_range=(10.0, 200.0), 
                 samples_per_size=10, 
                 seed=42):
        self.T_range = T_range
        self.K_range = K_range
        self.w_range = w_range
        self.p_range = p_range
        self.s_range = s_range
        self.P_range = P_range
        self.cost_range = cost_range
        self.samples_per_size = samples_per_size
        random.seed(seed)


    def generate_instance(self, index, T=None, K=None):
        if T is None:
            T = random.randint(self.T_range[0], self.T_range[1] - 1)
        if K is None:
            K = random.randint(self.K_range[0], self.K_range[1] - 1)

        w = [round(random.uniform(*self.w_range), 3) for _ in range(T)]
        p = [round(random.uniform(*self.p_range), 3) for _ in range(T)]
        s = [round(random.uniform(*self.s_range), 3) for _ in range(K)]
        c = [[round(random.uniform(*self.cost_range), 3) for _ in range(K)] for _ in range(T)]

        raw_P = random.uniform(*self.P_range)
        s_max = max(s) if K > 0 else 0.0
        P = max(0.0, min(raw_P, max(0.0, s_max - 1e-4)))
        P = round(P, 3)

        return {
            "index": index,
            "T": T,
            "K": K,
            "w": w,
            "p": p,
            "s": s,
            "cost": c,
            "P": P
        }

    def solve_tsp_control_lp(self, inst):
        T = inst["T"]
        K = inst["K"]
        w = inst["w"]
        p = inst["p"]
        s = inst["s"]
        c = inst["cost"]
        P = inst["P"]

        try:
            m = gp.Model("tsp_emission_control")
            m.Params.OutputFlag = 0

            x = m.addVars(T, K, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

            m.setObjective(
                gp.quicksum(c[i][j] * x[i, j] for i in range(T) for j in range(K)),
                GRB.MINIMIZE
            )

            for i in range(T):
                m.addConstr(gp.quicksum(x[i, j] for j in range(K)) <= p[i], name=f"balance_{i}")
            rhs = sum(w[i] * p[i] for i in range(T)) * P
            m.addConstr(
                gp.quicksum(w[i] * s[j] * x[i, j] for i in range(T) for j in range(K)) >= rhs,
                name="reduction"
            )

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                return round(float(m.ObjVal), 3)
            else:
                return None
        except Exception:
            return None

    def generate_instances(self, output_path="/data1/LLMOptChall/LLMs-OPT/results/pollution/pollution_instances.jsonl"):
        index = 0
        with open(output_path, "w", encoding="utf-8") as fout:
            for T in range(self.T_range[0], self.T_range[1]):
                for K in range(self.K_range[0], self.K_range[1]):
                    count = 0
                    while count < self.samples_per_size:
                        inst = self.generate_instance(index, T, K)
                        opt_cost = self.solve_tsp_control_lp(inst)
                        if opt_cost is not None:
                            inst["optimal_cost"] = opt_cost
                            fout.write(json.dumps(inst, ensure_ascii=False) + "\n")
                            count += 1
                            index += 1
        print(f"Generating Complete: {index} valid Pollution instances are saved at {output_path}")

    def make_template(self, input_path, output_path):
        
        self.input_path = input_path
        self.output_path = output_path
        self.template = (
            "A region seeks to design an air-pollution control plan to reduce total suspended particulate (TSP) emissions from several industrial point sources."
            "Initially, no control measures have been applied.\n"
            "The characteristics of each emission source are as follows:\n"
            "{source_lines}\n\n"
            "To mitigate emissions, several control methods are available, each characterized by a specific removal efficiency:\n"
            "{method_lines}\n\n"
            "Applying a control method to a source incurs an additional cost per unit of production."
            "The cost structure for all source–method combinations is summarized below"
            "{cost_lines}\n\n"
            "Please choose how to apply control methods to each source (sources may adopt multiple methods simultaneously), "
            "and note that a source may also remain partially uncontrolled if necessary.\n"
            "The goal is to ensure that the total TSP emissions are reduced by at least proportion P = {P} of E0, while minimizing the total cost.\n"
            "\n"
        )


    def _src_label(self, i):
        return f"Station {i+1}"

    def _mtd_label(self, j):
        return f"Method{j+1}"

    def _fmt_sources(self, w, p):
        return "\n".join([
            f"* For {self._src_label(i)}, emission factor is {w[i]}, production is {p[i]})."
            for i in range(len(w))
        ])

    def _fmt_methods(self, s):
        return "\n".join([f"* By using {self._mtd_label(j)}, pollution will be reduced by {1 - s[j]:.0%}" for j in range(len(s))])

    def _fmt_costs_by_method(self, cost):
        if not cost:
            return ""
        T, K = len(cost), len(cost[0])
        for row in cost:
            if len(row) != K:
                raise ValueError("All rows in 'cost' must have the same length K.")

        header = ["Site \\ Method"] + [self._mtd_label(j) for j in range(K)]
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("|-" + "-|" * K)

        for i in range(T):
            row_cells = [self._src_label(i)] + [str(cost[i][j]) for j in range(K)]
            lines.append("| " + " | ".join(row_cells) + " |")

        return "\n".join(lines)

    def make_nl_example(self, data):
        T = data["T"]
        K = data["K"]
        w = data["w"]
        p = data["p"]
        s = data["s"]
        cost = data["cost"]
        P = data["P"]

        if not (len(w) == len(p) == T):
            raise ValueError("Lengths of w and p must both equal T.")
        if len(cost) != T or any(len(row) != K for row in cost):
            raise ValueError("Cost matrix must be T×K.")
        if len(s) != K:
            raise ValueError("Length of s must equal K.")

        source_lines = self._fmt_sources(w, p)
        method_lines = self._fmt_methods(s)
        cost_lines = self._fmt_costs_by_method(cost)

        nl = self.template.format(
            source_lines=source_lines,
            method_lines=method_lines,
            cost_lines=cost_lines,
            P=P
        )
        return nl, T, K

    def map_to_nl(self, input_path = "/data1/LLMOptChall/LLMs-OPT/results/pollution/pollution_instances.jsonl", output_path = "/data1/LLMOptChall/LLMs-OPT/results/pollution/pollution_nl.jsonl"):
        total = 0
        self.make_template(input_path, output_path)

        with open(self.input_path, "r", encoding="utf-8") as fin, open(self.output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                data = json.loads(line)

                nl, T, K = self.make_nl_example(data)

                out = {
                    "index": data["index"],
                    "problem_type": "Pollution",
                    "problem_size": [T, K],
                    "nl_problem": nl
                }
                if "optimal_cost" in data:
                    out["answer"] = data["optimal_cost"]

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                total += 1

        print(f"The natural language problem is generated, and a total of {total} instances are saved to {self.output_path}")