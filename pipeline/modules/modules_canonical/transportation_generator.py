import random
import json
import gurobipy as gp
from gurobipy import GRB
from OPTEngine.modules.base_generator import BaseGenerator

class TransportationGenerator(BaseGenerator):
    def __init__(self,
                 n_range=(3, 11),
                 m_range=(3, 11),
                 supply_range=(50, 200),
                 demand_share=(0.60, 0.95),
                 cost_range=(1, 20),
                 samples_per_size=10,
                 seed=42):

        self.n_range = n_range
        self.m_range = m_range
        self.supply_range = supply_range
        self.demand_share = demand_share
        self.cost_range = cost_range
        self.samples_per_size = samples_per_size
        random.seed(seed)

    def generate_instance(self, index, n=None, m=None):
        if n is None:
            n = random.randint(self.n_range[0], self.n_range[1] - 1)
        if m is None:
            m = random.randint(self.m_range[0], self.m_range[1] - 1)

        supply = [random.randint(*self.supply_range) for _ in range(n)]
        total_supply = sum(supply)

        rho = random.uniform(*self.demand_share)  # ≤ 1
        total_demand = max(1, int(total_supply * rho))
        raw = [random.random() + 1e-9 for _ in range(m)]
        s = sum(raw)
        demand = [max(1, int(total_demand * x / s)) for x in raw]

        gap = total_demand - sum(demand)

        k = 0
        while gap != 0 and m > 0:
            j = k % m
            if gap > 0:
                demand[j] += 1
                gap -= 1
            else:
                if demand[j] > 1:
                    demand[j] -= 1
                    gap += 1
            k += 1

        cost = [[random.randint(*self.cost_range) for _ in range(m)] for _ in range(n)]

        return {
            "index": index,
            "n": n,
            "m": m,
            "supply": supply,
            "demand": demand,
            "cost": cost
        }

    def solve_transportation(self, inst):
        n = inst["n"]
        m = inst["m"]
        e = inst["supply"]
        d = inst["demand"]
        c = inst["cost"]

        try:
            mdl = gp.Model("transportation")
            mdl.Params.OutputFlag = 0

            x = mdl.addVars(n, m, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
            mdl.setObjective(gp.quicksum(c[i][j] * x[i, j] for i in range(n) for j in range(m)),
                             GRB.MINIMIZE)
            for i in range(n):
                mdl.addConstr(gp.quicksum(x[i, j] for j in range(m)) <= e[i], name=f"supply_{i}")

            for j in range(m):
                mdl.addConstr(gp.quicksum(x[i, j] for i in range(n)) == d[j], name=f"demand_{j}")

            mdl.optimize()

            if mdl.Status == GRB.OPTIMAL:
                return float(mdl.ObjVal)
            else:
                return None
        except Exception:
            return None

    def generate_instances(self, output_path="/data1/LLMOptChall/LLMs-OPT/results/transportation/transportation_instances.jsonl"):
        index = 0
        with open(output_path, "w", encoding="utf-8") as fout:
            for n in range(self.n_range[0], self.n_range[1]):
                for m in range(self.m_range[0], self.m_range[1]):
                    count = 0
                    while count < self.samples_per_size:
                        instance = self.generate_instance(index, n, m)
                        opt_cost = self.solve_transportation(instance)
                        if opt_cost is not None:
                            instance["optimal_cost"] = opt_cost
                            fout.write(json.dumps(instance, ensure_ascii=False) + "\n")
                            count += 1
                            index += 1
        print(f"Generating Complete: {index} valid Transportation instances are saved at {output_path}")

    def make_template(self, input_path, output_path):
        
        self.input_path = input_path
        self.output_path = output_path
        self.template = (
            "Consider a transportation problem that aims to minimize the total shipping cost from production sites A to sales destinations B.\n\n"
            "The available supply at each production site (set A) is given as follows: \n"
            "{supply_lines}\n\n"
            "The demand that must be met at each sales destinations (set B) is specified below: \n"
            "{demand_lines}\n\n"
            "The unit shipping cost from each production site to each destination is as follows:\n"
            "{cost_lines}\n\n"
            "Please choose shipment to minimize the total cost."
        )

    def _site_label(self, i):
        return f"A{i+1}"

    def _dest_label(self, j):
        return f"B{j+1}"

    def _fmt_supply(self, supply):
        return "\n".join([f"*{self._site_label(i)}'s supply is {s}." for i, s in enumerate(supply)])

    def _fmt_demand(self, demand):
        return "\n".join([f"*{self._dest_label(j)}'s demand is {d}." for j, d in enumerate(demand)])

    def _fmt_cost(self, cost):

        if not cost:
            return ""
        n, m = len(cost), len(cost[0])
        for row in cost:
            if len(row) != m:
                raise ValueError("All rows in 'cost' must have the same length (m).")
            
        header = ["From / To"] + [self._dest_label(j) for j in range(m)]
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("|-" + "-|" * m)

        for i in range(n):
            row_cells = [self._site_label(i)] + [str(cost[i][j]) for j in range(m)]
            lines.append("| " + " | ".join(row_cells) + " |")

        return "\n".join(lines)

    def make_nl_example(self, data):
        supply = data["supply"]
        demand = data["demand"]
        cost = data["cost"]

        n = len(supply)
        m = len(demand)
        if len(cost) != n or any(len(row) != m for row in cost):
            raise ValueError("Cost matrix shape must be nxm, matching lengths of supply and demand.")

        supply_lines = self._fmt_supply(supply)
        demand_lines = self._fmt_demand(demand)
        cost_lines = self._fmt_cost(cost)

        nl = self.template.format(
            supply_lines=supply_lines,
            demand_lines=demand_lines,
            cost_lines=cost_lines
        )
        return nl, n, m

    def map_to_nl(self, input_path = "/data1/LLMOptChall/LLMs-OPT/results/transportation/transportation_instances.jsonl", output_path="/data1/LLMOptChall/LLMs-OPT/results/transportation/transportation_nl.jsonl"):
        
        total = 0
        self.make_template(input_path, output_path)

        with open(self.input_path, "r", encoding="utf-8") as fin, open(self.output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                data = json.loads(line)

                nl, n, m = self.make_nl_example(data)

                out = {
                    "index": data["index"],
                    "problem_type": "Transportation",
                    "problem_size": [n, m],      # n=|A|, m=|B|
                    "nl_problem": nl
                }
                if "optimal_cost" in data:
                    out["answer"] = data["optimal_cost"]

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                total += 1

        print(f"The natural language problem is generated, and a total of {total} instances are saved to {self.output_path}")