import random
import json
import os
import gurobipy as gp
from gurobipy import GRB
from OPTEngine.modules.base_generator import BaseGenerator

class ProductionGenerator(BaseGenerator):
    def __init__(self,
                 I_range=(3, 11),         
                 J_range=(2, 6),
                 profit_range=(5.0, 20.0),
                 time_range=(0.2, 2.0),
                 ref_x_total_range=(50.0, 200.0),
                 capacity_relax=(1.00, 1.50),
                 samples_per_size=5,
                 seed=0):

        self.I_range = I_range
        self.J_range = J_range
        self.profit_range = profit_range
        self.time_range = time_range
        self.ref_x_total_range = ref_x_total_range
        self.capacity_relax = capacity_relax
        self.samples_per_size = samples_per_size
        random.seed(seed)

    def _rand_nonneg_vector_with_sum(self, n, total):
        z = [random.random() + 1e-9 for _ in range(n)]
        s = sum(z)
        vec = [round(total * zi / s, 3) for zi in z]
        diff = round(total - sum(vec), 3)
        if abs(diff) > 1e-3:
            vec[0] = round(vec[0] + diff, 3)
        return vec

    def generate_instance(self, index, I=None, J=None):
        if I is None:
            I = random.randint(self.I_range[0], self.I_range[1] - 1)
        if J is None:
            J = random.randint(self.J_range[0], self.J_range[1] - 1)

        p = [round(random.uniform(*self.profit_range), 3) for _ in range(I)]
        time_mat = [[round(random.uniform(*self.time_range), 3) for _ in range(J)] for _ in range(I)]

        x_total = round(random.uniform(*self.ref_x_total_range), 3)
        x_ref = self._rand_nonneg_vector_with_sum(I, x_total)

        T = []
        for j in range(J):
            load_j = sum(time_mat[i][j] * x_ref[i] for i in range(I))
            beta = round(random.uniform(*self.capacity_relax), 3)
            T.append(round(load_j * beta, 3))

        instance = {
            "index": index,
            "I": I,
            "J": J,
            "p": p,
            "time": time_mat,
            "x_ref": x_ref,
            "Tmax": T
        }
        return instance

    def solve_productmix_lp(self, inst):
        I = inst["I"]
        J = inst["J"]
        p = inst["p"]
        t = inst["time"]
        T = inst["Tmax"]

        try:
            m = gp.Model("product_mix_lp")
            m.Params.OutputFlag = 0

            x = m.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
            m.setObjective(gp.quicksum(p[i] * x[i] for i in range(I)), GRB.MAXIMIZE)
            for j in range(J):
                m.addConstr(gp.quicksum(t[i][j] * x[i] for i in range(I)) <= T[j], name=f"cap_op_{j}")

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                return round(float(m.ObjVal), 3)
            else:
                return None
        except Exception:
            return None

    def generate_instances(self, output_path="/data1/LLMOptChall/LLMs-OPT/results/production_2/production_2_instances.jsonl"):
        index = 0
        total = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fout:
            for I in range(self.I_range[0], self.I_range[1]):
                for J in range(self.J_range[0], self.J_range[1]):
                    count = 0
                    while count < self.samples_per_size:
                        inst = self.generate_instance(index, I, J)
                        opt_profit = self.solve_productmix_lp(inst)
                        if opt_profit is not None:
                            inst["optimal_profit"] = opt_profit
                            fout.write(json.dumps(inst, ensure_ascii=False) + "\n")
                            count += 1
                            index += 1
                            total += 1

        print(f"Generating Complete: {index} valid Production instances are saved at {output_path}")


    def make_template(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.template = (
            "A factory intends to produce {I} types of products, each of which requires {J} processing operations to complete.\n"
            "The profit earned per {unit_label} of each product is given as follows:\n"
            "{profit_lines}\n\n"
            "Processing time required per for each operation is as followed:\n"
            "{time_lines}\n\n"
            "{op_cap_lines}\n\n"
            "On the premise of guaranteeing for each operation {op_range}, total processing time must not exceed its available time. \n"
            "Please schedule the production plan to maximize total profit.\n"
        )
    def _prod_label(self, i, product_names=None):
        if product_names and 0 <= i < len(product_names):
            return product_names[i]
        return chr(ord('A') + i)

    def _op_label(self, j, operation_names=None):
        if operation_names and 0 <= j < len(operation_names):
            return operation_names[j]
        return f"Operation {j+1}"

    def _fmt_profit_lines(self, p, product_names, unit_label):
        lines = []
        for i, pi in enumerate(p):
            lines.append(f"* For Product {self._prod_label(i, product_names)}, profit is {pi} per {unit_label}")
        return "\n".join(lines)

    def _fmt_time_lines(self, time_mat, product_names, operation_names, unit_label):
        I = len(time_mat)
        if I == 0:
            return ""
        J = len(time_mat[0])
        op_keys = [self._op_label(j, operation_names) for j in range(J)]

        lines = []
        for i in range(I):
            row = time_mat[i]
            if len(row) != J:
                raise ValueError("All rows in 'time' must have length J.")
            pairs = [f"{op_keys[j]} needs {row[j]} ({unit_label} time)" for j in range(J)]
            lines.append(f"*For Product {self._prod_label(i, product_names)}, " + ", ".join(pairs))
        return "\n".join(lines)

    def _fmt_op_cap_lines(self, Tmax, operation_names):
        lines = []
        for j, Tj in enumerate(Tmax):
            lines.append(f"* {self._op_label(j, operation_names)}'s total production time must not exceed {Tj}")
        return "\n".join(lines)

    def make_nl_example(self, data):
        I = data["I"]
        J = data["J"]
        p = data["p"]
        time_mat = data["time"]
        Tmax = data["Tmax"]
        product_names = data.get("product_names")
        operation_names = data.get("operation_names")
        unit_label = data.get("unit_label", "kg")

        if not (len(p) == I and len(time_mat) == I and len(Tmax) == J):
            raise ValueError("Lengths of p/time/Tmax must match I/J.")
        for row in time_mat:
            if len(row) != J:
                raise ValueError("Each row of 'time' must have length J.")

        profit_lines = self._fmt_profit_lines(p, product_names, unit_label)
        time_lines = self._fmt_time_lines(time_mat, product_names, operation_names, unit_label)
        op_cap_lines = self._fmt_op_cap_lines(Tmax, operation_names)
        op_range = ", ".join([self._op_label(j, operation_names) for j in range(J)])

        nl = self.template.format(
            I=I,
            J=J,
            unit_label=unit_label,
            profit_lines=profit_lines,
            time_lines=time_lines,
            op_cap_lines=op_cap_lines,
            op_range=op_range
        )
        return nl

    def map_to_nl(self, input_path="/data1/LLMOptChall/LLMs-OPT/results/production_2/production_2_instances.jsonl", 
                  output_path = "/data1/LLMOptChall/LLMs-OPT/results/production_2/production_2_nl.jsonl"):
        total = 0
        self.make_template(input_path, output_path)

        with open(self.input_path, "r", encoding="utf-8") as fin, open(self.output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                nl = self.make_nl_example(data)

                out = {
                    "index": data["index"],
                    "problem_type": "Production",
                    "problem_size": [data["I"], data["J"]],
                    "nl_problem": nl,
                }
                if "optimal_profit" in data:
                    out["answer"] = data["optimal_profit"]

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                total += 1

        print(f"The natural language problem is generated, and a total of {total} instances are saved to {self.output_path}")