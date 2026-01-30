# modules/knapsack_generator.py
import random
import json
import os
import gurobipy as gp
from gurobipy import GRB
from modules.base_generator import BaseGenerator


class KnapsackGenerator(BaseGenerator):
    def __init__(
        self,
        n_items_range=(5, 30),
        weight_range=(1, 50),
        value_range=(10, 300),
        capacity_ratio=0.7,
        samples_per_type=10,
        seed=42,
    ):
        super().__init__(samples_per_type=samples_per_type, seed=seed)
        self.n_items_range = n_items_range
        self.weight_range = weight_range
        self.value_range = value_range
        self.capacity_ratio = capacity_ratio
        random.seed(seed)

    def generate_instance(self, index, n_items):
        items = []
        total_weight = 0

        for _ in range(n_items):
            w = random.randint(*self.weight_range)
            v = random.randint(*self.value_range)
            items.append({"weight": w, "value": v})
            total_weight += w

        capacity = int(total_weight * self.capacity_ratio)

        return {
            "index": index,
            "n_items": n_items,
            "items": items,
            "capacity": capacity,
        }

    def solve_knapsack(self, items, capacity):
        """
        Standard 0-1 knapsack consistent with the NL template:
        - maximize total value
        - total weight <= capacity
        - each item is either taken (1) or not taken (0)
        """
        try:
            m = gp.Model("knapsack")
            m.Params.OutputFlag = 0

            n = len(items)
            x = m.addVars(n, vtype=GRB.BINARY, name="x")

            m.setObjective(
                gp.quicksum(items[i]["value"] * x[i] for i in range(n)),
                GRB.MAXIMIZE,
            )

            m.addConstr(
                gp.quicksum(items[i]["weight"] * x[i] for i in range(n)) <= capacity,
                name="capacity_constraint",
            )

            # ✅ REMOVED: must_choose_item1
            # 因为当前 template 并没有说 “必须选择 Item 1”，所以求解器不能强制它。

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                return float(m.ObjVal)
            else:
                return None
        except Exception:
            return None

    def generate_instances(
        self,
        output_path="/data1/LLMOptChall/LLMs-OPT/ppl/results/knapsack/knapsack_instances_5.jsonl",
    ):
        index = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fout:
            for n_items in range(self.n_items_range[0], self.n_items_range[1] + 1):
                count = 0
                while count < self.samples_per_type:
                    instance = self.generate_instance(index, n_items)
                    opt_value = self.solve_knapsack(instance["items"], instance["capacity"])
                    if opt_value is not None:
                        instance["answer"] = opt_value
                        fout.write(json.dumps(instance) + "\n")
                        count += 1
                        index += 1

        print(f"Generation completed: {index} valid Knapsack instances saved to {output_path}")

    def make_nl_example(self, items, capacity):
        items_list = "\n".join(
            [
                f"* Item {i+1}: weight {item['weight']}kg, value {item['value']} points"
                for i, item in enumerate(items)
            ]
        )

        template = (
"Consider a constrained selection problem arising from a capacity-limited packing scenario."
"A finite collection of discrete items is available, where each item is described by two numerical attributes: a weight contribution and a corresponding value contribution, as specified below:"
"{items_list}\n"

"A single container with a fixed carrying capacity of {capacity} kg is available."
"From the given collection, a subset of items must be chosen such that the cumulative weight of the selected items does not exceed the container’s capacity."
"Items are indivisible and may either be fully included or entirely excluded."

"The objective is to determine a selection whose total value contribution is maximized among all selections that satisfy the capacity constraint."
        )

        return template.format(items_list=items_list, capacity=capacity)

    def map_to_nl(
        self,
        input_path="/data1/LLMOptChall/LLMs-OPT/ppl/results/knapsack/knapsack_instances_5.jsonl",
        output_path="/data1/LLMOptChall/LLMs-OPT/ppl/results/knapsack/knapsack_nl_5.jsonl",
    ):
        total = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            for line in fin:
                data = json.loads(line)
                nl = self.make_nl_example(data["items"], data["capacity"])

                out = {
                    "index": data["index"],
                    "problem_type": "Knapsack",
                    "problem_size": data["n_items"],
                    "nl_problem": nl,
                    "answer": data["answer"],
                }

                fout.write(json.dumps(out) + "\n")
                total += 1

        print(f"Natural language problem generation completed: {total} instances saved to {output_path}")