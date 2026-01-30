# modules/knapsack_generator.py
import random
import json
import os
import gurobipy as gp
from gurobipy import GRB
from OPTEngine.modules.base_generator import BaseGenerator


class KnapsackGenerator(BaseGenerator):
    def __init__(
        self,
        n_items_range=(5, 30),
        weight_range=(1, 50),
        value_range=(10, 300),
        capacity_ratio=0.7,
        samples_per_type=10,
        seed=42,
        bonus_value=50,  # ✅ NEW: fixed bonus added if at least one item is selected
    ):
        super().__init__(samples_per_type=samples_per_type, seed=seed)
        self.n_items_range = n_items_range
        self.weight_range = weight_range
        self.value_range = value_range
        self.capacity_ratio = capacity_ratio
        self.bonus_value = bonus_value
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
            "bonus_value": self.bonus_value,  # ✅ NEW
        }

    def solve_knapsack(self, items, capacity, bonus_value):
        """
        0-1 knapsack + fixed bonus:
        - maximize sum(value_i * x_i) + bonus_value * y
        - y = 1 if at least one item is selected; otherwise y = 0
        - total weight <= capacity
        - x_i in {0,1}, y in {0,1}
        """
        try:
            m = gp.Model("knapsack")
            m.Params.OutputFlag = 0

            n = len(items)
            x = m.addVars(n, vtype=GRB.BINARY, name="x")
            y = m.addVar(vtype=GRB.BINARY, name="y")  # ✅ NEW: indicator for selecting any item

            # Capacity constraint
            m.addConstr(
                gp.quicksum(items[i]["weight"] * x[i] for i in range(n)) <= capacity,
                name="capacity_constraint",
            )

            # ✅ NEW: y = 1 if any x[i] = 1
            # Minimal correct formulation: y >= x[i] for all i
            for i in range(n):
                m.addConstr(y >= x[i], name=f"activate_bonus_{i}")

            # Objective: base value + fixed bonus if any item selected
            m.setObjective(
                gp.quicksum(items[i]["value"] * x[i] for i in range(n)) + float(bonus_value) * y,
                GRB.MAXIMIZE,
            )

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                return float(m.ObjVal)
            else:
                return None
        except Exception:
            return None

    def generate_instances(
        self,
        output_path="/data1/LLMOptChall/LLMs-OPT/results_new/knapsack/knapsack_instances_obj.jsonl",
    ):
        index = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fout:
            for n_items in range(self.n_items_range[0], self.n_items_range[1] + 1):
                count = 0
                while count < self.samples_per_type:
                    instance = self.generate_instance(index, n_items)
                    opt_value = self.solve_knapsack(
                        instance["items"], instance["capacity"], instance["bonus_value"]
                    )
                    if opt_value is not None:
                        instance["answer"] = opt_value
                        fout.write(json.dumps(instance) + "\n")
                        count += 1
                        index += 1

        print(f"Generation completed: {index} valid Knapsack instances saved to {output_path}")

    def make_nl_example(self, items, capacity, bonus_value):
        items_list = "\n".join(
            [
                f"* Item {i+1}: weight {item['weight']}kg, value {item['value']} points"
                for i, item in enumerate(items)
            ]
        )

        template = (
            "A hiker is preparing for a 3-day outdoor hiking trip. "
            "They need to select the most valuable combination of equipment and supplies "
            "from many available options within the limited backpack capacity.\n\n"
            "The items include:\n{items_list}\n\n"
            "The backpack has a maximum weight capacity of {capacity} kg. "
            "Each item must be either taken in its entirety or left behind.\n\n"
            "If the hiker selects at least one item, "
            "they receive an additional fixed bonus of {bonus_value} points added to the total value "
            "(this bonus is added only once, no matter how many items are selected).\n\n"
            "The goal is to choose a subset of items to maximize the total value"
            "without exceeding the weight limit."
        )

        return template.format(items_list=items_list, capacity=capacity, bonus_value=bonus_value)

    def map_to_nl(
        self,
        input_path="/data1/LLMOptChall/LLMs-OPT/results_new/knapsack/knapsack_instances_obj.jsonl",
        output_path="/data1/LLMOptChall/LLMs-OPT/results_new/knapsack/knapsack_nl_obj.jsonl",
    ):
        total = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            for line in fin:
                data = json.loads(line)
                nl = self.make_nl_example(data["items"], data["capacity"], data["bonus_value"])

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