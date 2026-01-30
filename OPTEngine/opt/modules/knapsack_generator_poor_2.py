# modules/knapsack_generator.py
import random
import json
import gurobipy as gp
from gurobipy import GRB
from modules.base_generator import BaseGenerator


class KnapsackGenerator(BaseGenerator):
    def __init__(self, n_items_range=(5, 30), weight_range=(1, 50),
                 value_range=(10, 300), capacity_ratio=0.7,
                 samples_per_type=10, seed=42):

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
            "capacity": capacity
        }

    def solve_knapsack(self, items, capacity):
        try:
            m = gp.Model("knapsack")
            m.Params.OutputFlag = 0

            n = len(items)
            x = m.addVars(n, vtype=GRB.BINARY, name="x")

            # objective
            m.setObjective(
                gp.quicksum(items[i]["value"] * x[i] for i in range(n)),
                GRB.MAXIMIZE
            )

            # capacity constraint
            m.addConstr(
                gp.quicksum(items[i]["weight"] * x[i] for i in range(n)) <= capacity,
                name="capacity_constraint"
            )

            # ⭐ NEW — 二选一（Exactly one of item1 and item2）
            if n >= 2:
                m.addConstr(
                    x[0] + x[1] == 1,
                    name="either_item1_or_item2"
                )

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                return m.ObjVal
            else:
                return None

        except:
            return None

    # ✅ MATCH STRUCTURE — Use the same output directory naming pattern
    def generate_instances(self, output_path="/data1/LLMOptChall/LLMs-OPT/results_try/knapsack/knapsack_instances_poor_1.jsonl"):
        index = 0

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

        print(f"Generation completed: {index} valid knapsack instances saved to {output_path}")

    def make_nl_example(self, items, capacity):

        items_list = "\n".join([
            f"* Item {i+1}: weight {item['weight']}kg, value {item['value']} points"
            for i, item in enumerate(items)
        ])

        # ⭐ NEW — Add rule description for the constraint
        rule_sentence = (
            "\nAdditionally, exactly one of Item 1 and Item 2 must be selected.\n"
        )

        template = (
            "A hiker is preparing for a 3-day outdoor hiking trip. "
            "They must choose a set of items to maximize total value while respecting the backpack weight limit.\n\n"
            "The items available are:\n"
            "{items_list}\n"
            f"{rule_sentence}"
            "The backpack can carry at most {capacity} kg. The hiker must decide which items to take."
        )

        return template.format(items_list=items_list, capacity=capacity)

    # ⭐ NEW — THIS WAS MISSING IN YOUR FIRST VERSION
    # Fully copied structure from the second version
    def map_to_nl(self,
                  input_path="/data1/LLMOptChall/LLMs-OPT/results_try/knapsack/knapsack_instances_poor_1.jsonl",
                  output_path="/data1/LLMOptChall/LLMs-OPT/results_try/knapsack/knapsack_nl_poor_1.jsonl"):

        total = 0

        with open(input_path, "r") as fin, open(output_path, "w") as fout:

            for line in fin:
                data = json.loads(line)
                nl = self.make_nl_example(data["items"], data["capacity"])

                out = {
                    "index": data["index"],
                    "problem_type": "Knapsack",
                    "problem_size": data["n_items"],
                    "nl_problem": nl,
                    "answer": data["answer"]
                }

                fout.write(json.dumps(out) + "\n")
                total += 1

        print(f"Natural language problem generation completed: {total} instances saved to {output_path}")