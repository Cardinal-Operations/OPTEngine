# modules/knapsack_generator.py
import random
import json
import gurobipy as gp
from gurobipy import GRB
import os
from modules.base_generator import BaseGenerator


class KnapsackGenerator(BaseGenerator):
    def __init__(self,
                 n_items_range=(5, 30),
                 weight_range=(1, 50),
                 value_range=(10, 300),
                 capacity_ratio=0.7,
                 samples_per_type=10,
                 seed=42):

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

            m.setObjective(
                gp.quicksum(items[i]["value"] * x[i] for i in range(n)),
                GRB.MAXIMIZE
            )

            lhs = gp.quicksum(items[i]["weight"] * x[i] for i in range(n))

            # Rule 2: If Item 3 is selected, capacity reduced by 2
            if n >= 3:
                m.addConstr(
                    lhs <= capacity - 10 * x[2],
                    name="capacity_with_item3_penalty"
                )
            else:
                m.addConstr(
                    lhs <= capacity,
                    name="capacity_constraint"
                )

            # Rule 1: Exactly one of Item 1 and Item 2
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

        except Exception:
            return None

    def generate_instances(self,
                           output_path="/data1/LLMOptChall/LLMs-OPT/poor_grounding/results/knapsack/knapsack_instances_hard.jsonl"):

        index = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fout:
            for n_items in range(self.n_items_range[0], self.n_items_range[1] + 1):

                count = 0
                while count < self.samples_per_type:

                    instance = self.generate_instance(index, n_items)
                    opt_value = self.solve_knapsack(
                        instance["items"],
                        instance["capacity"]
                    )

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

        rule_sentence = (
            "\nAdditionally, exactly one of Item 1 and Item 2 must be selected.\n"
            "\nIf Item 3 is selected, the effective backpack capacity is reduced by 10 kg.\n"
        )

        template = (
            "A hiker is preparing for a 3-day outdoor hiking trip. "
            "They must choose a set of items to maximize total value while respecting the backpack weight limit.\n\n"
            "The items available are:\n"
            "{items_list}\n"
            f"{rule_sentence}"
            "The backpack can carry at most {capacity} kg. "
            "The hiker must decide which items to take."
        )

        return template.format(
            items_list=items_list,
            capacity=capacity
        )

    def map_to_nl(self,
                  input_path="/data1/LLMOptChall/LLMs-OPT/poor_grounding/results/knapsack/knapsack_instances_hard.jsonl",
                  output_path="/data1/LLMOptChall/LLMs-OPT/poor_grounding/results/knapsack/knapsack_nl_hard.jsonl"):

        total = 0

        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            for line in fin:
                data = json.loads(line)

                nl = self.make_nl_example(
                    data["items"],
                    data["capacity"]
                )

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