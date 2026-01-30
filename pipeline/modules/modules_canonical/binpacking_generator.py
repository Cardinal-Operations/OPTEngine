import random
import json
import os
import gurobipy as gp
from gurobipy import GRB
from OPTEngine.modules.base_generator import BaseGenerator

class BinPackingGenerator(BaseGenerator):
    def __init__(self, 
                 n_items_range=(10, 41),
                 bin_capacity=500,
                 weight_range=(0, 100),
                 samples_per_type=10,
                 seed=42):
        
        super().__init__(samples_per_type=samples_per_type, seed=seed)
        self.n_items_range = n_items_range
        self.bin_capacity = bin_capacity
        self.weight_range = weight_range
        random.seed(seed)

    def generate_instance_1(self, index, n_items):
        items = [random.randint(*self.weight_range) for _ in range(n_items)]
        return {
            "index": index,
            "n_items": n_items,
            "bin_capacity": self.bin_capacity,
            "items": items
        }
    
    def generate_instance_3(self, index, n_items):
        items = []
        for _ in range(n_items):
            if random.random() < 0.5:
                items.append(random.randint(30, 40))
            else:
                items.append(random.randint(60, 70))

        return {
            "index": index,
            "n_items": n_items,
            "bin_capacity": self.bin_capacity,
            "items": items
        }
    def generate_instance_2(self, index, n_items):
        items = []
        for _ in range(n_items):
            r = random.random()
            if r < 0.4:
                items.append(random.randint(25, 35))
            elif r < 0.8:
                items.append(random.randint(40, 55))
            else:
                items.append(random.randint(65, 75))

        return {
            "index": index,
            "n_items": n_items,
            "bin_capacity": self.bin_capacity,
            "items": items
        }
    
    def generate_instance(self, index, n_items):
        items = []
        for _ in range(n_items):
            if random.random() < 0.5:
                items.append(random.randint(31, 34))
            else:
                items.append(random.randint(66, 70))

        return {
            "index": index,
            "n_items": n_items,
            "bin_capacity": 100,
            "items": items
        }

    def solve_binpacking(self, items):
        n = len(items)
        try:
            m = gp.Model("binpacking")
            m.Params.OutputFlag = 0

            x = m.addVars(n, n, vtype=GRB.BINARY)
            y = m.addVars(n, vtype=GRB.BINARY)

            m.setObjective(gp.quicksum(y[j] for j in range(n)), GRB.MINIMIZE)

            for i in range(n):
                m.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)

            for j in range(n):
                m.addConstr(gp.quicksum(items[i] * x[i, j] for i in range(n)) <= self.bin_capacity * y[j])

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                return int(m.ObjVal)
            else:
                return None
        except:
            return None

    def generate_instances(self, output_path="/data1/LLMOptChall/LLMs-OPT/results_try/binpacking/binpacking_instances_hard.jsonl"):
        index = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fout:
            for n_items in range(*self.n_items_range):
                count = 0
                while count < self.samples_per_type:
                    instance = self.generate_instance(index, n_items)
                    opt_bins = self.solve_binpacking(instance["items"])
                    if opt_bins is not None:
                        instance["answer"] = opt_bins
                        fout.write(json.dumps(instance) + "\n")
                        count += 1
                        index += 1
        print(f"Generation completed: {index} valid BinPacking instances saved to {output_path}")

    def make_nl_example(self, items, bin_capacity):
        item_lines = "\n".join([
            f"* Product {chr(ord('A') + i)}: weight {w}kg"
            for i, w in enumerate(items)
        ])
        template = (
            "A warehouse manager needs to pack different products into identical shipping containers.\n\n"
            "The available items include:\n"
            "{item_lines}\n\n"
            "Each shipping container has a maximum weight capacity of {bin_capacity}kg. "
            "The manager's goal is to use the minimum number of containers while ensuring all products are packed. "
            "Each product must be assigned to exactly one container, and the total weight in each container cannot exceed its capacity."
        )
        template_1 = (
            "A manager needs to pack products into identical containers."
            "Items:\n{item_lines}\n"
            "Each container holds up to {bin_capacity} kg."
            "Pack all items using the fewest containers. "
            "Each item must go into exactly one container, and the total weight in a container cannot exceed its limit."
        )
        return template.format(item_lines=item_lines, bin_capacity=bin_capacity)

    def map_to_nl(self, input_path="/data1/LLMOptChall/LLMs-OPT/results_try/binpacking/binpacking_instances_hard.jsonl", 
                  output_path="/data1/LLMOptChall/LLMs-OPT/results_try/binpacking/binpacking_nl_hard.jsonl"):
        total = 0
        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            for line in fin:
                data = json.loads(line)
                nl = self.make_nl_example(data["items"], data["bin_capacity"])
                out = {
                    "index": data["index"],
                    "problem_type": "BinPacking",
                    "problem_size": data["n_items"],
                    "nl_problem": nl,
                    "answer": data["answer"]
                }
                fout.write(json.dumps(out) + "\n")
                total += 1

        print(f"Natural language problem generation completed: {total} instances saved to {output_path}")