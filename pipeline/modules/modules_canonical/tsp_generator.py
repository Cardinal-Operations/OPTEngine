import random
import json
import math
import os
import gurobipy as gp
from gurobipy import GRB
from OPTEngine.modules.base_generator import BaseGenerator

class TSPGenerator(BaseGenerator):
    def __init__(self, n_cities_range=(4, 20), coord_range=(0, 200), samples_per_type=10, seed=42):
        super().__init__(samples_per_type=samples_per_type, seed=seed)
        self.n_cities_range = n_cities_range
        self.coord_range = coord_range
        random.seed(seed)

    def generate_coordinates(self, n_cities):
        return [
            [random.randint(*self.coord_range), random.randint(*self.coord_range)]
            for _ in range(n_cities)
        ]

    def generate_instance(self, index, n_cities=None):
        if n_cities is None:
            n_cities = random.randint(*self.n_cities_range)
        coords = self.generate_coordinates(n_cities)
        return {
            "index": index,
            "n_cities": n_cities,
            "coords": coords
        }

    def solve_tsp(self, coords):
        n = len(coords)
        if n < 3:
            return None

        dist = {
            (i, j): math.dist(coords[i], coords[j])
            for i in range(n) for j in range(n) if i != j
        }

        try:
            m = gp.Model("tsp")
            m.Params.OutputFlag = 0
            x = m.addVars(dist.keys(), vtype=GRB.BINARY)

            m.setObjective(gp.quicksum(x[i, j] * dist[i, j] for i, j in dist), GRB.MINIMIZE)

            for i in range(n):
                m.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1)
                m.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1)

            u = m.addVars(n, vtype=GRB.INTEGER)
            for i in range(1, n):
                for j in range(1, n):
                    if i != j:
                        m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                return m.ObjVal
            else:
                return None
        except:
            return None

    def generate_instances(self, output_path="/data1/LLMOptChall/LLMs-OPT/poor_grounding_hard/tsp/tsp_instances_original.jsonl"):
        index = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fout:
            for n_cities in range(*self.n_cities_range):
                count = 0
                while count < self.samples_per_type:
                    instance = self.generate_instance(index, n_cities)
                    coords = [tuple(p) for p in instance["coords"]]
                    opt_len = self.solve_tsp(coords)
                    if opt_len is not None:
                        instance["answer"] = opt_len
                        fout.write(json.dumps(instance) + "\n")
                        count += 1
                        index += 1
        print(f"Generation completed: {index} valid BinPacking instances saved to {output_path}")

    def compute_distance_matrix(self, coords):
        n = len(coords)
        D = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dist = math.dist(coords[i], coords[j])
                D[i][j] = D[j][i] = dist
        return D

    def make_nl_example(self, coords, distance_matrix):
        n = len(coords)
        names = [chr(ord('A') + i) for i in range(n)]

        city_lines = "\n".join([
            f"* City {names[i]}: Located at coordinates ({coords[i][0]},{coords[i][1]})"
            for i in range(n)
        ])

        distance_lines = []
        for i in range(n):
            for j in range(i + 1, n):
                distance_lines.append(f"* {names[i]} to {names[j]}: {distance_matrix[i][j]:.1f} km")
        distance_text = "\n".join(distance_lines)

        example_routes = "\n".join([
            f"* {' → '.join(names)} → {names[0]}",
            f"* {names[0]} → {names[-1]} → " + " → ".join(reversed(names[1:-1])) + f" → {names[0]}",
            f"* {names[0]} → {names[1]} → {names[-1]} → " + " → ".join(names[2:-1]) + f" → {names[0]}"
        ])

        template_1 = f"""Consider a delivery service that needs to visit {n} cities:
{city_lines}

The distances between cities are measured in kilometers:
{distance_text}

The goal is to find the shortest possible route that visits each city exactly once and returns to the starting city. This creates a 'tour' that minimizes the total travel distance. The problem becomes significantly more complex as the number of cities increases, as the number of possible routes grows factorially with n cities (n!).

For example, with just these {n} cities, starting from City A, some possible routes are:
{example_routes}

The challenge is to determine which of all possible routes has the minimum total distance."""
        
        template = f"""Consider a delivery service that needs to visit {n} cities.

The distances between cities are measured in kilometers:
{distance_text}

The goal is to find the shortest possible route that visits each city exactly once and returns to the starting city."""
        return template

    def map_to_nl(self, input_path="/data1/LLMOptChall/LLMs-OPT/poor_grounding_hard/tsp/tsp_instances_original.jsonl", 
                  output_path="/data1/LLMOptChall/LLMs-OPT/poor_grounding_hard/tsp/tsp_nl_original.jsonl"):
        total = 0
        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            for line in fin:
                data = json.loads(line)
                coords = [tuple(p) for p in data["coords"]]
                D = self.compute_distance_matrix(coords)
                nl = self.make_nl_example(coords, D)

                out = {
                    "index": data["index"],
                    "problem_type": "TSP",
                    "problem_size": data["n_cities"],
                    "nl_problem": nl,
                    "answer": data["answer"]
                }

                fout.write(json.dumps(out) + "\n")
                total += 1

        
        print(f"Natural language problem generation completed: {total} instances saved to {output_path}")