import random
import json
import math
import os
import gurobipy as gp
from gurobipy import GRB
from modules.base_generator import BaseGenerator

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
        """
        求解时全局禁止 0↔1 两条弧：不创建 (0,1) 和 (1,0) 的变量。
        """
        n = len(coords)
        if n < 3:
            return None

        # 构造距离：显式不包含 0↔1 两条弧
        dist = {
            (i, j): math.dist(coords[i], coords[j])
            for i in range(n) for j in range(n)
            if i != j and not ((i == 0 and j == 1) or (i == 1 and j == 0))
        }

        try:
            m = gp.Model("tsp")
            m.Params.OutputFlag = 0

            # 只对 dist 中有的弧建变量
            x = m.addVars(dist.keys(), vtype=GRB.BINARY)

            m.setObjective(
                gp.quicksum(x[i, j] * dist[i, j] for (i, j) in dist),
                GRB.MINIMIZE
            )

            # 每个城市入度 = 1、出度 = 1（只对存在的弧求和）
            for i in range(n):
                m.addConstr(gp.quicksum(x[j, i] for j in range(n) if (j, i) in x) == 1)
                m.addConstr(gp.quicksum(x[i, j] for j in range(n) if (i, j) in x) == 1)

            # MTZ 去子环约束
            u = m.addVars(n, vtype=GRB.INTEGER)
            for i in range(1, n):
                for j in range(1, n):
                    if i != j and (i, j) in x:
                        m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                return m.ObjVal
            else:
                return None
        except:
            return None
    def generate_instances(
        self,
        output_path="/data1/LLMOptChall/LLMs-OPT/results_poor_grounding/tsp/tsp_instances.jsonl"
    ):
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
        print(f"Generation completed: {index} valid TSP instances saved to {output_path}")

    def compute_distance_matrix(self, coords):
        """
        所有距离矩阵默认就禁止 0↔1：把 D[0][1], D[1][0] 设为 inf。
        这样任何用到这个矩阵的地方（包括 NL 描述）都体现这个约束。
        """
        n = len(coords)
        D = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dist = math.dist(coords[i], coords[j])
                D[i][j] = D[j][i] = dist

        # 全局禁止第一个和第二个城市之间的直接连边
        if n >= 2:
            D[0][1] = D[1][0] = float("inf")
        return D
    
    def compute_distance_matrix_forbid_first_two(self, coords, big_M=float("inf")):
        return self.compute_distance_matrix(coords, forbidden_edges=[(0, 1)], big_M=big_M)
    
    def make_nl_example(self, coords, distance_matrix):
        n = len(coords)
        names = [chr(ord('A') + i) for i in range(n)]

        city_lines = "\n".join([
            f"City {names[i]}: Located at coordinates ({coords[i][0]},{coords[i][1]})"
            for i in range(n)
        ])

        distance_lines = []
        for i in range(n):
            for j in range(i + 1, n):
                dij = distance_matrix[i][j]

                # 对 0-1 边（或者任何 inf 边）输出“没有直接道路”
                if math.isinf(dij):
                    distance_lines.append(
                        f""
                    )
                else:
                    distance_lines.append(
                        f"{names[i]} to {names[j]}: {dij:.1f} km"
                    )
        distance_text = "\n".join(distance_lines)

        # 注意：example_routes 在原模板里其实没被用到，这里保持原样
        example_routes = "\n".join([
            f"* {' → '.join(names)} → {names[0]}",
            f"* {names[0]} → {names[-1]} → "
            + " → ".join(reversed(names[1:-1]))
            + f" → {names[0]}",
            f"* {names[0]} → {names[1]} → {names[-1]} → "
            + " → ".join(names[2:-1])
            + f" → {names[0]}",
        ])

        template = f"""Consider a delivery service that needs to visit {n} cities:
                {city_lines}

                The distances between cities are measured in kilometers:{distance_text}.
                There is a no direct road between City {names[0]} and City {names[1]}.

                The goal is to find the shortest possible route that visits each city exactly once and returns to the starting city."""
        return template
    
    def map_to_nl(self, input_path="/data1/LLMOptChall/LLMs-OPT/results_poor_grounding/tsp/tsp_instances.jsonl", 
                  output_path="/data1/LLMOptChall/LLMs-OPT/results_poor_grounding/tsp/tsp_nl.jsonl"):
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