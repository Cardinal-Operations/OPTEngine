import random
import json
import math
import os
import gurobipy as gp
from gurobipy import GRB
from modules.base_generator import BaseGenerator


class TSPGenerator(BaseGenerator):
    def __init__(
        self,
        n_cities_range=(4, 20),
        coord_range=(0, 200),
        samples_per_type=10,
        seed=42
    ):
        super().__init__(samples_per_type=samples_per_type, seed=seed)
        self.n_cities_range = n_cities_range
        self.coord_range = coord_range
        random.seed(seed)

    # --------------------------------------------------
    # 基础：生成城市坐标
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 核心：TSP 求解（禁止 0↔1；(1,2) 与 (2,3) 二选一）
    # --------------------------------------------------
    def solve_tsp(self, coords):
        """
        Constraints (consistent with the NL template):
        1) No direct road between City 0 and City 1 (do not create x[0,1], x[1,0]).
        2) Exactly one of the undirected roads (1,2) and (2,3) is included in the tour.
           With directed arc vars, this is:
           (x[1,2]+x[2,1]) + (x[2,3]+x[3,2]) == 1
        """
        n = len(coords)
        if n < 4:  # template references cities 0,1,2,3
            return None

        # 距离字典：不包含 0↔1
        dist = {
            (i, j): math.dist(coords[i], coords[j])
            for i in range(n)
            for j in range(n)
            if i != j and not ((i == 0 and j == 1) or (i == 1 and j == 0))
        }

        try:
            m = gp.Model("tsp")
            m.Params.OutputFlag = 0

            # 决策变量
            x = m.addVars(dist.keys(), vtype=GRB.BINARY, name="x")

            # 目标函数
            m.setObjective(
                gp.quicksum(x[i, j] * dist[i, j] for (i, j) in dist),
                GRB.MINIMIZE
            )

            # 入度 = 1，出度 = 1（只对存在的弧求和）
            for i in range(n):
                m.addConstr(
                    gp.quicksum(x[j, i] for j in range(n) if (j, i) in x) == 1,
                    name=f"in_{i}"
                )
                m.addConstr(
                    gp.quicksum(x[i, j] for j in range(n) if (i, j) in x) == 1,
                    name=f"out_{i}"
                )

            # ✅ 二选一 XOR：undirected (1,2) 与 (2,3) 恰好选一条
            e12 = gp.quicksum(x[a, b] for (a, b) in [(1, 2), (2, 1)] if (a, b) in x)
            e23 = gp.quicksum(x[a, b] for (a, b) in [(2, 3), (3, 2)] if (a, b) in x)
            m.addConstr(e12 + e23 == 1, name="xor_edge_12_23")

            # MTZ 去子环（加边界更稳）
            u = m.addVars(n, lb=0, ub=n - 1, vtype=GRB.INTEGER, name="u")
            m.addConstr(u[0] == 0, name="u0")
            for i in range(1, n):
                for j in range(1, n):
                    if i != j and (i, j) in x:
                        m.addConstr(
                            u[i] - u[j] + n * x[i, j] <= n - 1,
                            name=f"mtz_{i}_{j}"
                        )

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                return m.ObjVal
            else:
                return None

        except gp.GurobiError:
            return None

    # --------------------------------------------------
    # 距离矩阵（NL & solver 一致）
    # --------------------------------------------------
    def compute_distance_matrix(self, coords):
        """
        距离矩阵中：
        - D[0][1] = D[1][0] = inf （表示无直接道路）
        """
        n = len(coords)
        D = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                dist = math.dist(coords[i], coords[j])
                D[i][j] = D[j][i] = dist

        if n >= 2:
            D[0][1] = D[1][0] = float("inf")

        return D

    # --------------------------------------------------
    # NL 模板（与约束严格一致）
    # --------------------------------------------------
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
                if not math.isinf(dij):
                    distance_lines.append(f"{names[i]} to {names[j]}: {dij:.1f} km")
        distance_text = "\n".join(distance_lines)

        template = f"""Consider a delivery service that needs to visit {n} cities:
{city_lines}

The distances between cities are measured in kilometers:
{distance_text}

There is no direct road between City {names[0]} and City {names[1]}.
In addition, exactly one of the following two roads must be included in the tour: the road between City {names[1]} and City {names[2]}, the road between City {names[2]} and City {names[3]}.

The goal is to find the shortest possible route that visits each city exactly once
and returns to the starting city.
"""
        return template

    # --------------------------------------------------
    # 批量生成实例
    # --------------------------------------------------
    def generate_instances(
        self,
        output_path="/data1/LLMOptChall/LLMs-OPT/poor_grounding/results/tsp/tsp_instances_new.jsonl"
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

    # --------------------------------------------------
    # NL 映射
    # --------------------------------------------------
    def map_to_nl(
        self,
        input_path="/data1/LLMOptChall/LLMs-OPT/poor_grounding/results/tsp/tsp_instances_new.jsonl",
        output_path="/data1/LLMOptChall/LLMs-OPT/poor_grounding/results/tsp/tsp_nl_new.jsonl"
    ):
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