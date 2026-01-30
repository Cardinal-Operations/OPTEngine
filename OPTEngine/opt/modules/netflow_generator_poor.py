import random
import json
import os
import gurobipy as gp
from gurobipy import GRB
from modules.base_generator import BaseGenerator

class NetFlowGenerator(BaseGenerator):
    def __init__(
        self,
        n_nodes_range=(3, 15),
        supply_range=(10, 100),
        demand_range=(10, 100),
        shipping_cost_range=(1, 10),
        capacity_range=(5, 100),
        samples_per_type=10,
        seed=42
    ):
        super().__init__(samples_per_type=samples_per_type, seed=seed)
        self.n_nodes_range = n_nodes_range
        self.supply_range = supply_range
        self.demand_range = demand_range
        self.shipping_cost_range = shipping_cost_range
        self.capacity_range = capacity_range
        random.seed(seed)

    def generate_instance(self, index, n_nodes):
        warehouses = [f"w{i}" for i in range(n_nodes)]
        stores = [f"s{j}" for j in range(n_nodes)]

    # Positive supply / demand
        supply = {w: random.randint(*self.supply_range) for w in warehouses}
        demand = {s: random.randint(*self.demand_range) for s in stores}

    # Balance supply/demand (may reduce values accidentally to 0)
        total_supply = sum(supply.values())
        total_demand = sum(demand.values())

        if total_supply > total_demand:
            diff = total_supply - total_demand
            for w in warehouses:
                dec = min(diff, supply[w]-1)  # ensure stays >= 1
                supply[w] -= dec
                diff -= dec
                if diff <= 0:
                    break
        elif total_demand > total_supply:
            diff = total_demand - total_supply
            for s in stores:
                dec = min(diff, demand[s]-1)  # ensure stays >= 1
                demand[s] -= dec
                diff -= dec
                if diff <= 0:
                    break

    # Ensure final values are strictly > 0
        for w in warehouses:
            supply[w] = max(1, supply[w])
        for s in stores:
            demand[s] = max(1, demand[s])

    # Capacity also strictly > 0
        arcs = [(w, s) for w in warehouses for s in stores]
        shipping_costs = {arc: random.randint(*self.shipping_cost_range) for arc in arcs}
        capacities = {arc: max(1, random.randint(*self.capacity_range)) for arc in arcs}

    # Forbidden arc must be set to 0
        forbidden_arc = ("w0", "s1")
        if forbidden_arc in capacities:
            capacities[forbidden_arc] = 0

        return {
            "index": index,
            "n_nodes": n_nodes,
            "warehouses": warehouses,
            "stores": stores,
            "supply": supply,
            "demand": demand,
            "arcs": arcs,
            "shipping_costs": shipping_costs,
            "capacities": capacities
        }

    def solve_netflow(self, instance):
        try:
            m = gp.Model("netflow")
            m.Params.OutputFlag = 0

            arcs = instance["arcs"]
            costs = instance["shipping_costs"]
            caps = instance["capacities"]
            supply = instance["supply"]
            demand = instance["demand"]
            warehouses = instance["warehouses"]
            stores = instance["stores"]

            flow = m.addVars(arcs, lb=0, name="flow")

            m.setObjective(
                gp.quicksum(costs[(i, j)] * flow[i, j] for (i, j) in arcs),
                GRB.MINIMIZE
            )

            for w in warehouses:
                m.addConstr(
                    gp.quicksum(flow[w, s] for s in stores) <= supply[w],
                    name=f"supply_{w}"
                )

            for s in stores:
                m.addConstr(
                    gp.quicksum(flow[w, s] for w in warehouses) == demand[s],
                    name=f"demand_{s}"
                )

            for (i, j) in arcs:
                m.addConstr(flow[i, j] <= caps[(i, j)], name=f"cap_{i}_{j}")

            # ✨ NEW — Hard constraint for forbidden route
            if ("w0", "s1") in flow:
                m.addConstr(flow["w0", "s1"] == 0, name="forbid_w0_s1")

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                return m.ObjVal
            else:
                return None
        except:
            return None


    # ✨ NEW — The entire generate_instances() function was missing before
    def generate_instances(self, output_path="/data1/LLMOptChall/LLMs-OPT/results_try/netflow/netflow_instances_poor.jsonl"):
        index = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as fout:
            for n_nodes in range(*self.n_nodes_range):
                count = 0
                while count < self.samples_per_type:
                    instance = self.generate_instance(index, n_nodes)
                    opt_cost = self.solve_netflow(instance)

                    if opt_cost is not None:
                        instance["answer"] = opt_cost
                        instance["shipping_costs"] = {f"{i}->{j}": c for (i, j), c in instance["shipping_costs"].items()}
                        instance["capacities"] = {f"{i}->{j}": c for (i, j), c in instance["capacities"].items()}
                        fout.write(json.dumps(instance) + "\n")

                        count += 1
                        index += 1

        print(f"Generation completed: {index} valid NetFlow instances saved to {output_path}")


    # ✨ NEW — This entire NL function was missing too
    def make_nl_example(self, warehouses, stores, supply, demand, capacities, costs):
        n = len(warehouses)

        # ✅ 用编号而不是字母，避免 Warehouse D / Store D 混在一起
        name_map = {}
        for i, w in enumerate(warehouses, start=1):
            name_map[w] = f"Warehouse {i}"
        for j, s in enumerate(stores, start=1):
            name_map[s] = f"Store {j}"

        warehouse_lines = "\n".join([
            f"* {name_map[w]}: Supply capacity = {supply[w]} units" for w in warehouses
        ])

        store_lines = "\n".join([
            f"* {name_map[s]}: Demand = {demand[s]} units" for s in stores
        ])

        arc_lines = ""
        for w in warehouses:
            arc_lines += f"* From {name_map[w]}:\n"
            for s in stores:
                arc_key = f"{w}->{s}"
                arc_lines += (
                    f"  - to {name_map[s]}: "
                    f"capacity = {capacities[arc_key]}, cost = {costs[arc_key]}\n"
                )

        # ✅ 这里动态用 w0 / s1 映射到对应“第几个”仓库 / 商店
        forbidden_sentence = ""
        if "w0" in name_map and "s1" in name_map:
            forbidden_sentence = (
                f"\nAdditionally, {name_map['w0']} is NOT allowed to ship goods to "
                f"{name_map['s1']}"
            )

        template = (
            f"A logistics company needs to ship goods from {n} warehouses to {n} retail stores:\n"
            f"Each warehouse has a supply capacity:\n"
            f"{warehouse_lines}\n"
            f"Each retail store has a fixed demand:\n"
            f"{store_lines}\n\n"
            f"The transportation routes have the following capacities and shipping costs:\n"
            f"{arc_lines.strip()}\n"
            f"{forbidden_sentence}"
            "The objective is to minimize total shipping cost while meeting all store demand "
            "and respecting all constraints."
        )
        return template
    
    def map_to_nl(self, input_path="/data1/LLMOptChall/LLMs-OPT/results_try/netflow/netflow_instances_poor.jsonl", output_path="/data1/LLMOptChall/LLMs-OPT/results_try/netflow/netflow_nl_poor.jsonl"):
        total = 0
        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            for line in fin:
                data = json.loads(line)
                nl = self.make_nl_example(
                    data["warehouses"],
                    data["stores"],
                    data["supply"],
                    data["demand"],
                    data["capacities"],
                    data["shipping_costs"]
                )
                out = {
                    "index": data["index"],
                    "problem_type": "NetworkFlow",
                    "problem_size": data["n_nodes"],
                    "nl_problem": nl,
                    "answer": data["answer"]
                }
                fout.write(json.dumps(out) + "\n")
                total += 1

        print(f"Natural language problem generation completed: {total} instances saved to {output_path}")