import random
import json
import os
import gurobipy as gp
from gurobipy import GRB
from OPTEngine.modules.base_generator import BaseGenerator

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

        supply = {w: random.randint(*self.supply_range) for w in warehouses}
        demand = {s: random.randint(*self.demand_range) for s in stores}

        # 平衡供需
        total_supply = sum(supply.values())
        total_demand = sum(demand.values())

        if total_supply > total_demand:
            diff = total_supply - total_demand
            for w in warehouses:
                dec = min(diff, supply[w])
                supply[w] -= dec
                diff -= dec
                if diff <= 0:
                    break
        elif total_demand > total_supply:
            diff = total_demand - total_supply
            for s in stores:
                dec = min(diff, demand[s])
                demand[s] -= dec
                diff -= dec
                if diff <= 0:
                    break

        arcs = [(w, s) for w in warehouses for s in stores]
        shipping_costs = {arc: random.randint(*self.shipping_cost_range) for arc in arcs}
        capacities = {arc: random.randint(*self.capacity_range) for arc in arcs}

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

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                return m.ObjVal
            else:
                return None
        except:
            return None

    def generate_instances(self, output_path="/data1/LLMOptChall/LLMs-OPT/results_try/netflow/netflow_instances.jsonl"):
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
        print(f"Generation completed: {index} valid BinPacking instances saved to {output_path}")

    def make_nl_problem(self, warehouses, stores, supply, demand, capacities, costs):
        n = len(warehouses)
        name_map = {w: f"Warehouse {chr(ord('A') + i)}" for i, w in enumerate(warehouses)}
        name_map.update({s: f"Store {chr(ord('D') + j)}" for j, s in enumerate(stores)})

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
                arc_lines += f"  - to {name_map[s]}: capacity = {capacities[arc_key]}, cost = {costs[arc_key]}\n"

        template = (
            f"A logistics company needs to ship goods from {n} warehouses to {n} retail stores:\n"
            f"Each warehouse has a supply capacity:\n"
            f"{warehouse_lines}\n"
            f"Each retail store has a fixed demand:\n"
            f"{store_lines}\n\n"
            f"The transportation routes between each warehouse and store have specific capacity limits and shipping costs (cost per unit):\n"
            f"{arc_lines.strip()}\n\n"
            "The company wants to determine how many units of goods to ship from each warehouse to each store in order to minimize the total shipping cost, "
            "while satisfying all store demands, not exceeding any warehouse’s supply, and respecting the capacity limits of each transportation route."
        )
        return template

    def map_to_nl(self, input_path="/data1/LLMOptChall/LLMs-OPT/results_try/netflow/netflow_instances.jsonl", output_path="/data1/LLMOptChall/LLMs-OPT/results_try/netflow/netflow_nl.jsonl"):
        total = 0
        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            for line in fin:
                data = json.loads(line)
                nl = self.make_nl_problem(
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