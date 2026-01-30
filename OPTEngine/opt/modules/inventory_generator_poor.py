import random
import json
import gurobipy as gp
from gurobipy import GRB
from modules.base_generator import BaseGenerator

class Inventory_generator(BaseGenerator):
    def __init__(self,
                 T_range=(5, 21),                # time T
                 demand_range=(10, 60),          # demand
                 I0_range=(0, 100),              # I0
                 Qmin_range=(0, 20),             # Q_min(buy)
                 Qmax_range=(20, 80),            # Q_max
                 p_range=(1, 6),                 # unit price
                 h_range=(1, 3),                 # unit holding cost 
                 c_range=(6, 15),                # out of store cost
                 capacity_factor=(0.8, 1.6),     # Capacity
                 samples_per_T=5,                # sample amount
                 seed=0):
        
        self.T_range = T_range
        self.demand_range = demand_range
        self.I0_range = I0_range
        self.Qmin_range = Qmin_range
        self.Qmax_range = Qmax_range
        self.p_range = p_range
        self.h_range = h_range
        self.c_range = c_range
        self.capacity_factor = capacity_factor
        self.samples_per_T = samples_per_T
        random.seed(seed)

    # ---------- generate One instance randomly ----------
    def generate_instance(self, index, T=None):
        if T is None:
            T = random.randint(self.T_range[0], self.T_range[1] - 1)

        demand = [random.randint(*self.demand_range) for _ in range(T)]
        I0 = random.randint(*self.I0_range)
        Qmin = random.randint(*self.Qmin_range)
        Qmax = random.randint(max(Qmin + 1, self.Qmax_range[0]), self.Qmax_range[1])  # Qmax > Qmin

        # no lead time now

        p = random.randint(*self.p_range)
        h = random.randint(*self.h_range)
        c = random.randint(max(h + 1, self.c_range[0]), self.c_range[1])

        avg_d = sum(demand) / T
        cap_coef = random.uniform(*self.capacity_factor)
        C = max(I0, int(cap_coef * avg_d))

        return {
            "index": index,
            "T": T,
            "I0": I0,
            "Qmin": Qmin,
            "Qmax": Qmax,
            "demand": demand,
            "p": p,
            "h": h,
            "c": c,
            "C": C
        }

    # ---------- return the optimal cost of the instance ----------
    def solve_inventory_lp(self, inst):
        T = inst["T"]
        I0 = inst["I0"]
        Qmin = inst["Qmin"]
        Qmax = inst["Qmax"]
        D = inst["demand"]
        p = inst["p"]
        h = inst["h"]
        c = inst["c"]
        C = inst["C"]

        try:
            m = gp.Model("inventory_lp")
            m.Params.OutputFlag = 0

            # order quantity x_t, immediate arrival (no lead)
            x = m.addVars(range(1, T + 1), lb=Qmin, ub=Qmax,
                          vtype=GRB.CONTINUOUS, name="x")
            # inventory I_t
            I = m.addVars(range(0, T + 1), lb=0.0, ub=C,
                          vtype=GRB.CONTINUOUS, name="I")
            # shortage S_t
            S = m.addVars(range(1, T + 1), lb=0.0,
                          vtype=GRB.CONTINUOUS, name="S")

            # initial inventory
            m.addConstr(I[0] == I0, name="init_inventory")

            # inventory balance: I_t - S_t = I_{t-1} + x_t - D_t
            for t in range(1, T + 1):
                m.addConstr(
                    I[t] - S[t] == I[t - 1] + x[t] - D[t - 1],
                    name=f"bal_{t}"
                )

            # capacity on on-hand inventory after ordering
            for t in range(1, T + 1):
                m.addConstr(
                    I[t - 1] + x[t] <= C,
                    name=f"cap_{t}"
                )

            # objective: ordering + holding + shortage costs
            m.setObjective(
                p * gp.quicksum(x[t] for t in range(1, T + 1)) +
                h * gp.quicksum(I[t] for t in range(1, T + 1)) +
                c * gp.quicksum(S[t] for t in range(1, T + 1)),
                GRB.MINIMIZE
            )

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                return float(m.ObjVal)
            else:
                return None
        except Exception as e:
            return None

    def generate_instances(self, output_path="/data1/LLMOptChall/LLMs-OPT/results_poor_grounding/inventory/inventory_instances.jsonl"):
        index = 0
        total = 0

        with open(output_path, "w", encoding="utf-8") as fout:
            for T in range(self.T_range[0], self.T_range[1]):
                count = 0
                while count < self.samples_per_T:
                    inst = self.generate_instance(index, T)
                    opt_cost = self.solve_inventory_lp(inst)
                    if opt_cost is not None:
                        inst["optimal_cost"] = opt_cost
                        fout.write(json.dumps(inst, ensure_ascii=False) + "\n")
                        count += 1
                        index += 1
                        total += 1

        print(f"Generating Complete: {total} valid Inventory Planning instances are saved at {output_path}")

    def make_template(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        # 删除关于 lead time 的叙述，改为“当期立即可用”
        self.template = (
            "A factory must develop an ordering and inventory plan for a key material "
            "over a planning horizon of {T} days.\n"
            "The initial inventory at the beginning of the planning period is {I0} units.\n"
            "In each period t = 1, ..., {T}, the supplier allows the factory to place an order "
            "whose quantity must lie between {Qmin} and {Qmax} units.\n"
            "Any order placed in a given period becomes immediately available in that same period "
            "to satisfy demand or replenish inventory.\n"
            "The demand for the material in each period is given as follows:\n"
            "{demand_lines}\n\n"
            "Shortages are permitted, but any unmet demand will not be backordered.\n"
            "Throughout the planning horizon, material quantities are allowed to be fractional, "
            "and the total amount of on-hand inventory at any time must not exceed the warehouse "
            "capacity of {C} units.\n"
            "The total cost over the planning horizon consists of three components: "
            "the ordering cost, which equals {p} per unit ordered; "
            "the holding cost, which equals {h} per unit of inventory carried from one period to the next; "
            "and the shortage penalty, which equals {c} per unit of unmet demand. "
            "Please determine the optimal order quantity for each period and track the resulting "
            "inventory and shortage levels so as to minimize the total cost."
        )

    def _fmt_demand_lines(self, demand):
        return "\n".join([f"* Period {t}: demand {d} units" for t, d in enumerate(demand, start=1)])

    def make_nl_example(self, data):
        T = data["T"]
        demand = data["demand"]
        if len(demand) != T:
            raise ValueError(f"Length of 'demand' ({len(demand)}) must equal T ({T}).")

        demand_lines = self._fmt_demand_lines(demand)
        nl = self.template.format(
            T=T,
            I0=data["I0"],
            Qmin=data["Qmin"],
            Qmax=data["Qmax"],
            demand_lines=demand_lines,
            C=data["C"],
            p=data["p"],
            h=data["h"],
            c=data["c"],
        )
        return nl

    def map_to_nl(self, input_path = "/data1/LLMOptChall/LLMs-OPT/results_poor_grounding/inventory/inventory_instances.jsonl", 
                  output_path="/data1/LLMOptChall/LLMs-OPT/results_poor_grounding/inventory/inventory_nl.jsonl"):

        self.make_template(input_path, output_path)
        total = 0

        with open(self.input_path, "r", encoding="utf-8") as fin, \
             open(self.output_path, "w", encoding="utf-8") as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                nl = self.make_nl_example(data)

                out = {
                    "index": data["index"],
                    "problem_type": "Inventory",
                    "problem_size": data["T"],
                    "nl_problem": nl
                }

                if "optimal_cost" in data:
                    out["answer"] = data["optimal_cost"]

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                total += 1

        print(f"The natural language problems have been generated {total} instances saved to {self.output_path}")