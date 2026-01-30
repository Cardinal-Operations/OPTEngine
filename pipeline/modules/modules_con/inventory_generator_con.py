import random
import json
import gurobipy as gp
import os
from gurobipy import GRB
from modules.base_generator import BaseGenerator

class Inventory_generator(BaseGenerator):
    def __init__(self,
                 T_range=(5, 21),                # time T
                 demand_range=(10, 60),          # demand
                 I0_range=(0, 100),              # I0
                 Qmin_range=(0, 20),             # Q_min(buy)
                 Qmax_range=(20, 80),            # Q_max
                 lead_range=(0, 4),              # lead time
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
        self.lead_range = lead_range
        self.p_range = p_range
        self.h_range = h_range
        self.c_range = c_range
        self.capacity_factor = capacity_factor
        self.samples_per_T = samples_per_T
        random.seed(seed)

    def generate_instance(self, index, T=None):
        if T is None:
            T = random.randint(self.T_range[0], self.T_range[1] - 1)

        demand = [random.randint(*self.demand_range) for _ in range(T)]
        I0 = random.randint(*self.I0_range)
        Qmin = random.randint(*self.Qmin_range)
        Qmax = random.randint(max(Qmin + 1, self.Qmax_range[0]), self.Qmax_range[1])  # Qmax > Qmin

        lead = random.randint(*self.lead_range)
        p = random.randint(*self.p_range)
        h = random.randint(*self.h_range)
        c = random.randint(max(h + 1, self.c_range[0]), self.c_range[1])

        avg_d = sum(demand) / T
        cap_coef = random.uniform(*self.capacity_factor)
        C = max(I0, int(cap_coef * avg_d))

        t2 = random.randint(1, T)
        Imin = random.randint(0, max(0, C)) 

        return {
            "index": index,
            "T": T,
            "I0": I0,
            "Qmin": Qmin,
            "Qmax": Qmax,
            "lead": lead,
            "demand": demand,
            "p": p,
            "h": h,
            "c": c,
            "C": C,
            "t2": t2,
            "Imin": Imin
        }

    @staticmethod
    def solutions_different(x_base, x_con, tol=1e-6):
        if x_base is None or x_con is None:
            return False
        keys = sorted(set(x_base.keys()) | set(x_con.keys()))
        for t in keys:
            vb = float(x_base.get(t, 0.0))
            vc = float(x_con.get(t, 0.0))
            if abs(vb - vc) > tol:
                return True
        return False

    @staticmethod
    def objectives_different(obj_base, obj_con, tol=1e-6):
        if obj_base is None or obj_con is None:
            return False
        return abs(float(obj_base) - float(obj_con)) > tol

    def solve_inventory_lp(self, inst, add_extra_constraints=True, return_solution=True):
        """
        add_extra_constraints:
          - True: add I[t2] >= Imin
          - False: baseline (no extra constraint)
        return_solution:
          - True: return (obj, x_sol)
          - False: return obj only
        """
        T = inst["T"]
        I0 = inst["I0"]
        Qmin = inst["Qmin"]
        Qmax = inst["Qmax"]
        l = inst["lead"]
        D = inst["demand"]
        p = inst["p"]
        h = inst["h"]
        c = inst["c"]
        C = inst["C"]

        t2 = inst["t2"]
        Imin = inst["Imin"]

        try:
            m = gp.Model("inventory_lp")
            m.Params.OutputFlag = 0

            x = m.addVars(range(1, T + 1), lb=Qmin, ub=Qmax, vtype=GRB.CONTINUOUS, name="x")
            I = m.addVars(range(0, T + 1), lb=0.0, ub=C, vtype=GRB.CONTINUOUS, name="I")
            S = m.addVars(range(1, T + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="S")

            m.addConstr(I[0] == I0, name="init_inventory")

            for t in range(1, T + 1):
                arrivals = x[t - l] if t - l >= 1 else 0.0
                m.addConstr(I[t] - S[t] == I[t - 1] + arrivals - D[t - 1], name=f"bal_{t}")

            for t in range(1, T + 1):
                arrivals = x[t - l] if t - l >= 1 else 0.0
                m.addConstr(I[t - 1] + arrivals <= C, name=f"cap_{t}")

            if add_extra_constraints:
                m.addConstr(I[t2] >= Imin, name="min_inventory_day_t2")

            m.setObjective(
                p * gp.quicksum(x[t] for t in range(1, T + 1)) +
                h * gp.quicksum(I[t] for t in range(1, T + 1)) +
                c * gp.quicksum(S[t] for t in range(1, T + 1)),
                GRB.MINIMIZE
            )

            m.optimize()

            if m.Status != GRB.OPTIMAL:
                return (None, None) if return_solution else None

            obj = float(m.ObjVal)

            if not return_solution:
                return obj

            x_sol = {t: float(x[t].X) for t in range(1, T + 1)}
            return obj, x_sol

        except Exception:
            return (None, None) if return_solution else None

    def generate_instances(self,
                           output_path="/DATA/disk2/LLMs-OPT/poor_grounding_0/results/inventory/inventory_instances_xi.jsonl",
                           sol_tol=1e-6,
                           obj_tol=1e-6,
                           max_tries_per_T=2000):
        """
        只保留同时满足：
          (1) constrained 最优解 != baseline 最优解
          (2) constrained 最优目标值 != baseline 最优目标值
        """
        index = 0
        total = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fout:
            for T in range(self.T_range[0], self.T_range[1]):
                count = 0
                tries = 0

                while count < self.samples_per_T and tries < max_tries_per_T:
                    tries += 1
                    inst = self.generate_instance(index, T)

                    # baseline (no extra constraint)
                    base_obj, base_x = self.solve_inventory_lp(
                        inst, add_extra_constraints=False, return_solution=True
                    )
                    if base_obj is None:
                        continue

                    con_obj, con_x = self.solve_inventory_lp(
                        inst, add_extra_constraints=True, return_solution=True
                    )
                    if con_obj is None:
                        continue

                    if not self.solutions_different(base_x, con_x, tol=sol_tol):
                        continue

                    if not self.objectives_different(base_obj, con_obj, tol=obj_tol):
                        continue

                    inst["optimal_cost_baseline"] = base_obj
                    inst["optimal_cost"] = con_obj

                    fout.write(json.dumps(inst, ensure_ascii=False) + "\n")
                    count += 1
                    index += 1
                    total += 1

                if count < self.samples_per_T:
                    print(
                        f"[WARN] T={T}: only generated {count}/{self.samples_per_T} valid instances "
                        f"after {tries} tries (might be too hard to find instances where both solution "
                        f"and objective change)."
                    )

        print(f"Generating Complete: {total} filtered Inventory Planning instances are saved at {output_path}")

    def make_template(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        self.template = (
            "A factory must develop an ordering and inventory plan for a key material "
            "over a planning horizon of {T} days.\n"
            "The initial inventory at the beginning of the planning period is {I0} units.\n"
            "In each period t = 1, ..., {T}, the supplier allows the factory to place an order "
            "whose quantity must lie between {Qmin} and {Qmax} units.\n"
            "However, each order placed will take {lead} day(s) to arrive before it can be used "
            "to satisfy demand or replenish inventory.\n"
            "The demand for the material in each period is given as follows:\n"
            "{demand_lines}\n\n"
            "Shortages are permitted, but any unmet demand will not be backordered.\n"
            "Throughout the planning horizon, material quantities are allowed to be fractional, "
            "and the total amount of on-hand inventory at any time must not exceed the warehouse "
            "capacity of {C} units.\n"
            "\n"
            "On day {t2}, the on-hand inventory must be at least {Imin} units.\n"
            "\n"
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
            lead=data["lead"],
            demand_lines=demand_lines,
            C=data["C"],
            p=data["p"],
            h=data["h"],
            c=data["c"],
            t2=data["t2"],
            Imin=data["Imin"],
        )
        return nl

    def map_to_nl(self,
                  input_path="/DATA/disk2/LLMs-OPT/poor_grounding_0/results/inventory/inventory_instances_xi.jsonl",
                  output_path="/DATA/disk2/LLMs-OPT/poor_grounding_0/results/inventory/inventory_nl_xi.jsonl"):

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