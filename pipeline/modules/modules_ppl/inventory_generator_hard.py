import random
import json
import os
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

    # ---------- generate One instance randomly ----------
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
            "C": C
        }

    # ---------- return the optimal cost of the instance ----------
    def solve_inventory_lp(self, inst):
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

        try:
            m = gp.Model("inventory_lp")
            m.Params.OutputFlag = 0

            x = m.addVars(range(1, T + 1), lb=Qmin, ub=Qmax, vtype=GRB.CONTINUOUS, name="x") 
            I = m.addVars(range(0, T + 1), lb=0.0, ub=C, vtype=GRB.CONTINUOUS, name="I") 
            S = m.addVars(range(1, T + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="S")

            m.addConstr(I[0] == I0, name="init_inventory")

            for t in range(1, T + 1):
                arrivals = 0.0
                if t - l >= 1:
                    arrivals = x[t - l]
                m.addConstr(I[t] - S[t] == I[t - 1] + arrivals - D[t - 1], name=f"bal_{t}")

            for t in range(1, T + 1):
                arrivals = x[t - l] if t - l >= 1 else 0.0
                m.addConstr(I[t-1] + arrivals <= C, name=f"cap_{t}")

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

    def generate_instances(self, output_path="/data1/LLMOptChall/LLMs-OPT/ppl/results/inventory/inventory_instances_3.jsonl"):
        index = 0
        total = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
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

        self.template = (
            "A production facility is required to synthesize an intertemporal procurement and inventory-control policy for a mission-critical raw material "
"over a finite, discretely indexed planning horizon of {T} days.\n"
"At the inception of the horizon (prior to period 1), the system is endowed with an initial on-hand stock of {I0} units.\n"
"In each period t = 1, ..., {T}, the decision-maker may submit a replenishment order to an upstream supplier, subject to an admissible-order "
"constraint stipulating that the order quantity must lie within the closed interval [{Qmin}, {Qmax}] units.\n"
"Replenishment is subject to a deterministic, strictly positive lead time: any order placed in period t is realized only after {lead} day(s), "
"at which point the delivered material can be deployed to satisfy contemporaneous demand or to augment the inventory state.\n"
"The exogenously specified demand stream for the material over the planning horizon is given by the following deterministic profile:\n"
"{demand_lines}\n\n"
"Inventory shortfalls are allowed within the model; however, any demand that remains unfilled in the period of its occurrence is interpreted as "
"lost sales and is not propagated forward as backlog to subsequent periods.\n"
"Material quantities are modeled as continuous decision variables, and the storage subsystem is governed by a hard capacity constraint: at no "
"point in time may the on-hand inventory exceed {C} units.\n"
"The total cost accumulated over the planning horizon comprises three components: an ordering cost of {p} per unit ordered; a holding cost of "
"{h} per unit of inventory carried from one period to the next; and a shortage (lost-demand) penalty of {c} per unit of unmet demand.\n"
"Your task is to specify, for each period t, the order quantity so as to induce an evolution of the inventory and shortage levels that minimizes "
"the aggregate cost over the entire planning horizon."
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
        )
        return nl

    def map_to_nl(self, input_path = "/data1/LLMOptChall/LLMs-OPT/ppl/results/inventory/inventory_instances_3.jsonl", 
                  output_path="/data1/LLMOptChall/LLMs-OPT/ppl/results/inventory/inventory_nl_3.jsonl"):

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
