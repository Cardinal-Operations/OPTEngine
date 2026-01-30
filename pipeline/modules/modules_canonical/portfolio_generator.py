import random
import json
import gurobipy as gp
from gurobipy import GRB
from OPTEngine.modules.base_generator import BaseGenerator

class Portfolio_generator(BaseGenerator):
    def __init__(self, 
                 I_range=(5, 21),               # number of assets' category
                 r_range=(0.02, 0.20),          # expected return ri
                 v_range=(0.01, 0.30),          # risk number vi
                 l_max=0.10,                    # lower bound of percentage li
                 u_minmax=(0.30, 0.90),         # upper bound of percentage uj
                 Lmin_range=(0.20, 0.60),       # L_min factor
                 Rmin_factor=(0.60, 0.95),      # R_min factor
                 Vmax_factor=(1.00, 1.50),      # V_max factor
                 samples_per_I=10,              # number of samples
                 seed=42):
        
        self.I_range = I_range
        self.r_range = r_range
        self.v_range = v_range
        self.l_max = l_max
        self.u_minmax = u_minmax
        self.Lmin_range = Lmin_range
        self.Rmin_factor = Rmin_factor
        self.Vmax_factor = Vmax_factor
        self.samples_per_I = samples_per_I
        random.seed(seed)

    def _rand_list(self, I, lo, hi):
        return [random.uniform(lo, hi) for _ in range(I)]

    def _build_feasible_x(self, l, u):
        I = len(l)
        x = l[:]
        remain = 1.0 - sum(x)
        if remain < -1e-9:
            return None

        caps = [max(0.0, u[i] - x[i]) for i in range(I)]
        if remain <= 1e-12:
            s = sum(x)
            if s <= 0: 
                return None
            x = [xi / s for xi in x]
            return x

        w = [random.random() + 1e-9 for _ in range(I)]
        sw = sum(w)
        w = [wi / sw for wi in w]
        for i in range(I):
            add = min(caps[i], remain * w[i])
            x[i] += add

        remain = 1.0 - sum(x)
        if remain > 1e-10:
            for i in range(I):
                room = max(0.0, u[i] - x[i])
                if room <= 0:
                    continue
                add = min(room, remain)
                x[i] += add
                remain -= add
                if remain <= 1e-12:
                    break

        s = sum(x)
        if s <= 0:
            return None
        x = [min(max(x[i] / s, l[i]), u[i]) for i in range(I)]
        ss = sum(x)
        if ss <= 0:
            return None
        x = [xi / ss for xi in x]

        for i in range(I):
            if x[i] < l[i] - 1e-9 or x[i] > u[i] + 1e-9:
                return None
        if abs(sum(x) - 1.0) > 1e-6:
            return None
        return x

    def _enforce_liquidity(self, x, l, u, L_idx, L_min):
        I = len(x)
        liquid_now = sum(x[i] for i in L_idx)
        if liquid_now + 1e-9 >= L_min:
            return x

        need = L_min - liquid_now
        nonL = [j for j in range(I) if j not in L_idx]

        take_pool = sum(max(0.0, x[j] - l[j]) for j in nonL)
        add_pool  = sum(max(0.0, u[i] - x[i]) for i in L_idx)
        move = min(need, take_pool, add_pool)
        if move <= 1e-12:
            return None

        remaining = move
        for j in nonL:
            room = max(0.0, x[j] - l[j])
            if room <= 0:
                continue
            delta = min(room, remaining)
            x[j] -= delta
            remaining -= delta
            if remaining <= 1e-12:
                break
        inc = move
        for i in L_idx:
            room = max(0.0, u[i] - x[i])
            if room <= 0:
                continue
            delta = min(room, inc)
            x[i] += delta
            inc -= delta
            if inc <= 1e-12:
                break

        s = sum(x)
        if s <= 0:
            return None
        x = [max(l[t], min(u[t], xt)) for t, xt in enumerate(x)]
        s = sum(x)
        x = [xt / s for xt in x]

        if sum(x[i] for i in L_idx) + 1e-9 >= L_min and abs(sum(x) - 1.0) <= 1e-6:
            return x
        return None

    def generate_instance(self, index, I=None):
        if I is None:
            I = random.randint(self.I_range[0], self.I_range[1] - 1)

        r = self._rand_list(I, *self.r_range)
        v = self._rand_list(I, *self.v_range)

        l = [random.uniform(0.0, self.l_max) for _ in range(I)]
        sum_l = sum(l)
        if sum_l > 1.0:
            l = [li / (sum_l + 1e-12) for li in l]

        u = self._rand_list(I, *self.u_minmax)
        for i in range(I):
            u[i] = max(u[i], l[i] + 1e-6)
        if sum(u) < 1.0:
            gap = 1.0 - sum(u) + 1e-6
            u = [ui + gap / I for ui in u]

        x0 = self._build_feasible_x(l, u)
        if x0 is None:
            l = [li * 0.8 for li in l]
            u = [min(1.0, ui * 1.2) for ui in u]
            for i in range(I):
                u[i] = max(u[i], l[i] + 1e-6)
            if sum(u) < 1.0:
                add = (1.0 - sum(u) + 1e-6) / I
                u = [ui + add for ui in u]
            x0 = self._build_feasible_x(l, u)
            if x0 is None:
                raise RuntimeError("Failed to build feasible x from bounds.")
        k = random.randint(1, max(1, I - 1))
        L_idx = sorted(random.sample(range(I), k))

        sum_l_L = sum(l[i] for i in L_idx)
        sum_u_L = min(1.0, sum(u[i] for i in L_idx))
        raw_Lmin = random.uniform(*self.Lmin_range)
        L_min = min(max(raw_Lmin, sum_l_L), sum_u_L)

        xL = self._enforce_liquidity(x0[:], l, u, L_idx, L_min)
        if xL is None:
            L_min = min(sum_u_L, 1.0)
            xL = self._enforce_liquidity(x0[:], l, u, L_idx, L_min)
            if xL is None:
                L_min = sum(x0[i] for i in L_idx)
                xL = x0[:]

        ref_return = sum(r[i] * xL[i] for i in range(I))
        ref_risk   = sum(v[i] * xL[i] for i in range(I))

        alpha = random.uniform(*self.Rmin_factor)
        beta  = random.uniform(*self.Vmax_factor)

        R_min = alpha * ref_return
        V_max = beta  * ref_risk

        instance = {
            "index": index,
            "I": I,
            "r": r,
            "v": v,
            "l": l,
            "u": u,
            "L_set": L_idx,
            "L_min": L_min,
            "V_max": V_max,
            "R_min": R_min
        }
        return instance

    # ---------- solve by groubi ----------
    def solve_portfolio(self, inst):

        I     = inst["I"]
        r     = inst["r"]
        v     = inst["v"]
        l     = inst["l"]
        u     = inst["u"]
        L_set = inst["L_set"]
        L_min = inst["L_min"]
        V_max = inst["V_max"]
        R_min = inst["R_min"]

        try:
            m = gp.Model("portfolio")
            m.Params.OutputFlag = 0

            x = m.addVars(I, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
            m.setObjective(gp.quicksum(r[i] * x[i] for i in range(I)), GRB.MAXIMIZE)
            m.addConstr(gp.quicksum(x[i] for i in range(I)) == 1.0, name="sum_to_one")
            m.addConstr(gp.quicksum(v[i] * x[i] for i in range(I)) <= V_max, name="risk_budget")

            for i in range(I):
                m.addConstr(x[i] >= l[i], name=f"lb_{i}")
                m.addConstr(x[i] <= u[i], name=f"ub_{i}")

            m.addConstr(gp.quicksum(r[i] * x[i] for i in range(I)) >= R_min, name="min_return")
            if len(L_set) > 0:
                m.addConstr(gp.quicksum(x[i] for i in L_set) >= L_min, name="liquidity")

            m.optimize()

            if m.Status == GRB.OPTIMAL:
                opt_return = float(m.ObjVal)
                x_star = [x[i].X for i in range(I)]
                return opt_return, x_star
            else:
                return None, None
        except Exception:
            return None, None

    def generate_instances(self, output_path="/data1/LLMOptChall/LLMs-OPT/results/portfolio_3/portfolio_3_instances.jsonl", 
                                    save_x=False):
        index = 0
        with open(output_path, "w", encoding="utf-8") as fout:
            for I in range(self.I_range[0], self.I_range[1]):
                count = 0
                while count < self.samples_per_I:
                    try:
                        inst = self.generate_instance(index, I)
                    except RuntimeError:
                        continue

                    opt_val, x_star = self.solve_portfolio(inst)
                    if opt_val is not None:
                        inst["optimal_return"] = opt_val
                        if save_x and x_star is not None:
                            inst["x_opt"] = x_star
                        fout.write(json.dumps(inst) + "\n")
                        count += 1
                        index += 1
        print(f"Generating Complete: {index} valid Portfolio instances are saved at {output_path}")

    def make_template(self, input_path, output_path):

        self.input_path = input_path
        self.output_path = output_path

        self.template = (
            "An investor wishes to allocate capital among {I} asset classes with the goal of maximizing the total expected return of the portfolio.\n"
            "The characteristics of each asset are summarized as follows: {asset_lines} \n"
            "Each asset must receive a proportion of the total investment that satisfies its individual lower and upper bounds, "
            "and the total of all investment proportions must sum to one."
            "To ensure sufficient liquidity, the investor requires that the group of liquid assets, represented by the subset L ={{{L_assets}}}, "
            "collectively receive at least {L_min:.3f} of the total capital.\n"
            "At the same time, the overall portfolio risk, measured by a specified risk index, must not exceed {V_max:.3f},"
            "and the total expected return of the portfolio must be no less than {R_min:.3f}.\n"
            "Please determine the optimal portfolio weights that maximize total expected return subject to all constraints."
        )
    
    def _index_to_asset_label(self, i):
        return chr(ord('A') + i)

    def make_nl_example(self, data):
        I = data["I"]
        r = data["r"]
        v = data["v"]
        l = data["l"]
        u = data["u"]

        L_set = data["L_set"]
        L_assets = [self._index_to_asset_label(i) for i in L_set]


        asset_lines = "\n".join([
            f"* Asset {self._index_to_asset_label(i)}: expected return is {r[i]:.3f}, risk is {v[i]:.3f}, bounds [{l[i]:.3f}, {u[i]:.3f}]"
            for i in range(I)
        ])

        nl = self.template.format(
            I=I,
            asset_lines=asset_lines,
            L_assets=", ".join(L_assets),
            L_min=data["L_min"],
            V_max=data["V_max"],
            R_min=data["R_min"]
        )
        return nl

    def map_to_nl(self, input_path="/data1/LLMOptChall/LLMs-OPT/results/portfolio_3/portfolio_3_instances.jsonl", 
                  output_path="/data1/LLMOptChall/LLMs-OPT/results/portfolio_3/portfolio_3_nl.jsonl"):

        self.make_template(input_path=input_path, output_path=output_path)

        total = 0
        with open(self.input_path, "r", encoding="utf-8") as fin, \
            open(self.output_path, "w", encoding="utf-8") as fout:

            for line in fin:
                if not line.strip():
                    continue
                data = json.loads(line)

                nl = self.make_nl_example(data)

                out = {
                    "index": data["index"],
                    "problem_type": "Portfolio",
                    "problem_size": data["I"],
                    "nl_problem": nl
                }

                if "optimal_return" in data:
                    out["answer"] = data["optimal_return"]
                for k in ["is_feasible", "x_opt"]:
                    if k in data:
                        out[k] = data[k]

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                total += 1

        print(f"The natural language problem is generated, and a total of {total} instances are saved to {self.output_path}")
