import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("ZhangFamilyTrip")

# Decision variables
x_H = model.addVar(vtype=GRB.BINARY, name="Harry")
x_He = model.addVar(vtype=GRB.BINARY, name="Hermione")
x_R = model.addVar(vtype=GRB.BINARY, name="Ron")
x_F = model.addVar(vtype=GRB.BINARY, name="Fred")
x_G = model.addVar(vtype=GRB.BINARY, name="George")
x_Gi = model.addVar(vtype=GRB.BINARY, name="Ginny")

# Objective function
model.setObjective(
    1200*x_H + 1650*x_He + 750*x_R + 800*x_F + 800*x_G + 1500*x_Gi,
    GRB.MINIMIZE
)

# Constraints
model.addConstr(x_H + x_He + x_R + x_F + x_G + x_Gi <= 4, "Max_4_children")
model.addConstr(x_H + x_He + x_R + x_F + x_G + x_Gi >= 3, "Min_3_children")
model.addConstr(x_Gi == 1, "Take_Ginny")
model.addConstr(x_H + x_F <= 1, "Harry_no_Fred")
model.addConstr(x_H + x_G <= 1, "Harry_no_George")
model.addConstr(x_F >= x_G, "George_implies_Fred")
model.addConstr(x_He >= x_G, "George_implies_Hermione")

# Solve
model.optimize()
print(f"Just print the best solution: {model.ObjVal}")

# Display results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for v in model.getVars():
        print(f"{v.VarName} = {v.X}")
    print(f"Total cost: ${model.ObjVal:.2f}")
else:
    print("No solution found")
