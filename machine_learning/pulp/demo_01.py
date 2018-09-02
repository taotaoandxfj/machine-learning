import numpy as np
from pulp import *

paints = {"exterior", "interior", "theme"}

profit = {
    "exterior": 1000,
    "interior": 2000,
    "theme": 3000
}

M1 = {
    "exterior": 1,
    "interior": 2,
    "theme": 3
}

M2 = {
    "exterior": 0,
    "interior": 1,
    "theme": 2
}

prob = LpProblem("P", LpMaximize)  # 说明是求线性规划的最小值还是最大值

var = LpVariable.dict("paint", paints, 0, None, LpContinuous)  # x1,x2等的取值范围

prob += lpSum(profit[i] * var[i] for i in paints)  # 目标函数

# 约束条件
prob += lpSum(M1[i] * var[i] for i in paints) <= 10
prob += lpSum(M2[i] * var[i] for i in paints) <= 5

prob.writeLP("p.lp")

prob.solve()  # 模型求解
print("status:", LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)

print("maxValue", value(prob.objective))
