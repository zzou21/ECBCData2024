# import json

# path = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/Bias Identification/CosSimWordClusterResult.json"
# list = []
# with open(path, "r") as file:
#     content = json.load(file)
# for k, v in content.items():
#     for k1, v2 in v.items():
#         if len(v2) > 1:
#             for i in v2:
#                 list.append((k1, i[0]))
#         else:
#             list.append((k1, v2[0][0]))

# for i in list:
#     print(i)

import numpy as np

# Define the matrix and vector
A = np.array([[2.04389, -0.101008],
              [8.88543, -0.523506],
              [17.0594, -0.846013]])
b = np.array([0.329491, 1.11633e-6, 2.7001])

# Solve the system of linear equations
solution = np.linalg.lstsq(A, b, rcond=None)[0]
a, b = solution

print(a, b)