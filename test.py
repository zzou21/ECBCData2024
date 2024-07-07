import json
with open("/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/FindCosineSimilarityWords/tesetOptimizer.json", "r") as file:
    te = json.load(file)
for k, v in te.items():
    for k1, v1 in v.items():
        print(len(v1))

# import heapq

# # Initialize an empty heap
# top_20_heap = []

# # Function to add a tuple to the heap and maintain the top 20
# def add_tuple_to_top_20(tup):
#     if len(top_20_heap) < 2:
#         # If the heap has fewer than 20 elements, push the new tuple
#         heapq.heappush(top_20_heap, (tup[1], tup))
#     else:
#         # If the heap has 20 elements, push the new tuple and pop the smallest
#         heapq.heappushpop(top_20_heap, (tup[1], tup))

# # Example of appending tuples to the list and maintaining the top 20
# tuples = [
#     (1, 3, 5, 7),
#     (2, 2, 6, 8),
#     (3, 5, 1, 0),
#     (4, 1, 4, 9),
#     # Add more tuples as needed
# ]

# # Append tuples one by one
# for tup in tuples:
#     add_tuple_to_top_20(tup)

# # Extract the top 20 tuples from the heap
# top_20 = [tup for _, tup in sorted(top_20_heap, reverse=True)]

# # Print the top 20 tuples
# print(top_20)




# # # import json

# # # path = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/Bias Identification/CosSimWordClusterResult.json"
# # # list = []
# # # with open(path, "r") as file:
# # #     content = json.load(file)
# # # for k, v in content.items():
# # #     for k1, v2 in v.items():
# # #         if len(v2) > 1:
# # #             for i in v2:
# # #                 list.append((k1, i[0]))
# # #         else:
# # #             list.append((k1, v2[0][0]))

# # # for i in list:
# # #     print(i)

# # import numpy as np

# # # Define the matrix and vector
# # A = np.array([[2.04389, -0.101008],
# #               [8.88543, -0.523506],
# #               [17.0594, -0.846013]])
# # b = np.array([0.329491, 1.11633e-6, 2.7001])

# # # Solve the system of linear equations
# # solution = np.linalg.lstsq(A, b, rcond=None)[0]
# # a, b = solution

# # print(a, b)