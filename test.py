import json

path = "/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/Bias Identification/CosSimWordClusterResult.json"
list = []
with open(path, "r") as file:
    content = json.load(file)
for k, v in content.items():
    for k1, v2 in v.items():
        if len(v2) > 1:
            for i in v2:
                list.append((k1, i[0]))
        else:
            list.append((k1, v2[0][0]))

for i in list:
    print(i)