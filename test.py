import json

count = 0
with open ("/Users/Jerry/Desktop/Data+2024/Data+2024Code/ECBCData2024/ManuscriptMetadata/DocumentMetadata.json", "r") as file:
    content = json.load(file)
    for k, v in content.items():
        for ke, va in v.items():
            count += 1
print(count)