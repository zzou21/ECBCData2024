import json, csv

# solve the issue that not all files in the two CSVs are being read by JSON

list1 = set()
list2 = set()
with open("XMLProcessingAndTraining/StoringItalicsAndLineNumber.json", "r") as file:
    content = json.load(file)
    for key in content.keys():
        list1.add(key)
print(list1)
with open("/Users/Jerry/Desktop/EEBOClassificationsCSV/virginia_indians_new world_exploration_colony_colonization.csv", "r") as file:
    content = csv.reader(file)
    for row in content:
        list2.add(row[0])

with open("/Users/Jerry/Desktop/EEBOClassificationsCSV/colonization_colony_conversion.csv", "r") as file:
    content = csv.reader(file)
    for row in content:
        list2.add(row[0])
print(list2)