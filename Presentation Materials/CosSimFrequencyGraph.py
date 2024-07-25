import matplotlib.pyplot as plt

# Data
words = [
    "people", "labour", "knowledge", "religion", "nation", "men", "christ", "state",
    "instruction", "rude", "rich", "word", "virginia", "ease", "proceeding",
    "proceedings", "works", "riches", "plantation", "martyr", "ireland", "heathen",
    "misery", "worship", "peaceable", "lame", "labourers", "troubles", "sending", "nature"
]
values = [
    15, 7, 9, 9, 7, 5, 6, 5, 5, 5, 7, 6, 14, 6, 5, 14, 10, 5, 12, 7, 5, 5, 10, 6, 7, 5, 5, 6, 9, 5
]

# Combine words and values into a list of tuples
data = list(zip(words, values))

# Sort the list of tuples by the second element (the values)
data_sorted = sorted(data, key=lambda x: x[1])

# Unzip the sorted list of tuples into two lists: sorted_words and sorted_values
sorted_words, sorted_values = zip(*data_sorted)
# print(data)
for tuple in data:
    print(tuple[0], tuple[1])
# Create bar chart
# plt.figure(figsize=(10, 6))
# plt.bar(sorted_words, sorted_values, color='skyblue')
# plt.xlabel('Words')
# plt.ylabel('Values')
# plt.title('Words and Their Associated Numerical Values (Sorted)')
# plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
# plt.tight_layout()  # Adjust layout to prevent clipping of labels

# # Display the chart
# plt.show()
