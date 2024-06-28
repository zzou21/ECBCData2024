import matplotlib.pyplot as plt
import numpy as np
import statistics

# List of numbers
# numbers = [-3.740607500076294, -3.9700770378112793, -4.075359344482422, -3.6050376892089844, -3.8661997318267822, -3.9002931118011475, -3.7558939456939697, -4.256534576416016, -9.448721885681152, 3.0683906078338623]
numbers = [2.4974091053009033, 1.7714353799819946, 1.4192782640457153, 2.647120714187622, 2.35292387008667, 1.8375279903411865, 1.95516037940979, 2.654470920562744, -6.040181636810303, 5.419286727905273]

print(statistics.median(numbers))
print(statistics.mean(numbers))

# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(10, 2))

# Create a scatter plot on the number line
ax.scatter(numbers, [1] * len(numbers), color='blue', zorder=5)

# Set limits and labels
ax.set_ylim(0, 2)
ax.set_yticks([])  # Hide the y-axis

# Set grid and customize the appearance
# ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)

# Use numpy's arange to handle float values for xticks
ax.set_xticks(np.arange(min(numbers), max(numbers) + 1, 2))

# Draw a horizontal line at y=1
ax.axhline(y=1, color='black', linewidth=1)

# Display the plot
plt.title('File: A00151; Phase: Demo; Dataset: Truncated Demo Version')
plt.show()