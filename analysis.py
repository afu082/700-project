import matplotlib.pyplot as plt
import numpy as np
import plotter_copy




# plotter_copy.equation_inner_lens = [new_values_here]
# plotter_copy.equation_outer_lens = [new_values_here]
plotter_copy.refractive_index_air = 1
plotter_copy.refractive_index_lens = 2
# plotter_copy.initial_candelas_values = new_value_here
# plotter_copy.initial_angles = new_value_here





# Access the maps
initial_angle_candela_map = plotter_copy.initial_angle_candela_map
exit_angle_candela_map = plotter_copy.exit_angle_candela_map

# Filter out zero candela values
initial_angles, initial_candelas = zip(*[(angle, candela) for angle, candela in initial_angle_candela_map.items() if candela != 0])
exit_angles, exit_candelas = zip(*[(angle, candela) for angle, candela in exit_angle_candela_map.items() if candela != 0])


# Create bins for grouping angles
bins = range(0, 91, 5)  # 0-10, 10-20, ...,
bin_labels = [f'{start}-{start+4}' for start in bins[:-1]]

print(bin_labels)

# Group angles into 10-degree bins
initial_angles_binned = np.digitize(initial_angles, bins, right=False)
exit_angles_binned = np.digitize(exit_angles, bins, right=False)

# Print initial_angles_binned and exit_angles_binned
print(f"Initial Angles Binned: {initial_angles_binned}")
print(f"Exit Angles Binned: {exit_angles_binned}")

# Create dictionaries for 10-degree sections
initial_section_candelas = {label: [] for label in bin_labels}
exit_section_candelas = {label: [] for label in bin_labels}

# Print initial_section_candelas and exit_section_candelas
print(f"Initial Section Candelas: {initial_section_candelas}")
print(f"Exit Section Candelas: {exit_section_candelas}")

# Populate dictionaries with corresponding candelas
for angle, candela in zip(initial_angles_binned, initial_candelas):
    label = bin_labels[angle - 1]  # -1 to adjust for 0-based indexing
    initial_section_candelas[label].append(candela)

for angle, candela in zip(exit_angles_binned, exit_candelas):
    label = bin_labels[angle - 1]
    exit_section_candelas[label].append(candela)

# Print the populated dictionaries
print(f"Initial Section Candelas After Population: {initial_section_candelas}")
print(f"Exit Section Candelas After Population: {exit_section_candelas}")







# Plot the initial angle-candela map
plt.figure(figsize=(10, 5))
plt.scatter(initial_angles, initial_candelas, color='blue')
plt.title("Initial Angle-Candela Map")
plt.xlabel("Angles")
plt.ylabel("Candelas")
plt.grid(True)
plt.show()

# Plot the exit angle-candela map
plt.figure(figsize=(10, 5))
plt.scatter(exit_angles, exit_candelas, color='red')
plt.title("Exit Angle-Candela Map")
plt.xlabel("Angles")
plt.ylabel("Candelas")
plt.grid(True)
plt.show()

# Calculate statistics for initial angle-candela map
initial_candelas_np = np.array(initial_candelas)
initial_std_dev = np.std(initial_candelas_np)
initial_variance = np.var(initial_candelas_np)

# Calculate statistics for exit angle-candela map
exit_candelas_np = np.array(exit_candelas)
exit_std_dev = np.std(exit_candelas_np)
exit_variance = np.var(exit_candelas_np)

print(f"Statistics for Initial Angle-Candela Map:")
print(f"Standard Deviation: {initial_std_dev}")
print(f"Variance: {initial_variance}")

print(f"\nStatistics for Exit Angle-Candela Map:")
print(f"Standard Deviation: {exit_std_dev}")
print(f"Variance: {exit_variance}")
