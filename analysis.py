import math
import plotter_for_analysis
import matplotlib.pyplot as plt
import numpy as np




#Import plotter_for_analysis and run the convert function
refractive_index_air = 1
refractive_index_lens = 1.5
# lenses are in the form [a,b,c,d,e,f,g] where a,b,c,d,e are the coefficients of the equation ax^2 + bx + c = dy^2 + ey and f and g are the x ranges of the lens
equation_inner_lens = [1/50, 0, -0.25, 0, 1, -10, 10]
equation_outer_lens =  [0, 0, -1, 0, 1, -10, 10]
            

plotter_for_analysis.convert(refractive_index_air, refractive_index_lens, equation_inner_lens, equation_outer_lens)

def changing_refractive_index():
        for refractive_index_lens in [1.5, 2]:
            
            print(f"Running with refractive index lens = {refractive_index_lens}")
            print("Initial Angles:", initial_angles)
            print("Initial Candela Values:", initial_candela_values)
            plotter_for_analysis.convert(refractive_index_air, refractive_index_lens, equation_inner_lens, equation_outer_lens)

            initial_angle_candela_map = plotter_for_analysis.initial_angle_candela_map
            exit_angle_candela_map = plotter_for_analysis.exit_angle_candela_map

            print("Initial Angle Candela Map:", initial_angle_candela_map)
            print("Exit Angle Candela Map:", exit_angle_candela_map)
            print("="*30)



def extract_candelas_by_angle(angle_candela_map, bins):
    # Filter out zero candela values
    angles, candelas = zip(*[(angle, candela) for angle, candela in angle_candela_map.items() if candela != 0])

    # Group angles into specified bins
    angles_binned = np.digitize(angles, bins, right=False)

    # Create a dictionary for bins
    bin_labels = [f'{start}-{start+4}' for start in bins[:-1]]
    section_candelas = {label: [] for label in bin_labels}

    # Populate the dictionary with corresponding candelas
    for angle, candela in zip(angles_binned, candelas):
        label = bin_labels[angle - 1]  # -1 to adjust for 0-based indexing
        section_candelas[label].append(candela)

    return section_candelas


initial_angle_candela_map = plotter_for_analysis.initial_angle_candela_map
exit_angle_candela_map = plotter_for_analysis.exit_angle_candela_map

print(initial_angle_candela_map)
bins = range(0, 91, 5) 
initial_section_candelas = extract_candelas_by_angle(initial_angle_candela_map, bins)



print(f"Initial Section Candelas: {initial_section_candelas}")

def calculate_midpoints(bins):
    midpoints = [(start + end) / 2 for start, end in zip(bins[:-1], bins[1:])]
    return midpoints

def calculate_average_candelas(section_candelas):
    average_candelas = {label: sum(candelas) / len(candelas) if candelas else 0 for label, candelas in section_candelas.items()}
    return average_candelas



midpoints = calculate_midpoints(bins)
average_initial_candelas = calculate_average_candelas(initial_section_candelas)


# Combine midpoints and average candelas into a map
midpoint_initial_candela_map = dict(zip(midpoints, average_initial_candelas.values()))

# print("Midpoints:", midpoints)
# print("Average Initial Candelas:", average_initial_candelas)
# print("Midpoint Candela Map:", midpoint_initial_candela_map)

exit_section_candelas = extract_candelas_by_angle(exit_angle_candela_map, bins)

print(f"Exit Section Candelas: {exit_section_candelas}")

midpoints_exit = calculate_midpoints(bins)
average_exit_candelas = calculate_average_candelas(exit_section_candelas)

midpoint_exit_candela_map = dict(zip(midpoints_exit, average_exit_candelas.values()))




# Plot initial angle-candela map from initial_angle_candela_map
plt.figure(figsize=(10, 5))
plt.scatter(initial_angle_candela_map.keys(), initial_angle_candela_map.values(), color='blue')
plt.title("Initial Angles-Candela Map")
plt.xlabel("Angles in degrees°")
plt.ylabel("Relative Luminous Intensity")
plt.grid(True)
plt.show()

# Plot exit angle-candela map from exit_angle_candela_map
plt.figure(figsize=(10, 5))
plt.scatter(exit_angle_candela_map.keys(), exit_angle_candela_map.values(), color='red')
plt.title("Exit Angles-Candela Map")
plt.xlabel("Angles in degrees°")
plt.ylabel("Relative Luminous Intensity")
plt.grid(True)
plt.show()

# Calculate statistics for initial angle-candela map from initial_angle_candela_map
initial_candelas_np = np.array(list(initial_angle_candela_map.values()))
initial_std_dev = np.std(initial_candelas_np)
initial_variance = np.var(initial_candelas_np)

# Calculate statistics for exit angle-candela map from exit_angle_candela_map
exit_candelas_np = np.array(list(exit_angle_candela_map.values()))
exit_std_dev = np.std(exit_candelas_np)
exit_variance = np.var(exit_candelas_np)

print(f"Statistics for Initial Angle-Candela Map:")
print(f"Standard Deviation: {initial_std_dev}")
print(f"Variance: {initial_variance}")

print(f"\nStatistics for Exit Angle-Candela Map:")
print(f"Standard Deviation: {exit_std_dev}")
print(f"Variance: {exit_variance}")


# Calculate statistics for initial angle-candela map (values < 46)
initial_candelas_np_less_46 = np.array([value for key, value in initial_angle_candela_map.items() if key < 46])
initial_std_dev_less_46 = np.std(initial_candelas_np_less_46)
initial_variance_less_46 = np.var(initial_candelas_np_less_46)

# Calculate statistics for exit angle-candela map (values < 46)
exit_candelas_np_less_46 = np.array([value for key, value in exit_angle_candela_map.items() if key < 46])
exit_std_dev_less_46 = np.std(exit_candelas_np_less_46)
exit_variance_less_46 = np.var(exit_candelas_np_less_46)

print(f"\nStatistics for Initial Angle-Candela Map (Values less than 46):")
print(f"Standard Deviation: {initial_std_dev_less_46}")
print(f"Variance: {initial_variance_less_46}")

print(f"\nStatistics for Exit Angle-Candela Map (Values less than 46):")
print(f"Standard Deviation: {exit_std_dev_less_46}")
print(f"Variance: {exit_variance_less_46}")



# Plot midpoint initial candela map from midpoint_initial_candela_map
plt.figure(figsize=(10, 5))
plt.scatter(midpoint_initial_candela_map.keys(), midpoint_initial_candela_map.values(), color='green')
plt.title("Midpoint Initial Candela Map")
plt.xlabel("Angles in degrees°")
plt.ylabel("Relative Luminous Intensity")
plt.grid(True)
plt.show()

# Plot midpoint exit candela map from midpoint_exit_candela_map
plt.figure(figsize=(10, 5))
plt.scatter(midpoint_exit_candela_map.keys(), midpoint_exit_candela_map.values(), color='purple')
plt.title("Midpoint Exit Candela Map")
plt.xlabel("Angles in degrees°")
plt.ylabel("Relative Luminous Intensity")
plt.grid(True)
plt.show()

# Calculate statistics for midpoint initial candela map from midpoint_initial_candela_map
midpoint_initial_candelas_np = np.array(list(midpoint_initial_candela_map.values()))
midpoint_initial_std_dev = np.std(midpoint_initial_candelas_np)
midpoint_initial_variance = np.var(midpoint_initial_candelas_np)

# Calculate statistics for midpoint exit candela map from midpoint_exit_candela_map
midpoint_exit_candelas_np = np.array(list(midpoint_exit_candela_map.values()))
midpoint_exit_std_dev = np.std(midpoint_exit_candelas_np)
midpoint_exit_variance = np.var(midpoint_exit_candelas_np)

print(f"\nStatistics for Midpoint Initial Candela Map:")
print(f"Standard Deviation: {midpoint_initial_std_dev}")
print(f"Variance: {midpoint_initial_variance}")

print(f"\nStatistics for Midpoint Exit Candela Map:")
print(f"Standard Deviation: {midpoint_exit_std_dev}")
print(f"Variance: {midpoint_exit_variance}")


# Plot midpoint initial candela map for first 45 angles
plt.figure(figsize=(10, 5))
plt.scatter(midpoint_initial_candela_map.keys(), midpoint_initial_candela_map.values(), color='green')
plt.title("Midpoint Initial Candela Map (First 45 Angles)")
plt.xlabel("Angles in degrees°")
plt.ylabel("Relative Luminous Intensity")
plt.xlim(0, 45)  # Limit x-axis to first 45 angles
plt.grid(True)
plt.show()

# Plot midpoint exit candela map for first 45 angles
plt.figure(figsize=(10, 5))
plt.scatter(midpoint_exit_candela_map.keys(), midpoint_exit_candela_map.values(), color='purple')
plt.title("Midpoint Exit Candela Map (First 45 Angles)")
plt.xlabel("Angles in degrees°")
plt.ylabel("Relative Luminous Intensity")
plt.xlim(0, 45)  # Limit x-axis to first 45 angles
plt.grid(True)
plt.show()

# Calculate statistics for midpoint initial candela map (first 45 angles)
midpoint_initial_candelas_np = np.array([value for key, value in midpoint_initial_candela_map.items() if key < 46])
midpoint_initial_std_dev = np.std(midpoint_initial_candelas_np)
midpoint_initial_variance = np.var(midpoint_initial_candelas_np)

# Calculate statistics for midpoint exit candela map (first 45 angles)
midpoint_exit_candelas_np = np.array([value for key, value in midpoint_exit_candela_map.items() if key < 46])
midpoint_exit_std_dev = np.std(midpoint_exit_candelas_np)
midpoint_exit_variance = np.var(midpoint_exit_candelas_np)

print(f"\nStatistics for Midpoint Initial Candela Map (Angles less than 46 in 5s):")
print(f"Standard Deviation: {midpoint_initial_std_dev}")
print(f"Variance: {midpoint_initial_variance}")

print(f"\nStatistics for Midpoint Exit Candela Map (Angles less than 46 in 5s):")
print(f"Standard Deviation: {midpoint_exit_std_dev}")
print(f"Variance: {midpoint_exit_variance}")