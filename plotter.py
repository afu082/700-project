import matplotlib.pyplot as plt
import numpy as np

# Create the plot
plt.figure(figsize=(8, 6))

# Range of angles in degrees (from -90 to 90)
angles_deg = np.arange(-90, 91, 10)

# Convert angles to radians
angles_rad = np.radians(angles_deg)

# Loop through each angle and plot the lines
for angle_rad in angles_rad:
    x_vals = np.linspace(-10, 10, 400)
    y_vals = np.tan(angle_rad) * x_vals
    plt.plot(x_vals, y_vals)

# Set plot limits
plt.xlim(-10, 10)
plt.ylim(-20, 0)

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Lines at Different Angles')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend()

# Show the plot
plt.show()
