import numpy as np
import matplotlib.pyplot as plt
from math import pi


angles_incidence_degrees = np.arange(0, 181, 10)  # [0,10,20...180]



angles_incidence_radians = np.radians(angles_incidence_degrees)
print(angles_incidence_radians)

refractive_index_lens = 1.49 # acrylic
refractive_index_air = 1.000293 # air


# get gradients from angle
def formulate_equation(angle):
    print(angle)
    print(np.tan(np.radians(angle)))
    return np.tan(np.radians(angle))


# get angles from gradients
def convert_gradient_to_angle(gradient):
    print(gradient)
    print(np.degrees(np.arctan(gradient)))
    return np.degrees(np.arctan(gradient))



gradients_incidence = [formulate_equation(angle) for angle in angles_incidence_degrees]

inner_lens = -5
outer_lens = -10

intercepts_incidence = [inner_lens / gradient for gradient in gradients_incidence]

# pair gradients with intercepts
gradient_intercept_pairs_incidence = list(zip(gradients_incidence, intercepts_incidence))

# plot lines before refraction
plt.figure(figsize=(8, 6))
for i, pair in enumerate(gradient_intercept_pairs_incidence):
    
    # if pair[1] not null
    if(pair[1] >-10 and pair[1] < 10):
        # print(pair[0], pair[1])
    
        if(pair[1] < 0):
            x_vals = np.linspace(pair[1], 0, 400)
            y_vals = pair[0] * x_vals
            
        elif(pair[1] > 0):
            x_vals = np.linspace(0, pair[1], 400)
            y_vals = pair[0] * x_vals
            
        plt.plot(x_vals, y_vals, color = 'blue')
        
    
    if((pair[1] <-10) or (pair[1] > 10)):
        # print(pair[0], pair[1])
    
        if(pair[1] < 0):
            x_vals = np.linspace(-100, 100, 400)
            y_vals = pair[0] * x_vals
            
        elif(pair[1] > 0):
            x_vals = np.linspace(-100, 100, 400)
            y_vals = pair[0] * x_vals
            
        plt.plot(x_vals, y_vals, color = 'blue')
        
        

# angle_of_refraction_radians = np.arcsin((refractive_index_air / refractive_index_lens) * np.sin(((pi/2) - angles_incidence_radians)) if angles_incidence_radians < pi/2 else 0)
angle_of_refraction_radians = pi/2 - np.arcsin((refractive_index_air / refractive_index_lens) * np.sin(pi/2 - angles_incidence_radians))

angle_of_refraction_degrees = np.degrees(angle_of_refraction_radians)

print(angle_of_refraction_degrees)

gradients_refraction = [formulate_equation(angle) for angle in angle_of_refraction_degrees]
print(gradients_refraction)

pair = list(zip(gradients_refraction, intercepts_incidence))


new_c_values = [(inner_lens - pairs[0] * pairs[1]) for pairs in pair]



intercepts_incidence_outer = [((outer_lens - new_c_value ) / gradient) for new_c_value, gradient in zip(new_c_values, gradients_refraction)]
print(intercepts_incidence_outer)

gradient_intercept_pairs_refraction = list(zip(gradients_refraction, intercepts_incidence_outer, intercepts_incidence, new_c_values))


for i, pair in enumerate(gradient_intercept_pairs_refraction):
    print(pair[1])
    if pair[1] > -10 and pair[1] < 10:
        if pair[1] < 0:
            x_vals = np.linspace(pair[1], pair[2], 400)
            y_vals = pair[0] * x_vals + pair[3]
        elif pair[1] > 0:
            x_vals = np.linspace(pair[2], pair[1], 400)
            y_vals = pair[0] * x_vals + pair[3]
            
        plt.plot(x_vals, y_vals, color = 'blue')


        

# add horizontal line at y = -5
x_vals = np.linspace(-10, 10, 400)
y_vals = -5 * np.ones(400)
plt.plot(x_vals, y_vals, color = 'red', label = 'y = -5')

# add horizontal line at y = -7
x_vals = np.linspace(-10, 10, 400)
y_vals = -10 * np.ones(400)
plt.plot(x_vals, y_vals, color = 'green', label = 'y = -10')
















plt.xlim(-20, 20)
plt.ylim(-40, 10)


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
