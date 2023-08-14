import numpy as np
import matplotlib.pyplot as plt
from math import pi


angles_incidence_degrees = np.arange(10, 171, 10)  # [0,10,20...180]
# angles_incidence_degrees = [30,60]

inner_lens = -5
outer_lens = -10




angles_incidence_radians = np.radians(angles_incidence_degrees)

refractive_index_lens = 1.49 # acrylic
refractive_index_air = 1.000293 # air


# get gradients from angle
def formulate_equation(angle):
    return np.tan(np.radians(angle))


# get angles from gradients
def convert_gradient_to_angle(gradient):

    return np.degrees(np.arctan(gradient))



gradients_incidence = [formulate_equation(angle) for angle in angles_incidence_degrees]



intercepts_incidence = [inner_lens / gradient for gradient in gradients_incidence]

# pair gradients with intercepts
gradient_intercept_pairs_incidence = list(zip(gradients_incidence, intercepts_incidence))

# plot lines before refraction
plt.figure(figsize=(8, 6))
for i, pair in enumerate(gradient_intercept_pairs_incidence):
    
    # if pair[1] not null
    if(pair[1] >-10 and pair[1] < 10):
    
        if(pair[1] < 0):
            x_vals = np.linspace(pair[1], 0, 400)
            y_vals = pair[0] * x_vals
            
        elif(pair[1] > 0):
            x_vals = np.linspace(0, pair[1], 400)
            y_vals = pair[0] * x_vals
            
        plt.plot(x_vals, y_vals, color = 'blue')
        
    
    if((pair[1] <-10) or (pair[1] > 10)):
    
        if(pair[1] < 0):
            x_vals = np.linspace(-100, 100, 400)
            y_vals = pair[0] * x_vals
            
        elif(pair[1] > 0):
            x_vals = np.linspace(-100, 100, 400)
            y_vals = pair[0] * x_vals
            
        plt.plot(x_vals, y_vals, color = 'blue')
        
        

angle_of_refraction_radians = pi/2 - np.arcsin((refractive_index_air / refractive_index_lens) * np.cos( angles_incidence_radians))

angle_of_refraction_degrees = np.degrees(angle_of_refraction_radians)


gradients_refraction = [formulate_equation(angle) for angle in angle_of_refraction_degrees]


pair = list(zip(gradients_refraction, intercepts_incidence))


new_c_values = [(inner_lens - pairs[0] * pairs[1]) for pairs in pair]

intercepts_incidence_outer = [((outer_lens - new_c_value ) / gradient) for new_c_value, gradient in zip(new_c_values, gradients_refraction)]


gradient_intercept_pairs_refraction = list(zip(gradients_refraction, intercepts_incidence_outer, intercepts_incidence, new_c_values))


for i, pair in enumerate(gradient_intercept_pairs_refraction):
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
y_vals = inner_lens * np.ones(400)
plt.plot(x_vals, y_vals, color = 'red', label = 'inner part of lens')

# add horizontal line at y = -7
x_vals = np.linspace(-10, 10, 400)
y_vals = outer_lens * np.ones(400)
plt.plot(x_vals, y_vals, color = 'green', label = 'outer part of lens')




# plot points after outer lens
# need old equation, new equation of line, x val of intercept, y val of intercept (y = -10)


angle_of_exit_radians = pi/2 - np.arcsin((refractive_index_lens / refractive_index_air ) * np.cos(angle_of_refraction_radians))

angle_of_exit_degrees = np.degrees(angle_of_exit_radians)

gradients_refraction_exit = [formulate_equation(angle) for angle in angle_of_exit_degrees]

pair = list(zip(gradients_refraction_exit, intercepts_incidence_outer))

c_values_exit = [(outer_lens - pairs[0] * pairs[1]) for pairs in pair]

eq_3 = list(zip(gradients_refraction_exit, c_values_exit, intercepts_incidence_outer))

for i, pair in enumerate(eq_3):
    if(pair[1] >-10 and pair[1] < 10):

        if pair[2] < 0:
            x_vals = np.linspace(-1000, pair[2], 400)
            y_vals = pair[0] * x_vals + pair[1]
        elif pair[2] > 0:
            x_vals = np.linspace(pair[2], 1000, 400)
            y_vals = pair[0] * x_vals + pair[1]
            
        plt.plot(x_vals, y_vals, color = 'blue')

    






plt.xlim(-20, 20)
plt.ylim(-40, 0)


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
