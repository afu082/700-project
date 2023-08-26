import numpy as np
import matplotlib.pyplot as plt
from math import pi


angles_before_incidence_degrees = np.arange(10, 171, 10)  # [0,10,20...180]
# angles_incidence_degrees = [60]


# lens equations y = -5 
inner_lens = -5
outer_lens = -10

# NEW represent lens as equations 
eq_inner = [0, 0, -5]
eq_outer = [0, 0, -10]



# refractive indices
refractive_index_lens = 1.49 
refractive_index_air = 1.000293 # air


# get gradients from angle
def angle_to_gradient(angle):
    return np.tan(np.radians(angle))

# array of gradients (b)
gradients_incidence = [angle_to_gradient(angle) for angle in angles_before_incidence_degrees]

# array of intercepts (c)
intercepts_inner_lens = [inner_lens / gradient for gradient in gradients_incidence]

# NEW new equations in form y = ax^2 + bx + c [a,b,c]
equations_before_lens = [(0, gradient, intercept) for gradient, intercept in zip(gradients_incidence, intercepts_inner_lens)]

# plot lines before refraction
plt.figure(figsize=(8, 6))
for equation in equations_before_lens:
    
    # if pair[1] not null
    if(equation[2] >-10 and equation[2] < 10):
    
        if(equation[2] < 0):
            x_vals = np.linspace(equation[2], 0, 400)
            y_vals = equation[0] * x_vals**2 + equation[1] * x_vals
            
        elif(equation[2] > 0):
            x_vals = np.linspace(0, equation[2], 400)
            y_vals = equation[0] * x_vals**2 + equation[1] * x_vals
            
        plt.plot(x_vals, y_vals, color = 'blue')
        
    
    if((equation[2] <-10) or (equation[2] > 10)):
    
        if(equation[2] < 0):
            x_vals = np.linspace(-100, 100, 400)
            y_vals = equation[0] * x_vals**2 + equation[1] * x_vals
            
        elif(equation[2] > 0):
            x_vals = np.linspace(-100, 100, 400)
            y_vals = equation[0] * x_vals**2 + equation[1] * x_vals
            
        plt.plot(x_vals, y_vals, color = 'blue')
        
        
# angles of light in lens [10,20,30,...]
angles_in_lens_degrees = np.degrees(pi/2 - np.arcsin((refractive_index_air / refractive_index_lens) * np.cos( np.radians(angles_before_incidence_degrees))))

# gradients of light in lens [a,b,c,...]
gradients_in_lens = [angle_to_gradient(angle) for angle in angles_in_lens_degrees]

c_in_lens = [(inner_lens - pairs[0] * pairs[1]) for pairs in zip(gradients_in_lens, intercepts_inner_lens)]

equations_in_lens = [(0, gradients_in_lens, c_in_lens) for gradients_in_lens, c_in_lens in zip(gradients_in_lens, c_in_lens)]

intercepts_outer_lens = [((outer_lens - new_c_value ) / gradient) for new_c_value, gradient in zip(c_in_lens, gradients_in_lens)]

for vals in zip(gradients_in_lens, intercepts_outer_lens, intercepts_inner_lens, c_in_lens):
    if vals[1] > -10 and vals[1] < 10:
        if vals[1] < 0:
            x_vals = np.linspace(vals[1], vals[2], 400)
            y_vals = vals[0] * x_vals + vals[3]
        elif vals[1] > 0:
            x_vals = np.linspace(vals[2], vals[1], 400)
            y_vals = vals[0] * x_vals + vals[3]
            
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


angle_of_exit_degrees = np.degrees( pi/2 - np.arcsin((refractive_index_lens / refractive_index_air ) * np.cos(np.radians(angles_in_lens_degrees))))

gradients_after_lens = [angle_to_gradient(angle) for angle in angle_of_exit_degrees]

gradient_intercept_with_lens_pairs = list(zip(gradients_after_lens, intercepts_outer_lens))

c_values_exit = [(outer_lens - pairs[0] * pairs[1]) for pairs in gradient_intercept_with_lens_pairs]

equations_exit = [(0, gradients_after_lens, c_values_exit) for gradients_after_lens, c_values_exit in zip(gradients_after_lens, c_values_exit)]


for vals in zip(gradients_after_lens, c_values_exit, intercepts_outer_lens):
    if(vals[1] >-10 and vals[1] < 10):

        if vals[2] < 0:
            x_vals = np.linspace(-1000, vals[2], 400)
            y_vals = vals[0] * x_vals + vals[1]
        elif vals[2] > 0:
            x_vals = np.linspace(vals[2], 1000, 400)
            y_vals = vals[0] * x_vals + vals[1]
            
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
