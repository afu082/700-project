import numpy as np
import matplotlib.pyplot as plt
from math import pi
from math import inf


def derivative_coefficients(coefficients):
    a, b, c = coefficients
    derivative_a = 2 * a
    derivative_b = b
    derivative_c = c*0
    return [derivative_a, derivative_b, derivative_c]

# get gradients from angle
def angle_to_gradient(angle):
    return np.tan(np.radians(angle))

def gradient_to_angle(gradient):
    return np.degrees(np.arctan(gradient))

# solve quadratic equation
def solve_quadratic(a, b, c):
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None
    elif discriminant == 0:
        return -b / (2 * a)
    else:
        return (-b + np.sqrt(discriminant)) / (2 * a), (-b - np.sqrt(discriminant)) / (2 * a)
    
    
# # Example coefficients: [a, b, c, d, e]
# coefficients = [1, 0, -25, 1, 0,-10,10]  # Adjust these coefficients as needed

# A, B, C, D, E, START, END = coefficients

# # Generate x and y values
# x = np.linspace(START, END, 400)
# y = np.linspace(-20, 20, 400)
# X, Y = np.meshgrid(x, y)

# # Evaluate the ellipse equation for each (x, y) pair
# Z = A * X**2 + B * X + C + D * Y**2 - E * Y 

# # Create the plot
# plt.figure(figsize=(6, 6))
# plt.xlim(-20, 20)
# plt.ylim(-40, 0)
# plt.contour(X, Y, Z, levels=[0], colors='b')  # Plot the ellipse where Z = 0
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Plot of an Ellipse')
# plt.grid()
# plt.axis('equal')



    


angles_before_incidence_degrees = np.arange(10, 171, 10)  # [0,10,20...180]
# angles_incidence_degrees = [60]



# represent lens as equations  y = ax^2 + bx + c 
# [a,b,c, x_start, x_end]
eq_lens_inner = [1/10, 0, -3, -5.5, 5.5]

eq_lens_outer = [0, 0, -10, -10, 10]



# refractive indices
refractive_index_lens = 1.49 
refractive_index_air = 1.000293 # air


# array of gradients (b)
gradients_incidence = [angle_to_gradient(angle) for angle in angles_before_incidence_degrees]

# x value of where beam intercepts inner lens
if eq_lens_inner[0] == 0 and eq_lens_inner[1] == 0:
    x_intercepts_of_inner_lens = [eq_lens_inner[2] / gradient for gradient in gradients_incidence] 
elif eq_lens_inner[0] == 0 and eq_lens_inner[1] != 0:
    x_intercepts_of_inner_lens = [eq_lens_inner[2] / (gradient - eq_lens_inner[1]) for gradient in gradients_incidence]
    # for i, intercepts in enumerate(x_intercepts_of_inner_lens):
    #     if eq_lens_inner[1] * intercepts + eq_lens_inner[2] > 0:
    #         x_intercepts_of_inner_lens[i] = None
elif eq_lens_inner[0] != 0:
    intercepts = []
    x_intercepts_of_inner_lens = []
    for gradient in gradients_incidence:
        intercepts.append(solve_quadratic(eq_lens_inner[0], eq_lens_inner[1] - gradient, eq_lens_inner[2]))
    for i, intercept in enumerate(intercepts):
        if(gradients_incidence[i] < 0):
            x_intercepts_of_inner_lens.append(intercept[0])
        elif(gradients_incidence[i] > 0):
            x_intercepts_of_inner_lens.append(intercept[1])
        elif(gradients_incidence[i] == 0):
            x_intercepts_of_inner_lens.append(intercept[0])
            

        

# gradient intercept pairs
gradient_range_values = [(0, gradients_incidence, x_intercepts_of_inner_lens) for gradients_incidence, x_intercepts_of_inner_lens in zip(gradients_incidence, x_intercepts_of_inner_lens)]

# equations of lines before lens in form y = ax^2 + bx + c [a,b,c, x_start, x_end, valid]
equations_before_lens = [
    [0, gradients_incidence, 0, min(x_intercepts_of_inner_lens[i], 0), max(x_intercepts_of_inner_lens[i], 0), True]
    for i, gradients_incidence in enumerate(gradients_incidence)
]

# plot lines before refraction
plt.figure(figsize=(8, 6))

for i, equation in enumerate(equations_before_lens):
    #vert not plotting
    if(equation[3] > eq_lens_inner[3] and equation[4] < eq_lens_inner[4]):
        x_vals = np.linspace(equation[3], equation[4], 400)
        y_vals = equation[0] * x_vals**2 + equation[1] * x_vals + equation[2]

        plt.plot(x_vals, y_vals, color = 'blue')
        
    else:
        if(equation[3] < 0):
            x_vals = np.linspace(-100, 0, 400)
            y_vals = equation[0] * x_vals**2 + equation[1] * x_vals + equation[2]
            
        elif(equation[4] > 0):
            x_vals = np.linspace(0, 100, 400)
            y_vals = equation[0] * x_vals**2 + equation[1] * x_vals + equation[2]
        equations_before_lens[i][5] = False
        plt.plot(x_vals, y_vals, color = 'red')








# get angles of lines from gradients in equations_before_lens if valid
valid_angles_in_lens_degrees = []
for i, equation in enumerate(equations_before_lens):
    if(equation[5] == True):
        valid_angles_in_lens_degrees.append(gradient_to_angle(equation[1]))
        
        
        


# angles of light in lens [10,20,30,...]
angles_in_lens_degrees = np.degrees(pi/2 - np.arcsin((refractive_index_air / refractive_index_lens) * np.cos(np.radians(angles_before_incidence_degrees))))

# gradients of light in lens [a,b,c,...]
gradients_in_lens = [angle_to_gradient(angle) for angle in angles_in_lens_degrees]

c_in_lens = [(eq_lens_inner[2] - pairs[0] * pairs[1]) for pairs in zip(gradients_in_lens, x_intercepts_of_inner_lens)]

equations_in_lens = [(0, gradients_in_lens, c_in_lens) for gradients_in_lens, c_in_lens in zip(gradients_in_lens, c_in_lens)]

intercepts_outer_lens = [((eq_lens_outer[2] - new_c_value ) / gradient) for new_c_value, gradient in zip(c_in_lens, gradients_in_lens)]

for vals in zip(gradients_in_lens, intercepts_outer_lens, x_intercepts_of_inner_lens, c_in_lens):
    # TODO find limits based on quadratic equation
    if vals[1] > -10 and vals[1] < 10:
        if vals[1] < 0:
            x_vals = np.linspace(vals[1], vals[2], 400)
            y_vals = vals[0] * x_vals + vals[3]
        elif vals[1] > 0:
            x_vals = np.linspace(vals[2], vals[1], 400)
            y_vals = vals[0] * x_vals + vals[3]
            
        plt.plot(x_vals, y_vals, color = 'blue')


        

# plot inner lens
x_vals = np.linspace(eq_lens_inner[3], eq_lens_inner[4], 400)
y_vals = eq_lens_inner[0] * x_vals**2 + eq_lens_inner[1] * x_vals + eq_lens_inner[2]
plt.plot(x_vals, y_vals, color = 'red', label = 'inner part of lens')

# plot outer lens
x_vals = np.linspace(-10, 10, 400)
y_vals = eq_lens_outer[2] * np.ones(400)
plt.plot(x_vals, y_vals, color = 'green', label = 'outer part of lens')




# plot points after outer lens
# need old equation, new equation of line, x val of intercept, y val of intercept (y = -10)


angle_of_exit_degrees = np.degrees( pi/2 - np.arcsin((refractive_index_lens / refractive_index_air ) * np.cos(np.radians(angles_in_lens_degrees))))

gradients_after_lens = [angle_to_gradient(angle) for angle in angle_of_exit_degrees]

gradient_intercept_with_lens_pairs = list(zip(gradients_after_lens, intercepts_outer_lens))

c_values_exit = [(eq_lens_outer[2] - pairs[0] * pairs[1]) for pairs in gradient_intercept_with_lens_pairs]

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
