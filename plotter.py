import numpy as np
import matplotlib.pyplot as plt
from math import pi
from math import inf
import copy 


max_distance = 1000 


def generate_angles(start, finish, step):
    
    angles_degrees = np.arange(start, finish + 1, step)
    
    angles_radians = np.radians(np.arange(start, finish + 1, step))
    
    return angles_degrees, angles_radians
    
    

def find_derivative(coefficients):
    a, b, c = coefficients
    derivative_a = 2 * a
    derivative_b = b
    derivative_c = c*0
    return [derivative_a, derivative_b, derivative_c]



def gradient_to_angle(gradient):
    
    angles_radians = np.arctan(gradient)
    
    return angles_radians


def angle_to_gradient(angle_in_degrees):
    
    gradients = np.tan(np.radians(angle_in_degrees))
    
    return gradients



def solve_quadratic(a, b, c):
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None
    elif discriminant == 0:
        return -b / (2 * a)
    else:
        return (-b + np.sqrt(discriminant)) / (2 * a), (-b - np.sqrt(discriminant)) / (2 * a)
    

# using intial angle, find the equations of the light rays and their point of intersection with the lens
# returns a list of ray equations in the form [m, c, start, finish, diffract] where m is gradient, c is y intercept, and start and finish are the x ranges and diffract is a boolean which is true if the ray diffracts and false if it does not
# lenses are in the form [a,b,c,d,e,f,g] where a,b,c,d,e are the coefficients of the equation ax^2 + bx + c = dy^2 + ey and f and g are the x ranges of the lens
def find_initial_ray_equations(angles_degrees, lens_equation):        
    
    ray_equations = []
    for angle in angles_degrees:
        ray_equations.append([angle_to_gradient(angle), 0, 0, 0, True])
        
    x_intercept = []

    if lens_equation[0] == 0 and lens_equation[1] == 0:
        x_intercept = [(lens_equation[2] - ray_equations[i][1]) / ray_equations[i][0] for i in range(len(ray_equations))]
    elif lens_equation[0] == 0 and lens_equation[1] != 0:
        x_intercept = [(lens_equation[2] - ray_equations[i][1]) / (ray_equations[i][0] - lens_equation[1]) for i in range(len(ray_equations))]
    elif lens_equation[0] != 0:
        for i in range(len(ray_equations)):
            first_solution, second_solution = solve_quadratic(lens_equation[0], lens_equation[1] - ray_equations[i][0], lens_equation[2] - ray_equations[i][1])
            abs_diff_first = abs(first_solution)
            abs_diff_second = abs(second_solution)
            if abs_diff_first < abs_diff_second:
                x_intercept.append(first_solution)
            else:
                x_intercept.append(second_solution)
            
    for i in range(len(ray_equations)):
        
        ray_equations[i][2] = min(x_intercept[i], 0)
        ray_equations[i][3] = max(x_intercept[i], 0)
        
        if(ray_equations[i][2] < lens_equation[5] or ray_equations[i][2] > lens_equation[6]):
            ray_equations[i][2] = -max_distance
            ray_equations[i][4] = False
            
        if(ray_equations[i][3] < lens_equation[5] or ray_equations[i][3] > lens_equation[6]):
            ray_equations[i][3] = max_distance
            ray_equations[i][4] = False
    
    ray_equations_true = []
    ray_equations_false = []
    for ray in ray_equations:
        if(ray[4] == True):
            ray_equations_true.append(ray)
        else:
            ray_equations_false.append(ray)
        

    return ray_equations, ray_equations_true, ray_equations_false

# keep all lines but leave limits untouched when not diffracting
    

    
# plots lens in red
def plot_lens(lens_equation):
    x_vals = np.linspace(lens_equation[5], lens_equation[6], 400)
    y_vals = lens_equation[0] * x_vals**2 + lens_equation[1] * x_vals + lens_equation[2]
    plt.plot(x_vals, y_vals, color = 'red')
    
# only plots rays that diffract
def plot_initial_rays(ray_equations):
    for i, equation in enumerate(ray_equations):
        # if(equation[4] == True):
            x_vals = np.linspace(equation[2], equation[3], 400)
            y_vals = equation[0] * x_vals + equation[1] 
            plt.plot(x_vals, y_vals, color = 'blue')




def find_middle_ray_equations(initial_ray_equations, inner_lens_equation, outer_lens_equation, refractive_index_lens, refractive_index_air):
    # [m, c, start, finish, diffract]
    # [a,b,c,d,e,f,g] where a,b,c,d,e are the coefficients of the equation ax^2 + bx + c = dy^2 + ey and f and g are the x ranges of the lens
    # new ray equations copy initial ray equations
    middle_ray_equations = copy.deepcopy(initial_ray_equations)
    initial_angles_in_radians = [gradient_to_angle(equation[0]) for equation in initial_ray_equations]
    for i, angle in enumerate(initial_angles_in_radians):
        if(angle < 0):
            initial_angles_in_radians[i] = pi + angle
    theta_3 = np.zeros(len(initial_angles_in_radians))
    
    
    for i, initial_ray_equation in enumerate(initial_ray_equations):
        # only for rays that diffract
        if(initial_ray_equation[4] == True):
            # find theta_3
            a,b,c = find_derivative(inner_lens_equation[0:3])
            
            if(abs(initial_ray_equation[2]) > abs(initial_ray_equation[3])):
                tangent_at_x = a * initial_ray_equation[2] + b
            else:
                tangent_at_x = a * initial_ray_equation[3] + b
            theta_3[i] = gradient_to_angle(tangent_at_x)
            
            # new gradient of ray in lens
            # new_ray_equations[i][0] = angle_to_gradient(pi/2 - (-theta_3[i] + np.arcsin((refractive_index_air / refractive_index_lens) * np.cos(- theta_3[i] - (initial_angles_in_radians[i])))))
            middle_ray_equations[i][0] = angle_to_gradient(np.degrees(pi/2 - (np.arcsin((refractive_index_air / refractive_index_lens) * np.sin(pi/2 - theta_3[i] - initial_angles_in_radians[i])))+ theta_3[i]))
            # add theta_3 to the equation to account for tangent



            # new y intercept of ray in lens
            # wrong
            if(abs(initial_ray_equation[2]) > abs(initial_ray_equation[3])):
                middle_ray_equations[i][1] = (initial_ray_equations[i][0] * initial_ray_equations[i][2] + initial_ray_equations[i][1]) - middle_ray_equations[i][0] * initial_ray_equations[i][2]
            else:
                middle_ray_equations[i][1] = (initial_ray_equations[i][0] * initial_ray_equation[3] + initial_ray_equations[i][1]) - middle_ray_equations[i][0] * initial_ray_equation[3]
                
                
                
                
                
                
            # new x intercept of ray in lens
            
            intercepts_outer_lens = ((outer_lens_equation[2] - middle_ray_equations[i][1] ) / middle_ray_equations[i][0]) 
            
            # find intercept of inner lens
            
            if(abs(initial_ray_equation[2]) > abs(initial_ray_equation[3])):
                intercepts_inner_lens = initial_ray_equation[2]
            else:
                intercepts_inner_lens = initial_ray_equation[3]
            
            # for rays that go left
            if(initial_ray_equation[0] > 0):
                if(intercepts_outer_lens < 0):
                    if(intercepts_outer_lens < outer_lens_equation[5] or intercepts_outer_lens > outer_lens_equation[6]):
                        middle_ray_equations[i][2] = -max_distance
                        middle_ray_equations[i][3] = intercepts_inner_lens
                        middle_ray_equations[i][4] = False
                    else:
                        middle_ray_equations[i][2] = intercepts_outer_lens
                        middle_ray_equations[i][3] = intercepts_inner_lens
                        
                else:
                    if(intercepts_outer_lens < outer_lens_equation[5] or intercepts_outer_lens > outer_lens_equation[6]):
                        middle_ray_equations[i][2] = -max_distance
                        middle_ray_equations[i][3] = intercepts_inner_lens
                        middle_ray_equations[i][4] = False
                    else:
                        middle_ray_equations[i][2] = intercepts_inner_lens
                        middle_ray_equations[i][3] = -max_distance
                
            else:
                if(intercepts_outer_lens > 0):
                    if(intercepts_outer_lens < outer_lens_equation[5] or intercepts_outer_lens > outer_lens_equation[6]):
                        middle_ray_equations[i][2] = max_distance
                        middle_ray_equations[i][3] = intercepts_inner_lens
                        middle_ray_equations[i][4] = False
                    else:
                        middle_ray_equations[i][2] = intercepts_outer_lens
                        middle_ray_equations[i][3] = intercepts_inner_lens
                        
                else:
                    if(intercepts_outer_lens < outer_lens_equation[5] or intercepts_outer_lens > outer_lens_equation[6]):
                        middle_ray_equations[i][2] = max_distance
                        middle_ray_equations[i][3] = intercepts_inner_lens
                        middle_ray_equations[i][4] = False
                    else:
                        middle_ray_equations[i][2] = intercepts_inner_lens
                        middle_ray_equations[i][3] = max_distance
                
            
            
            
            
            
            
            

    return middle_ray_equations
            
            
    







plt.figure(figsize=(8, 6))



equation_inner_lens = [1/2, 0, -3, 0, 1, -10, 10]
equation_outer_lens = [0, 0, -5, 0, 1, -15, 15]
plot_lens(equation_inner_lens)
initial_ray_equations = find_initial_ray_equations(generate_angles(10, 170, 10)[0], equation_inner_lens)[0]
# initial_ray_equations = find_initial_ray_equations(generate_angles(50, 50, 10)[0], equation_inner_lens)[0]

plot_initial_rays(initial_ray_equations)

plot_lens(equation_outer_lens)

middle_ray_equations = find_middle_ray_equations(initial_ray_equations, equation_inner_lens, equation_outer_lens, 1.5, 1)

plot_initial_rays(middle_ray_equations)




plt.xlim(-20, 20)
plt.ylim(-40, 0)
    

    
plt.xlabel('x')
plt.ylabel('y')
plt.title('Lines at Different Angles')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()

    
    
