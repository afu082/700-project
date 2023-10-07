import numpy as np
import matplotlib.pyplot as plt
from math import pi
from math import inf
import copy 
import os

max_distance = 1000 


            



def print_degrees_of_ray_eqns(eqns):
    
    print([np.degrees(gradient_to_angle(ray[0])) for ray in eqns])


#generate_angles(10, 170, 1)[0]
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


#equation_inner_lens = [1/10, 0, -3, 0, 1, -10, 10]
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
        # angles neg y neg x (bottom left square) = positive grad, bottom right square = negative
        ray_equations.append([angle_to_gradient(angle), 0, 0, 0, True])


    # x intercepts between line and lens      
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
         # equation_inner_lens = [1/10, 0, -3, 0, 1, -10, 10]
        # a,b,c,d,  e,f,] where a,b,c,d,e are the coefficients of the equation ax^2 + bx + c = ????dy^2 + ey ??? and f and g are the x ranges of the lens
        #  ray_equations.append([angle_to_gradient(angle), 0, 0, 0, True])

        #[m, c, start, finish, diffract]
        # TODO could this use c instead of origin?  min(x_intercept[i], ray_equations[1])
        # x range ray (min always less than max)
        ray_equations[i][2] = min(x_intercept[i], 0)
        ray_equations[i][3] = max(x_intercept[i], 0)
        
        #[m, c, start, finish, diffract] 
        # [1/10, 0, -3, 0, 1, -10, 10]   equation_inner_lens  
        # if line x-intercept start point comes before or after ray equation starts/finishs boundaries 
        # if any part of endpoints of line is outside boundaries
        if(ray_equations[i][2] < lens_equation[5] or ray_equations[i][2] > lens_equation[6]):
            ray_equations[i][2] = -max_distance
            ray_equations[i][4] = False
        #[m, c, start, finish, diffract] 
        # [1/10, 0, -3, 0, 1, -10, 10]   equation_inner_lens  
        # if line x-intercept finish point comes before or after ray equation starts/finishs boundaries 
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
    # print("angles in intial_eqations found")
    # print_degrees_of_ray_eqns(ray_equations_true)
    return ray_equations, ray_equations_true, ray_equations_false

# keep all lines but leave limits untouched when not diffracting
    

    
# plots lens in red
def plot_lens(lens_equation):
    x_vals = np.linspace(lens_equation[5], lens_equation[6], 400)
    y_vals = lens_equation[0] * x_vals**2 + lens_equation[1] * x_vals + lens_equation[2]
    plt.plot(x_vals, y_vals, color = 'violet')
    
# only plots rays that diffract
def plot_initial_rays(ray_equations):
    for i, equation in enumerate(ray_equations):
         if(equation[4] == True):
            x_vals = np.linspace(equation[2], equation[3], 400)
            y_vals = equation[0] * x_vals + equation[1] 
            plt.plot(x_vals, y_vals, color = 'blue')
            
def plot_middle_rays(ray_equations):
    for i, equation in enumerate(ray_equations):
         if(equation[4] == True):
            x_vals = np.linspace(equation[2], equation[3], 400)
            y_vals = equation[0] * x_vals + equation[1] 
            plt.plot(x_vals, y_vals, color = 'red')
            
            
            
def plot_exit_rays(ray_equations):
    for i, equation in enumerate(ray_equations):
         if(equation[4] == True):
            x_vals = np.linspace(equation[2], equation[3], 400)
            y_vals = equation[0] * x_vals + equation[1] 
            plt.plot(x_vals, y_vals, color = 'green')




def find_middle_ray_equations(initial_ray_equations, inner_lens_equation, outer_lens_equation, refractive_index_lens, refractive_index_air):
    # [m, c, start, finish, diffract]
    # [a,b,c,d,e] where a,b,c,d,e are the coefficients of the equation ax^2 + bx + c = and d and e are the x ranges of the lens
    # new ray equations copy initial ray equations
    middle_ray_equations = copy.deepcopy(initial_ray_equations)
    initial_angles_in_radians = [gradient_to_angle(equation[0]) for equation in initial_ray_equations]
    initial_angles_in_degrees= np.degrees(initial_angles_in_radians)
    #convert negative angle to positive
    for i, angle in enumerate(initial_angles_in_radians):
        if(angle < 0):
            #eg 180 + (-10 degrees)
            initial_angles_in_radians[i] = pi + angle
    theta_3 = np.zeros(len(initial_angles_in_radians))
    initial_angles_in_degrees2= np.degrees(initial_angles_in_radians)
    
    for i, initial_ray_equation in enumerate(initial_ray_equations):
        # only for rays that diffract
        if(initial_ray_equation[4] == True):
            # find theta_3
            a,b,c = find_derivative(inner_lens_equation[0:3])
            #if it travels more to the left of origin
            if(abs(initial_ray_equation[2]) > abs(initial_ray_equation[3])):
                # GRADIENT
                tangent_at_x = a * initial_ray_equation[2] + b
                #tangent_at_x= [a, b, np.nan, np.nan, np.nan]
            else:
                tangent_at_x = a * initial_ray_equation[3] + b
                #tangent_at_x= [a, b, np.nan, np.nan, np.nan]
            theta_3[i] = gradient_to_angle(tangent_at_x)
            # print("np.degrees( theta_3[i])")
            # print(np.degrees( theta_3[i]))
            
            #theta_3[i] = gradient_to_angle(tangent_at_x[0])
            
            # new gradient of ray in lens
            # new_ray_equations[i][0] = angle_to_gradient(pi/2 - (-theta_3[i] + np.arcsin((refractive_index_air / refractive_index_lens) * np.np.cos(- theta_3[i] - (initial_angles_in_radians[i])))))
            # angle to normal of tangent angle, trig



            #middle_ray_equations[i][0] = angle_to_gradient(np.degrees(pi/2 - (np.arcsin((refractive_index_air / refractive_index_lens) * np.sin(pi/2 - theta_3[i] - initial_angles_in_radians[i])))+ theta_3[i]))
            # add theta_3 to the equation to account for tangent

            if abs((theta_3[i])) > (gradient_to_angle(initial_angles_in_radians[i])):
                middle_ray_equations[i][0] = angle_to_gradient(np.degrees(pi/2 - (np.arcsin((refractive_index_lens / refractive_index_air) * np.sin(pi/2 + theta_3[i] - initial_angles_in_radians[i])))+ theta_3[i]))
            else:
                middle_ray_equations[i][0] = angle_to_gradient(np.degrees(pi/2 - (np.arcsin((refractive_index_lens / refractive_index_air) * np.sin(pi/2 - theta_3[i] - initial_angles_in_radians[i])))- theta_3[i]))
                   
            #####[m, c, start, finish, diffract]
            ##### if line starts more to left than right of orin
            # new y intercept of ray in lens
            if(abs(initial_ray_equation[2]) > abs(initial_ray_equation[3])):
                # initial = exit
                # m * start (xintercept) + c = m2 *  start + c2
                # rearange solve for c2
                middle_ray_equations[i][1] = (initial_ray_equations[i][0] * initial_ray_equations[i][2] + initial_ray_equations[i][1]) - middle_ray_equations[i][0] * initial_ray_equations[i][2]
            else:
                middle_ray_equations[i][1] = (initial_ray_equations[i][0] * initial_ray_equation[3] + initial_ray_equations[i][1]) - middle_ray_equations[i][0] * initial_ray_equation[3]
                
                
            #### 
                
            ##### [m, c, start, finish, diffract]
            ##### equation_outer_lens = [0, 0, -5, 0, 1, -10, 10]

            # new x intercept of ray in lens
            if(outer_lens_equation[0] == 0 and outer_lens_equation[1] == 0):
                #mx+c = outer lens y
                intercepts_outer_lens = ((outer_lens_equation[2] - middle_ray_equations[i][1] ) / middle_ray_equations[i][0]) 
                #mx+c = outer lens mx + c
            elif(outer_lens_equation[0] == 0 and outer_lens_equation[1] != 0):
                intercepts_outer_lens = (outer_lens_equation[2] - middle_ray_equations[i][1]) / (middle_ray_equations[i][0] - outer_lens_equation[1])
                
            else:
                
                #mx+c = outer lens ax^2 + mx + c
                first_solution, second_solution = solve_quadratic(outer_lens_equation[0], outer_lens_equation[1] - middle_ray_equations[i][0], outer_lens_equation[2] - middle_ray_equations[i][1])
                if(initial_ray_equation[0] > 0):
                    #if gradient is positive, then goes from origin to left, take left solution
                    intercepts_outer_lens = min(first_solution, second_solution)
                else:
                    intercepts_outer_lens = max(first_solution, second_solution)
                
                
                
            
            # find intercept of inner lens
            # line goes more left than right of origin
            if(abs(initial_ray_equation[2]) > abs(initial_ray_equation[3])):
                intercepts_inner_lens = initial_ray_equation[2] #start
            else:
                intercepts_inner_lens = initial_ray_equation[3] # end
            
            # for rays that go left
            if(initial_ray_equation[0] > 0):
                if(intercepts_outer_lens < 0):
                    if(intercepts_outer_lens < outer_lens_equation[5] or intercepts_outer_lens > outer_lens_equation[6]):
                        middle_ray_equations[i][3] = -max_distance
                        middle_ray_equations[i][2] = intercepts_inner_lens
                        middle_ray_equations[i][4] = False
                    else:

                        middle_ray_equations[i][3] = intercepts_outer_lens
                        middle_ray_equations[i][2] = intercepts_inner_lens
                # goes left but intercept right of origin        
                else:
                    if(intercepts_outer_lens < outer_lens_equation[5] or intercepts_outer_lens > outer_lens_equation[6]):
                        middle_ray_equations[i][3] = -max_distance
                        middle_ray_equations[i][2] = intercepts_inner_lens
                        middle_ray_equations[i][4] = False
                    else:
                        middle_ray_equations[i][2] = intercepts_inner_lens
                        # if(middle_ray_equations[i][0] < max_distance):
                        middle_ray_equations[i][3] = -max_distance
            # ray goes right    
            else:
                if(intercepts_outer_lens > 0):
                    if(intercepts_outer_lens < outer_lens_equation[5] or intercepts_outer_lens > outer_lens_equation[6]):
                        middle_ray_equations[i][3] = max_distance
                        middle_ray_equations[i][2] = intercepts_inner_lens
                        middle_ray_equations[i][4] = False
                    else:
                        middle_ray_equations[i][3] = intercepts_outer_lens
                        middle_ray_equations[i][2] = intercepts_inner_lens
                        
                else:
                    if(intercepts_outer_lens < outer_lens_equation[5] or intercepts_outer_lens > outer_lens_equation[6]):
                        middle_ray_equations[i][3] = max_distance
                        middle_ray_equations[i][2] = intercepts_inner_lens
                        middle_ray_equations[i][4] = False
                    else:
                        middle_ray_equations[i][2] = intercepts_inner_lens
                        middle_ray_equations[i][3] = max_distance
                
            
            
            
            
            
            
            
    ray_equations_true = []
    ray_equations_false = []
    for ray in middle_ray_equations:
        if(ray[4] == True):
            ray_equations_true.append(ray)
        else:
            ray_equations_false.append(ray)
    # print("angles in middle_eqations found")
    # print_degrees_of_ray_eqns(ray_equations_true)

    return middle_ray_equations
            
            
    


def find_exit_ray_equations(middle_ray_equations, outer_lens_equation, refractive_index_lens, refractive_index_air):
    #print (middle_ray_equations)
    exit_ray_equations = copy.deepcopy(middle_ray_equations)
    middle_angles_in_radians = [gradient_to_angle(equation[0]) for equation in middle_ray_equations]
    for i, angle in enumerate(middle_angles_in_radians):
        if(angle < 0):
            middle_angles_in_radians[i] = pi + angle
    theta_3 = np.zeros(len(middle_angles_in_radians))
    
    # for rads in middle_angles_in_radians:
    #     print("middle_angles_in_degrees at start exit ray equations" )
    #     print(np.degrees(rads))
    # print()
    ###
    #[m, c, start, finish, diffract]
    # print("middle")
    # print("[m, c, start, finish, diffract]")
    # for equation in middle_ray_equations:
    #     print(equation)
    # print(len(middle_ray_equations))
    # print("len diffract")
    # print(sum(1 for middle_ray_equation in middle_ray_equations if middle_ray_equation[4]))
    arcsin_input = (refractive_index_lens / refractive_index_air) * np.sin(pi/2 - theta_3[i] - middle_angles_in_radians[i])
    for i, middle_ray_equation in enumerate(middle_ray_equations):
        # only for rays that diffract
        if(middle_ray_equation[4] == True):
            # find theta_3
            a,b,c = find_derivative(outer_lens_equation[0:3])
                # derivative_a = 2 * a
                # derivative_b = b
                # derivative_c = c*0
            #[a,b,c,d,e] where a,b,c,d,e are the coefficients of the equation ax^2 + bx + c = and d and e are the x ranges of the lens
            #[m, c, start, finish, diffract]
            # if line goes from left to right
            
            #equation_outer_lens = [0, 0, -5, 0, 1, -10, 10]
            

            # further from origin
            if(abs(middle_ray_equation[2]) > abs(middle_ray_equation[3])):
                tangent_at_x = a * middle_ray_equation[2] + b
                
                #tangent_at_x = [a, b, np.nan, np.nan, np.nan]
            else:
                
                tangent_at_x = a * middle_ray_equation[3] + b
                #tangent_at_x= [a, b, np.nan, np.nan, np.nan]
            theta_3[i] = gradient_to_angle(tangent_at_x)

            
            
            # new gradient of ray 
            # print(middle_ray_equation[0])

            arcsin_input = (refractive_index_lens / refractive_index_air) * np.sin(pi/2 - theta_3[i] - middle_angles_in_radians[i])


            # print("arcsin_input")
            # print(arcsin_input)
            
            # handle cases of light refracting greater than critical angle #TODO ?
 
            
            # ray going left
            if(middle_ray_equation[3] < middle_ray_equation[2]):
                # print("aaaaaaaaa")
                # print(np.degrees(gradient_to_angle(middle_ray_equations[i][0])))
                
                # ANGLE POSITIVE
                if (arcsin_input >= 0 and arcsin_input < 1):

                    if abs((theta_3[i])) > (gradient_to_angle(middle_ray_equations[i][0])):
                        exit_ray_equations[i][0] = angle_to_gradient(np.degrees(pi/2 - (np.arcsin((refractive_index_lens / refractive_index_air) * np.sin(pi/2 + theta_3[i] - middle_angles_in_radians[i])))+ theta_3[i]))
                    else:
                        exit_ray_equations[i][0] = angle_to_gradient(np.degrees(pi/2 - (np.arcsin((refractive_index_lens / refractive_index_air) * np.sin(pi/2 - theta_3[i] - middle_angles_in_radians[i])))- theta_3[i]))
                   
                    # print(np.degrees(gradient_to_angle(exit_ray_equations[i][0])))
                    # print(np.degrees(theta_3[i]))
                else:
                    
                    exit_ray_equations[i][0] = np.nan
            #ray goes right
            else:
                #negative angle
                if(arcsin_input < 0 and arcsin_input > -1):
                    exit_ray_equations[i][0] = angle_to_gradient(np.degrees(pi/2 - (np.arcsin((refractive_index_lens / refractive_index_air) * np.sin(pi/2 - theta_3[i] - middle_angles_in_radians[i])))- theta_3[i]))
                else:
                    exit_ray_equations[i][0] = np.nan

            
            # new y intercept of ray 
            # goes left ????????????????????????
            if(abs(middle_ray_equations[i][2]) > abs(middle_ray_equations[i][3])):
                #inital ray mx+c = exit ray mx + c
                # intercept start or end
                exit_ray_equations[i][1] = (middle_ray_equations[i][0] * middle_ray_equations[i][2] + middle_ray_equations[i][1]) - exit_ray_equations[i][0] * middle_ray_equations[i][2]
            else:
                exit_ray_equations[i][1] = (middle_ray_equations[i][0] * middle_ray_equation[3] + middle_ray_equations[i][1]) - exit_ray_equations[i][0] * middle_ray_equation[3]
                
                
            # new x limit of rays
            # stary furtherest from origin of middle ray (abs(start) always less)
            # eg -1 -6,  1, 6
            exit_ray_equations[i][2] = middle_ray_equations[i][3]
            # if goes more left
            if(middle_ray_equations[i][2] > middle_ray_equations[i][3]):
                exit_ray_equations[i][3] = -max_distance
            else:
                exit_ray_equations[i][3] = max_distance
                

    return exit_ray_equations
            
                





plt.figure(figsize=(8, 6))

# rays = [m, c, start, finish, diffract]
#lenses are in the form [a,b,c,d,e,f,g] where a,b,c,d,e are the coefficients of the equation ax^2 + bx + c = dy^2 + ey and f and g are the x ranges of the lens

equation_inner_lens =  [0, 0, -1, 0, 1, -10, 10]
equation_outer_lens = [1/50, 0, -10, 0, 1, -10, 10]


input_file_path = "./TestFile1/test.ies"
input_filename = os.path.splitext(os.path.basename(input_file_path))[0]
    

initial_angles = []
initial_candelas_values = []
vertical_line_index = None
horizontal_angle_index = None
num_angles_index = None
num_vertical_angles = None
num_horizontal_angles = None

# # edit num lines
# line = lines[num_angles_index]
# numbers = line.split()
# numbers[3] = str(len(exit_angles_in_degrees))
# #numbers[4] = str(num_horizontal_lines) TODO account for horizontal lines?
# new_line = " ".join(numbers) + "\n"
# lines[num_angles_index] = new_line

with open(input_file_path, 'r') as file:
    lines = file.readlines()


# find index of 'TILT='
for i, line in enumerate(lines):
    if 'TILT=' in line:
        if 'TILT=NONE' in line:
            num_angles_index= i+1
            vertical_line_index = i+3
        else:
            #TILE=INCLUDE OR <filespec>
            num_angles_index = i+2
            vertical_line_index= i+4
        break


# edit num lines
line = lines[num_angles_index]
numbers = line.split()
num_vertical_angles = int(numbers[3])
num_horizontal_angles = int(numbers[4])
#numbers[3] = str(len(exit_angles_in_degrees))
#numbers[4] = str(num_horizontal_lines) TODO account for horizontal lines?
# new_line = " ".join(numbers) + "\n"
# lines[num_angles_index] = new_line

# Check if "TILT=" was found
if vertical_line_index is not None:

    # collect first line data
    line = lines[vertical_line_index]
    parts = line.split()
    for part in parts:
        initial_angles.append(float(part))

    # index vertical_line_index
    for i in range(vertical_line_index+1, len(lines)):
        line = lines[i]

        parts = line.split()
        for part in parts:
            if '0.0' == part or '0.00' == part or '0.000' == part:
                horizontal_angle_index = i
                break
            initial_angles.append(float(part))

        if horizontal_angle_index is not None:
            break
            
if horizontal_angle_index is not None:
    for i in range(horizontal_angle_index + 1, len(lines)):
        if num_horizontal_angles==1:
            line = lines[i]
            parts = line.split()
            for part in parts:
                initial_candelas_values.append(float(part))

        else:
            line = lines[i]
            parts = line.split()
            candela_values = [float(part) for part in parts]
            initial_candelas_values.append(candela_values)         



#print the angles

print("initial_angles")
initial_angles.pop(0)
print(initial_angles)

print("initial_candelas_values[0]")
print(initial_candelas_values[0])

print("len(initial_angles)")
print(len(initial_angles))

if num_horizontal_angles == 1:
    print("len(initial_candelas_values)")
    print(len(initial_candelas_values))
else:
    print("len(initial_candelas_values[0])")
    print(len(initial_candelas_values[0]))

print()  # Just for a blank line



# initial_candelas_values= [
#     100, 99.9875, 99.95, 99.8875, 99.8, 99.6875, 99.55, 99.3875, 99.2, 98.9875,
#     98.75, 98.4875, 98.2, 97.8875, 97.55, 97.1875, 96.8, 96.3875, 95.95, 95.4875,
#     95, 94.4875, 93.95, 93.3875, 92.8, 92.1875, 91.55, 90.8875, 90.2, 89.4875,
#     88.75, 87.9875, 87.2, 86.3875, 85.55, 84.6875, 83.8, 82.8875, 81.95, 80.9875,
#     80, 78.9875, 77.95, 76.8875, 75.8, 74.6875, 73.55, 72.3875, 71.2, 69.9875,
#     68.75, 67.4875, 66.2, 64.8875, 63.55, 62.1875, 60.8, 59.3875, 57.95, 56.4875,
#     55, 53.4875, 51.95, 50.3875, 48.8, 47.1875, 45.55, 43.8875, 42.2, 40.4875,
#     38.75, 36.9875, 35.2, 33.3875, 31.55, 29.6875, 27.8, 25.8875, 23.95, 21.9875,
#     20, 17.9875, 15.95, 13.8875, 11.8, 9.6875, 7.55, 5.3875, 3.2, 0.9875, -1.25]



initial_ray_equations = find_initial_ray_equations(initial_angles, equation_inner_lens)[0]

# initial_ray_equations = find_initial_ray_equations(initial_angles, equation_inner_lens)[0]
# initial_ray_equations = find_initial_ray_equations(generate_angles(150, 170, 10)[0], equation_inner_lens)[0]
middle_ray_equations = find_middle_ray_equations(initial_ray_equations, equation_inner_lens, equation_outer_lens, 1, 1)



exit_ray_equations = find_exit_ray_equations(middle_ray_equations, equation_outer_lens, 1, 1)

print("\nangles in exit_ray_equations found")
print_degrees_of_ray_eqns(exit_ray_equations)

# print(len(initial_ray_equations))
# print(len(middle_ray_equations))
# print(len(exit_ray_equations))
#########


# exit_angles = [np.degrees(gradient_to_angle(exit_ray[0])) for exit_ray in exit_ray_equations]
# converted_exit_angles = [90 - angle for angle in exit_angles]
# print(converted_exit_angles)

new_exit_ray_equations = copy.deepcopy(exit_ray_equations)
exit_angles1 = [] 

for ray in new_exit_ray_equations:
    old_gradient = ray[0]
    old_angle = gradient_to_angle(old_gradient)
    new_angle = pi/2 - old_angle

    # print("old degrees -> new degrees")
    # print(f"{np.degrees(old_angle)} {np.degrees(new_angle)}")
    # print(np.degrees(old_angle) + np.degrees(new_angle) )
    exit_angles1.append(np.degrees(new_angle))
    # new_gradient = angle_to_gradient(new_angle)



exit_angles1= [90] + exit_angles1
exit_angles1[-1] = 0
    # ray[0] = new_gradient
# print("exit angles1")
# print(exit_angles1)
exit_angles = exit_angles1[::-1]

#print_degrees_of_ray_eqns(exit_ray_equations)
angles_dont_exit =[]
exit_candelas = []
i=0
j=0
exit_ray_equations.reverse()
while i < len(exit_ray_equations):
    if exit_ray_equations[i][4] == False:
        #exit_angles.append(initial_angles[i])
        exit_candelas.append(0)
        angles_dont_exit.append(exit_angles[i])

    else:
        #exit_angles.append(np.degrees(gradient_to_angle(exit_ray_equations[j][0])))
        exit_candelas.append(initial_candelas_values[i])
        #j+=1
        #print(np.degrees(gradient_to_angle(exit_ray_equations[j][0])))
    i+=1


angles_dont_exit.reverse()
exit_candelas.append(0)



# print(exit_candelas)
# print("angles dont exist from below and up")
# print(angles_dont_exit[-1])
# print("number of angles lost")
# print(len(angles_dont_exit))



# for i, j in range(len(initial_angles)):
#         if exit_ray_equations[i][4] == True:
#             exit_angles.append(gradient_to_angle(exit_ray_equations[j][0]))
#             exit_candelas.append(initial_candelas_values[i])
#         else:
#             exit_angles.append(initial_angles[i])
#             exit_candelas.append(0)



# exit_angles = [gradient_to_angle(lst[0]) if lst[4] == True else initial_angles[i] for i, lst in enumerate(exit_ray_equations)]

# exit_angles_in_degrees = [0] + exit_angles_in_degrees
# print ("exit angles in degrees in output")
# print(exit_angles)



# exit_candelas = [initial_candelas_values[0]]
# for i in range(len(initial_candelas_values)):
#         if exit_ray_equations[i][4] == True:
#             exit_candelas.append(initial_candelas_values[i])
#         else:
#             exit_candelas.append(0)
        



# # Find indices of np.nan in list1
# nan_indices = [i for i, x in enumerate(exit_angles_in_degrees) if isinstance(x, float) and np.isnan(x)]
# # Remove corresponding elements from both lists
# for i in reversed(nan_indices):
#     del exit_angles_in_degrees[i]
#     if num_horizontal_angles==1:
#         del exit_candelas[i]
#     else:
#         for sublist in exit_candelas:
#             del sublist[i]
    

# print(len(exit_angles))
# print(len(exit_candelas))

# print(exit_angles)

exit_angles_no_nan = []
exit_candelas_no_nan = []

for i in range(len(exit_angles)):
    if not np.isnan(exit_angles[i]):
        exit_angles_no_nan.append(exit_angles[i])
        exit_candelas_no_nan.append(exit_candelas[i])

print(exit_angles_no_nan)
print(exit_candelas_no_nan)

exit_angles= exit_angles_no_nan
exit_candelas= exit_candelas_no_nan

# edit num vertical lines
line = lines[num_angles_index]
numbers = line.split()
numbers[3] = str(len(exit_angles))
new_line = " ".join(numbers) + "\n"
lines[num_angles_index] = new_line



output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# create file
output_lines = []

output_lines.extend(lines[:vertical_line_index])
output_lines.append(" ".join(map(str, [round(angle, 2) for angle in exit_angles])))
output_lines.append("\n")
output_lines.append(lines[horizontal_angle_index])

if num_horizontal_angles == 1:
    output_lines.append(" ".join(map(str, exit_candelas)))
else:
    for sublist in exit_candelas:
        output_lines.append(" ".join(map(str, sublist)))
        output_lines.append("\n")


output_lines.append("\n")
output_file_path = os.path.join(output_dir, f"{input_filename}_output.ies")
with open(output_file_path, 'w') as output_file:
    output_file.write(''.join(output_lines))


########################################
plot_lens(equation_inner_lens)
plot_lens(equation_outer_lens)
plot_initial_rays(initial_ray_equations)
plot_middle_rays(middle_ray_equations)
plot_exit_rays(exit_ray_equations)



plt.xlim(-20, 20)
plt.ylim(-40, 0)
    

    
plt.xlabel('x')
plt.ylabel('y')
plt.title('Lines at Different Angles')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()


    
