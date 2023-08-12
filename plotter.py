import numpy as np
import matplotlib.pyplot as plt


array_angles = np.arange(-90, 91, 10)


def formulate_equation(angle):
    return np.tan(np.radians(angle))

gradients = [formulate_equation(angle) for angle in array_angles]


inner_lens = -5
outer_lens = -7

intercepts = [-5 / gradient for gradient in gradients]

    
# pair gradients with intercepts
gradient_intercept_pairs_1 = list(zip(gradients, intercepts))



plt.figure(figsize=(8, 6))

for i, pair in enumerate(gradient_intercept_pairs_1):
    
    # if pair[1] not null
    if(pair[1] >-10 and pair[1] < 10):
        print(pair[0], pair[1])
    
        if(pair[1] < 0):
            x_vals = np.linspace(pair[1], 0, 400)
            y_vals = pair[0] * x_vals
            
        elif(pair[1] > 0):
            x_vals = np.linspace(0, pair[1], 400)
            y_vals = pair[0] * x_vals
            
        plt.plot(x_vals, y_vals, color = 'blue')
        
    
    if((pair[1] <-10) or (pair[1] > 10)):
        print(pair[0], pair[1])
    
        if(pair[1] < 0):
            x_vals = np.linspace(-100, 100, 400)
            y_vals = pair[0] * x_vals
            
        elif(pair[1] > 0):
            x_vals = np.linspace(-100, 100, 400)
            y_vals = pair[0] * x_vals
            
        plt.plot(x_vals, y_vals, color = 'blue')
        

# add horizontal line at y = -5
x_vals = np.linspace(-10, 10, 400)
y_vals = -5 * np.ones(400)

plt.plot(x_vals, y_vals, color = 'red', label = 'y = -5')


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
