# 700-project

When analysis.py is run the lens equations and refractive index at the top of analysis.py is used. These equations are a 2d representation of the lens shape. 

The initial relative luminous intensity at angles from origin from a lensless LED is in 1 degree increments in test.ies and in 0.25 increments in test2.ies. 
The output file is saved to the output directory, in the format test._output_ies or test2.ies respectively. If more rays of light are required to be used then the formula relative luminous intensity = cos (angle) can be used. The new angles (in order) would be placed in the 11th line and the new relative luminous intensity would be put in the 13th line. The file chosen is input on line 367 of "plotter_for_analysis.py". Then analysis would be run which uses plotter_for_analysis. 

The relative luminous intensity is also grouped into 5 degree bins, eg all candelas from angles 40-45 are averaged and modelled to be at the point 42.5. This allowed another variance and standard deviation to be calculated which produces clearer information about the differences in distributions between the initial and exit (after leaving the lens) relative luminous intensity - angle maps.
