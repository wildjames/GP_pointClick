This will be a proof-of-concept GP learning sandbox for me to figure out how the basics of the maths works.

I will aim to produce a script that does the following:
- Create a blank plot that ranges from -10->+10 on the x and -10->+10 on the y
- When the user clicks a point on the plot, add a data point there (with some arbitrary error, 0.5)
- Fit a Gaussian process to the data on screen
- Plot errorbars on what the best fit about those data are
- Have the kernel used to generate the GP be user-changeable

Optionally:
- Variable errorbars
- Ability to twiddle the GP hiperparameters manually


There is also a jupyter notebook that I wrote when learning the mechanics of implimenting George. 
