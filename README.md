This will be a GP learning sandbox for me to figure out how the basics of the maths works.

I produced a script that does the following:
- Create a blank plot that ranges from -10->+10 on the x and -10->+10 on the y
- When the user clicks a point on the plot, add a data point there (with some arbitrary error, 0.5)
- Fit a Gaussian process to the data on screen
- Plot errorbars on what the best fit about those data are
- Have the kernel used to generate the GP be user-changeable

Optionally, I could add:
- Variable errorbars
- Ability to twiddle the GP hiperparameters manually
but probably wont...


There is also a jupyter notebook that I wrote when learning the mechanics of implimenting George, and a notebook that compares the george implimentation of changepoints with the bespoke code that Martin wrote. SPOILER: george is ~10x faster.


To power the plotting script, I used Bokeh. This runs slightly differently to Matplotlib, and needs a server to sustain the IO loop nicely. Rather than using python directly, run `bokeh serve --show GP_pointClick` from the directory *containing* this git repo. i.e., to run it from scratch:

`git clone https://github.com/wildjames/GP_pointClick`

`bokeh serve --show GP_pointClick`
