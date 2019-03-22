import matplotlib.pyplot as plt
import numpy as np
import george as g





bound = 10
yerr = 0.5

# Initialise the GP object
# Specify a stationary kernel. 
kernel = g.kernels.ExpSquaredKernel(0.5)

# Use that kernel to make a GP object
gp = g.GP(kernel)


##################################################################################################
########################### Plotting setup -- Irelevant to GP testing! ###########################
##################################################################################################


from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Column, Band, Whisker
from bokeh.io import curdoc
from bokeh.events import DoubleTap
from bokeh.models.widgets import markups
from bokeh.layouts import column

coordList=[]

TOOLS = "tap,reset"
p = figure(title='Double click to leave a dot.',
           tools=TOOLS,width=1000,height=700,
           x_range=(-bound, bound), y_range=(-5, 5))

# Prediction x points
xp = np.linspace(-10., 10., 100)

source = ColumnDataSource(data=dict(
    x=[], y=[], upper=[], lower=[]))

modelSource = ColumnDataSource(data=dict(
    xp=xp, yp=np.zeros_like(xp), upper=np.zeros_like(xp), lower=np.zeros_like(xp)
))

# Data
p.circle(source=source,x='x',y='y', color='black')
errorbars = Whisker(base='x', upper='upper', lower='lower', source=source,
            upper_head=None, lower_head=None, line_color='black')
p.add_layout(errorbars)

# GP Mean
p.line(source=modelSource, x='xp', y='yp', color='green')

# GP Variance
band = Band(base='xp', lower='lower', upper='upper', source=modelSource,
    level='underlay', fill_alpha=0.5, line_width=0, line_color='black', fill_color='green')
p.add_layout(band)

report = markups.Div(width=600)
report.text = "I'll put fit results here...</br>"


###################################################################################################
###################################################################################################
###################################################################################################

from scipy.optimize import minimize

# We need to convert our likelihood to the right stuff...
def neg_ln_like(p):
    # Get the log like for observables, p
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(source.data['y'])

def grad_neg_ln_like(p):
    # Get the gradient of the neg log like
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(source.data['y'])


def fit_data():
    '''Generate the mean and variance of GP curves, given the data'''

    # compute the covariance matrix and factorize it for a set of times and uncertainties.
    error = np.array(source.data['upper']) - np.array(source.data['lower'])
    error /= 2.
    
    gp.compute(source.data['x'], error)
    # Minimize the negative likelihood, i.e. maximize the likelihood of this data for the GP hiperparams
    result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    gp.set_parameter_vector(result.x)
    

    report.text += "GP Parameter vector:</br>{}</br></br>".format(gp.get_parameter_vector())
    report.text += str(result).replace('\n', '</br>') + '</br>'
    report.text += "</br>Final ln-likelihood: {0:.2f}</br>".format(gp.log_likelihood(source.data['y']))
    report.text += '------------------------------------------------------------------------'



    # Use the GP to predict the values of y at these x
    yp, yperr = gp.predict(source.data['y'], modelSource.data['xp'], return_var=True)

    modelSource.data['yp'] = yp
    modelSource.data['upper'] = yp + np.sqrt(yperr)
    modelSource.data['lower'] = yp - np.sqrt(yperr)


#add a dot where the click happened
def callback(event):
    '''Capture clicks, and add them to the observed data'''
    Coords=(event.x,event.y)
    coordList.append(Coords) 

    newdata = dict(
        x=[Coords[0]], 
        y=[Coords[1]],
        upper=[Coords[1]+yerr],
        lower=[Coords[1]-yerr]
    )

    source.stream(newdata)

    fit_data()

# Link the figure
p.on_event(DoubleTap, callback)

layout = column([p, report])

curdoc().add_root(layout)

