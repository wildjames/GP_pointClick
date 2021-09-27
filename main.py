
if __name__ == "__main__":
    print("This is a bokeh script, and isn't run like that! From the directory containing GP_pointClick/ you need to run:")
    print("bokeh serve --show GP_pointClick")

else:
    import matplotlib.pyplot as plt
    import numpy as np
    import george as g





    bound = 10
    yerr = 0.5


    ##################################################################################################
    ########################### Plotting setup -- Irelevant to GP testing! ###########################
    ##################################################################################################


    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, Column, Band, Whisker
    from bokeh.io import curdoc
    from bokeh.events import DoubleTap
    from bokeh.models.widgets import markups, Dropdown
    from bokeh.layouts import column, row

    coordList=[]

    TOOLS = "tap,reset"
    p = figure(title='Double click to leave a dot',
            tools=TOOLS,width=900,height=600,
            x_range=(-bound, bound), y_range=(-bound, bound))

    # Prediction x points
    xp = np.linspace(-bound, bound, 200)

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

    report = markups.Div(width=300)
    report.text = "I'll put fit results here...</br>"

    kernel_summary = markups.Div(width=300)
    kernel_summary.text = "I'll put kernel summaries here..."


    kernelList = [
        ('Matern-5/2', '0'),
        ('ExpSquared', '1'),
        ('RationalQuadratic', '2'),
        ('Exp', '3'),
        ('Matern-3/2', '4'),
        ('Constant', '5')
    ]
    selectKernel = Dropdown(width=200, label='Select a kernel', button_type='primary', menu=kernelList)

    ###################################################################################################
    ###################################### GP Stuff starts here! ######################################
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
        # This step is the bottleneck for large data sets!
        error = np.array(source.data['upper']) - np.array(source.data['lower'])
        error /= 2.
        gp.compute(source.data['x'], error)

        # Minimize the negative likelihood, i.e. maximize the likelihood of this data for the GP hiperparams
        result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
        gp.set_parameter_vector(result.x)

        # Report to the user
        report.text = ''
        report.text += "GP Parameter vector:</br>{}</br></br>".format(gp.get_parameter_vector())
        report.text += str(result).replace('\n', '</br>') + '</br></br>'
        report.text += "Final ln-likelihood: {0:.2f}</br></br>".format(gp.log_likelihood(source.data['y']))
        report.text += "GP Parameter vector:</br>{}</br>".format(gp.get_parameter_vector())
        report.text += '--------------------------------------------------------</br>'

        # Use the GP to predict the values of y at these x
        yp, yperr = gp.predict(source.data['y'], modelSource.data['xp'], return_var=True)

        # Push into model plotting data source.
        modelSource.data['yp'] = yp
        modelSource.data['upper'] = yp + np.sqrt(yperr)
        modelSource.data['lower'] = yp - np.sqrt(yperr)


    #add a dot where the click happened
    def callback(event):
        '''Capture clicks, and add them to the observed data'''
        Coords=(event.x,event.y)

        newdata = dict(
            x=[Coords[0]],
            y=[Coords[1]],
            upper=[Coords[1]+yerr],
            lower=[Coords[1]-yerr]
        )

        source.stream(newdata)

        try:
            gp
            fit_data()
        except:
            pass

    def change_kernel(event):
        new_kernel = int(event.item)

        global kernel
        global gp
        global kernel_summary

        try:
            try:
                # Get the mean separation between data
                x_data = np.array(sorted([ c for c in source.data['x'] ]))
                l = x_data[:-1] - x_data[1:]
                l = abs(np.mean(l))
            except:
                # Default value for when there's only one datum
                l = 0.5


            if new_kernel == 0:
                print("Changing kernel to Matern-5/2")

                kernel = g.kernels.Matern52Kernel(l)

                p.title.text = 'Double click to leave a dot. Using {}Kernel'.format('Matern-5/2')

                kernel_summary.text = "Matern-5/2:</br>"
                kernel_summary.text += "<img style='height: 100%; width: 100%; object-fit: contain' src='GP_pointClick/static/matern52.png'>"
                kernel_summary.text += 'r is the distance in x between two subsequent data.'

            elif new_kernel == 1:
                print('Changing kernel to ExpSquared')

                kernel = g.kernels.ExpSquaredKernel(l)

                p.title.text = 'Double click to leave a dot. Using {}Kernel'.format('ExpSquared')

                kernel_summary.text = "ExpSquared:</br>"
                kernel_summary.text += "<img style='height: 100%; width: 100%; object-fit: contain' src='GP_pointClick/static/expsquared.png'>"
                kernel_summary.text += 'r is the distance in x between two subsequent data.'

            elif new_kernel == 2:
                print("Changing kernel to RationalQuadratic")

                log_alpha = 1
                kernel = g.kernels.RationalQuadraticKernel(log_alpha=log_alpha, metric=l)

                p.title.text = 'Double click to leave a dot. Using {}Kernel, with log_alpha={}'.format('RationalQuadratic', log_alpha)

                kernel_summary.text = "RationalQuadratic:</br>"
                kernel_summary.text += "<img style='height: 100%; width: 100%; object-fit: contain' src='GP_pointClick/static/rationalquad.png'>"
                kernel_summary.text += "r is the distance in x between two subsequent data, and alpha is a scale length dictiating the 'memory' of the process."

            elif new_kernel == 3:
                print("Changing kernel to Exp")

                kernel = g.kernels.ExpKernel(l)

                kernel_summary.text = "Exp:</br>"
                kernel_summary.text += "<img style='height: 100%; width: 100%; object-fit: contain' src='GP_pointClick/static/exp.png'>"
                kernel_summary.text += 'r is the distance in x between two subsequent data.'

            elif new_kernel == 4:
                print("Changing kernel to Matern-3/2")

                kernel = g.kernels.Matern32Kernel(l)

                p.title.text = 'Double click to leave a dot. Using {}Kernel'.format('Matern-3/2')

                kernel_summary.text = "Matern-3/2:</br>"
                kernel_summary.text += "<img style='height: 100%; width: 100%; object-fit: contain' src='GP_pointClick/static/matern32.png'>"
                kernel_summary.text += 'r is the distance in x between two subsequent data.'

            elif new_kernel == 5:
                print("Changing to a constant kernel")

                val = 1
                kernel = g.kernels.ConstantKernel(val)

                p.title.text = 'Double click to leave a dot. Using {}Kernel with the value {}'.format('Constant', val)

                kernel_summary.text = ''

            gp = g.GP(kernel)
            fit_data()
        except:
            kernel_summary.text = 'I need at least 1 data point first! Add some, then reselect a kernel.'

    # Link the figure
    p.on_event(DoubleTap, callback)
    selectKernel.on_click(change_kernel)

    layout = row([p,
        column([selectKernel,kernel_summary, report])
    ])

    curdoc().add_root(layout)
