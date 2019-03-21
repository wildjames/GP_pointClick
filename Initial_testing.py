import matplotlib.pyplot as plt
import numpy as np
import george as g

# Test data
# np.random.seed(1234)
def f(x, yerr=None):
    if yerr is None: 
        yerr = np.zeros(x.shape)
    
    yo = np.sin(x) + yerr * np.random.randn(len(x))

    # yo = 0.5 * x + yerr*np.random.randn(len(x))

    return yo

xo = 10 * np.sort(np.random.rand(15))
yoerr = 0.2 * np.ones_like(xo)
yo = f(xo, yoerr)

# Initialise the GP object
print('Variance of y: {}'.format(np.var(yo)))
# Specify a kernel with the variance of yo, and a stationary kernel
kernel = np.var(yo) * g.kernels.ExpSquaredKernel(0.5)
print("Kernel has {} dimensions".format(kernel.ndim))
print("Kernel has {} parameter names".format(kernel.parameter_names))
for name in kernel.parameter_names:
    print('{:20}: {}'.format(name, kernel[name]))

# Use that kernel to make a GP object
gp = g.GP(kernel)

# Pre-compute the covariance matrix and factorize it for a set of times and uncertainties.
gp.compute(xo, yoerr)

# Define a prediction array of x values
xp = np.linspace(0, 10, 500)
# Use the GP to predict the values of y at these x
yp, yperr = gp.predict(yo, xp, return_var=True)


# Initialise plotting area
fig, ax = plt.subplots()

# Plot the actual form of the posterior
ax.plot(xp, f(xp), color='red')
# Plot the prediction variance
ax.fill_between(xp, yp+np.sqrt(yperr), yp-np.sqrt(yperr),
    alpha=0.3, color='black')
# Plot the prediction means
ax.plot(xp, yp, color='black')
# Plot the data
ax.errorbar(xo, yo, yoerr, markersize=5, fmt='.k', capsize=0)

print("Initial ln-likelihood: {0:.2f}".format(gp.log_likelihood(yo)))

plt.show()
plt.close()


