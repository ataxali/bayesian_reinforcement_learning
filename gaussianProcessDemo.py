import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, ExpSineSquared

# some simple example
#tobs = np.atleast_2d([1, 2, 3, 4, 5, 6]).T
#yobs = np.array([1, 1, 2, 1, 1, 2])

# suppose grid is 10 x 10
# suppose observations are complete
# there is one cat
#tobs = np.atleast_2d([7, 5, 6, 8, 1, 2, 11, 14, 7, 8, 6, 5, 4, 3, 2, 1]).T
#yobs = np.array([8, 8, 8, 8, 8, 8, 8, 8, 2, 1, 3, 4, 5, 4, 3, 2])

tobs = np.atleast_2d([1,2,3,4,5,6,7,8,9,10]).T
yobs = np.array([1,2,3,4,5,4,3,2,1,2])


# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
t = np.atleast_2d(np.linspace(0, 20, 1000)).T

# Instanciate a Gaussian Process model
#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
kernel = ExpSineSquared(length_scale=1, periodicity=1.0, periodicity_bounds=(1, 10),
                        length_scale_bounds=(1, 10),)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, alpha=0.2)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(tobs, yobs)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(t, return_std=True)

print("Posterior Log-Likelihood:", (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)))

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = plt.figure()
#plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(tobs, yobs, 'r.', markersize=10, label=u'Observations')
plt.plot(t, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([t, t[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

plt.show(block=True)

