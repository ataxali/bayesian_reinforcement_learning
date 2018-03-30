import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared


class GPPosterior:

    def __init__(self, history_manager, kernel=None):
        self.history_manager = history_manager
        self.fitted_model = None
        if not kernel:
            self.kernel = ExpSineSquared(length_scale=1, periodicity=1.0,
                                    periodicity_bounds=(2, 10),
                                    length_scale_bounds=(1, 3))
        else:
            self.kernel = kernel

    def update_posterior(self):
        pass

    def predict(self, time):
        self.fitted_model.predict(time, return_std=True)
