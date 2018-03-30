import numpy as np
import logger
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared


class GPPosterior:

    def __init__(self, history_manager, kernel=None, penalty_threshold=-1, log=None):
        self.history_manager = history_manager
        self.fitted_models_x = None
        self.fitted_models_y = None
        self.penalty_threshold = penalty_threshold
        self.log = log
        self.x_obs = None
        self.y_obs = None
        if not kernel:
            self.kernel = ExpSineSquared(length_scale=1, periodicity=1.0,
                                    periodicity_bounds=(2, 10),
                                    length_scale_bounds=(1, 3))
        else:
            self.kernel = kernel

    def update_posterior(self, n_restarts=10, a=0.01):
        # each history obs is <orig_state, action, reward, new_state, time>
        history = list(filter(lambda obs: obs[2] < self.penalty_threshold, self.history_manager.history))
        if not len(history): return
        classified_x = self.__classify_history(history, 0)
        classified_y = self.__classify_history(history, 1)

        self.x_obs = classified_x
        self.y_obs = classified_y

        def fit_models(classified_dat, parent):
            for dat in classified_dat:
                t_obs = np.atleast_2d(list(map(lambda obs: obs[0], dat))).T
                i_obs = np.array(list(map(lambda obs: obs[1], dat)))
                gp = GaussianProcessRegressor(kernel=self.kernel,
                                              n_restarts_optimizer=n_restarts,
                                              alpha=a).fit(t_obs, i_obs)
                parent.append(gp)

        self.fitted_models_x = []
        self.fitted_models_y = []
        fit_models(classified_x, self.fitted_models_x)
        fit_models(classified_y, self.fitted_models_y)

    def __classify_history(self, history, new_state_idx):
        hist_ts = set(map(lambda obs: obs[4], history))
        hist_dict = {key: list() for key in hist_ts}
        for obs in history: hist_dict[obs[4]].append(obs[3][new_state_idx])
        collisions = max(map(len, map(lambda x: set(x), hist_dict.values())))
        classes = [list() for _ in range(collisions)]

        def classify_obs(obs, t):
            # trivial case, only 1 class
            if len(classes) == 1:
                classes[0].append((t, obs))
                return

            last_class_vals = list(map(lambda lst: lst[len(lst) - 1] if lst else None, classes))

            # first try to classify to one-off sequence
            for i, val in enumerate(last_class_vals):
                if val and abs(val[0] - t) == 1 and abs(val[1] - obs) == 1:
                    classes[i].append((t, obs))
                    return

            # whoops, rough classification. Need to estimate
            error_message = "Cannot cleanly classify: " + str((t,obs)) + " -> " + str(classes)
            if self.log:
                logger.log(error_message, logger=self.log)
            else:
                print(error_message)

            # toss into next nearest neighbor, frobenius manhattan distance
            min_coord_diff = min(last_class_vals, key=lambda x: abs(x[1] - obs))
            classes[last_class_vals.index(min_coord_diff)].append((t, obs))

            info_message = "Classified " + str(obs) + " to class " + str(last_class_vals.index(min_coord_diff))
            if self.log:
                logger.log(info_message, logger=self.log)
            else:
                print(info_message)

        # initialize classes based on time with greatest collisions
        for t in sorted(hist_dict.keys(), key=int):
            t_obs = hist_dict[t]
            if len(set(t_obs)) == collisions:
                for i, obs in enumerate(set(t_obs)):
                    classes[i].append((t, obs))
                del hist_dict[t]
                break

        for t in sorted(hist_dict.keys(), key=int):
            for obs in hist_dict[t]:
                classify_obs(obs, t)

        # validate classes
        for i, c in enumerate(classes):
            if not c:
                classes.pop(i)
                if self.log:
                    logger.log("Expected more classes than classified!", logger=self.log)
                    logger.log(str(history), logger=self.log)
                else:
                    print("Expected more classes than classified!", str(history))

        return classes

    def predict(self, time):
        x_preds = []
        x_stds = []
        for gp in self.fitted_models_x:
            preds, stds = gp.predict(time, return_std=True)
            x_preds.append(preds)
            x_stds.append(stds)

        y_preds = []
        y_stds = []
        for gp in self.fitted_models_y:
            preds, stds = gp.predict(time, return_std=True)
            y_preds.append(preds)
            y_stds.append(stds)

        return (x_preds, x_stds), (y_preds, y_stds)
