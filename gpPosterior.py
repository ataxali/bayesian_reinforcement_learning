import numpy as np
import logger
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
import global_constants


class GPPosterior:

    def __init__(self, history_manager, kernel=None, penalty_threshold=-1, log=None):
        self.history_manager = history_manager
        self.fitted_models_x = []
        self.fitted_models_y = []
        self.penalty_threshold = penalty_threshold
        self.log = log
        self.x_obs = []
        self.y_obs = []
        self.static_states = []
        if not kernel:
            self.kernel = ExpSineSquared(length_scale=1, periodicity=1.0,
                                    periodicity_bounds=(2, 10),
                                    length_scale_bounds=(1, 3))
        else:
            self.kernel = kernel

    def update_static_states(self, state):
        self.static_states.append(tuple(state))

    def get_static_states(self):
        return self.static_states.copy()

    def update_posterior(self, n_restarts=10, a=0.01):
        # each history obs is <orig_state, action, reward, new_state, time>
        history = list(filter(lambda obs: obs[2] < self.penalty_threshold, self.history_manager.get_history()))
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
        hist_vals = set(map(lambda obs: obs[3][new_state_idx], history))

        hist_dict = {key: list() for key in hist_ts}
        for obs in history: hist_dict[obs[4]].append(obs[3][new_state_idx])

        hist_dict_vals = {key: list() for key in hist_vals}
        for obs in history: hist_dict_vals[obs[3][new_state_idx]].append(obs[4])

        collisions = max(map(len, map(lambda x: set(x), hist_dict.values())))
        classes = [list() for _ in range(collisions)]

        def classify_obs(obs, t):
            # trivial case, only 1 class
            if len(classes) == 1:
                classes[0].append((t, obs))
                return

            #last_class_vals = list(map(lambda lst: lst[len(lst) - 1] if lst else None, classes))
            max_class_vals = [max(map(lambda val: val[1], c)) for c in classes]
            sorted_max_class_vals = list(zip(sorted(range(len(max_class_vals)), key=lambda k: max_class_vals[k]),
                                        sorted(max_class_vals)))
            # first try to classify to one-off sequence
            #for i, c in enumerate(classes):
            #    for val in c:
            #        if val and abs(val[0] - t) == 1 and abs(val[1] - obs) == 1:
            #            classes[i].append((t, obs))
            #            return

            for sort_val in sorted_max_class_vals:
                if obs <= sort_val[1]:
                    classes[sort_val[0]].append((t, obs))
                    return

            # whoops, rough classification. Need to estimate
            error_message = "Cannot cleanly classify: " + str((t,obs)) + " -> " + str(classes)
            if self.log:
                logger.log(error_message, logger=self.log)
            else:
                if global_constants.print_debug: print(error_message)

            # toss into next nearest neighbor, frobenius manhattan distance
            min_coord_diff = min(sorted_max_class_vals, key=lambda x: abs(x[1] - obs))
            classes[sorted_max_class_vals.index(min_coord_diff)].append((t, obs))

            info_message = "Classified " + str(obs) + " to classes " + str(classes)
            if self.log:
                logger.log(info_message, logger=self.log)
            else:
                if global_constants.print_debug: print(info_message)

        # initialize classes based on time with greatest collisions
        coll_arr = []
        for t in sorted(hist_dict.keys(), key=int):
            t_obs = hist_dict[t]
            if len(set(t_obs)) == collisions:
                coll_arr.append((t, max(t_obs) - min(t_obs)))

        if coll_arr:
            coll_ranges = list(map(lambda val: val[1], coll_arr))
            max_range_idx = coll_ranges.index(min(coll_ranges))
            t_obs = hist_dict[coll_arr[max_range_idx][0]]
            for i, o in enumerate(sorted(set(t_obs))):
                classes[i].append((coll_arr[max_range_idx][0], o))
            del hist_dict[coll_arr[max_range_idx][0]]

        for val in sorted(hist_dict_vals.keys(), key=int):
            unique_obs = set(hist_dict_vals[val])
            if len(unique_obs) > 1:
                # the set below doesnt duplicate data
                for i, t in enumerate(sorted(set(hist_dict_vals[val]))):
                    classify_obs(val, t)
                    #classes[i].append((t, val))
            else:
                classify_obs(val, list(unique_obs)[0])

        # validate classes
        for i, c in enumerate(classes):
            if not c:
                classes.pop(i)
                if self.log:
                    logger.log("Expected more classes than classified!", logger=self.log)
                    logger.log(str(history), logger=self.log)
                else:
                    if global_constants.print_debug: print("Expected more classes than classified!", str(history))

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
