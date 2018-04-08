import numpy as np
from global_constants import print_debug
import sys


class ThompsonSampler(object):
    def __init__(self, history_manager, use_constant_boundary=None, move_weight=0.05,
                 move_discount=0.5, num_dirch_samples=100):
        self.history_manager = history_manager
        self.use_constant_boundary = use_constant_boundary
        self.move_weight = move_weight
        self.move_discount = move_discount
        self.num_dirch_samples = num_dirch_samples

    def get_action_set(self, action_set):
        # exploration vs exploitation
        # exploration means not backtracking
        # exploitation means that nearly the complete action set should be considered
        # history for thompson sampler is not assumed to be restricted to current game

        # under beta posterior: exploitation => for each move, alpha > 5, beta <= 1
        # under beta posterior: exploration  => maintain velocity in up/down, left/right directions

        action_psuedo_counts = self.history_manager.get_action_count_reward_dict()
        move_counts = list(map(lambda x: action_psuedo_counts[x][0], action_set))
        history_len = max(sum(move_counts), 1)
        weighted_history = float(history_len * self.move_weight)

        # first we use a beta sample to determine hyper parameter
        # we want to pick a number between 2 and 4, representing number to actions to reduce to
        # when history length > 1/move_weight, we select 3 or 4 moves
        n_sample_hyper = np.random.beta(a=weighted_history, b=1, size=1)[0]
        #n_sample_hyper = np.mean(np.random.beta(a=weighted_history, b=1, size=self.num_dirch_samples))
        # we want to avoid trivial trees, so choose between 2 and 4 moves
        if n_sample_hyper < 1.0/3.0:
            sample_hyper = 2
        elif n_sample_hyper < 2.0/3.0:
            sample_hyper = 3
        else:
            sample_hyper = 4

        print("TS Hyper:", n_sample_hyper)
        sys.stdout.flush()

        sample_hyper = min(sample_hyper, len(action_set))

        def weighted_sum(type):
            running_sum = 0
            discount_power = 0
            for obs in reversed(self.history_manager.history):
                if obs[1] == type:
                    running_sum += (self.move_discount ** discount_power)
                discount_power += 1

            return max(running_sum, 1)

        action_move_counts = list(map(weighted_sum, action_set))

        dirch_samples = np.random.dirichlet(action_move_counts, self.num_dirch_samples).transpose()
        dirch_means = list(map(np.mean, dirch_samples))

        reduced_action_set = []
        for i in range(sample_hyper):
            next_max_idx = dirch_means.index(max(dirch_means))
            reduced_action_set.append(action_set[next_max_idx])
            dirch_means.pop(next_max_idx)
            action_set.pop(next_max_idx)

        return reduced_action_set


