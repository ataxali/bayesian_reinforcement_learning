import numpy as np

print_debug = False


class ThompsonSampler(object):
    def __init__(self, history_manager, use_rewards=True, use_constant_boundary=None):
        self.history_manager = history_manager
        self.use_rewards = use_rewards
        self.use_constant_boundary = use_constant_boundary
        self.eps = 1

    def get_action_set(self, action_set):
        action_psuedo_counts = self.history_manager.get_action_count_reward_dict()
        alphas = list(map(lambda x: action_psuedo_counts[x][0], action_set))
        if self.use_rewards and any(alphas):
            rewards = list(map(lambda x: action_psuedo_counts[x][1], action_set))
            min_reward = min(rewards)
            rewards = [r - (2*min_reward) for r in rewards]
            history_len = sum(alphas)
            if print_debug: print("rewards:", rewards)
            if print_debug: print("move_counts:", alphas)
            alphas = [history_len*float(r)/float(a) if a else 0 for (a, r) in zip(alphas, rewards)]
            if print_debug: print("alphas:", alphas)
            if print_debug: print("sum_alphas:", sum(alphas))
        sum_alphas = sum(alphas)
        reduced_action_set = []
        for i in range(len(action_set)):
            alpha = alphas[i] + 1
            beta = sum_alphas - alphas[i] + 1
            if print_debug: print("alpha[" + str(i) + "]:", alpha)
            if print_debug: print("beta[" + str(i) + "]:", beta)
            action_prob = np.random.beta(alpha, beta)
            if self.use_constant_boundary:
                prob_threshold = self.use_constant_boundary
            else:
                # use float(alpha)/float(alpha+beta) for mean
                # otherwise median
                prob_threshold = (alpha - float(1)/float(3)) / (alpha + beta - float(2)/float(3))
            if action_prob > prob_threshold:
                reduced_action_set.append(action_set[i])
        # dont want states with no possible actions
        if len(reduced_action_set) == 0:
            reduced_action_set = action_set
        return reduced_action_set


