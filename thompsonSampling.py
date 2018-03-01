import numpy as np

print_debug = False


class HistoryManager(object):
    def __init__(self, actions):
        self.history = list()
        self.action_count_reward_dict = dict.fromkeys(actions, (0, 0))
        self.state_count_dict = dict()
        self.total_rewards = 0
        self.action_set = actions

    def reset_history(self):
        self.history = list()
        self.action_count_reward_dict = dict.fromkeys(self.action_set, (0, 0))
        self.state_count_dict = dict()
        self.total_rewards = 0

    def get_action_set(self):
        return self.action_set

    def get_action_count_reward_dict(self):
        return self.action_count_reward_dict

    def get_total_rewards(self):
        return self.total_rewards

    def add(self, observation):
        # each observation must be <orig_state, action, reward, new_state>
        if not isinstance(observation, tuple):
            observation = tuple(observation)
        if not len(observation) == 4:
            raise Exception("<orig_state, action, reward, new_state>")
        self.history.append(observation)
        if observation[1] in self.action_count_reward_dict:
            count, reward = self.action_count_reward_dict[observation[1]]
            self.action_count_reward_dict[observation[1]] = (count+1, reward+observation[2])
            self.total_rewards += observation[2]
        else:
            raise Exception(str(observation[1]),
                            " does not exist in action set dictionary")
        if not self.state_count_dict.keys():
            "Print adding init state"
            self.state_count_dict[tuple(observation[0])] = 1
        if observation[3] in self.state_count_dict:
            self.state_count_dict[observation[3]] += 1
        else:
            self.state_count_dict[observation[3]] = 1


class BootstrapHistoryManager(HistoryManager):
    def __init__(self, actions, batch_prop):
        super(BootstrapHistoryManager, self).__init__(actions)
        self.batch_prop = batch_prop

    def get_action_count_reward_dict(self):
        if len(self.history) == 0:
            return dict.fromkeys(self.action_set, (0, 0))

        bootstrap_sample_size = max(int(round(self.batch_prop * len(self.history))), 1)
        bootstrap_idxs = np.random.choice(len(self.history), bootstrap_sample_size, replace=True)
        bootstrap_sample = [self.history[i] for i in bootstrap_idxs]
        # history_with_bootstrap = self.history + bootstrap_sample
        action_reward_dict = self.action_count_reward_dict.copy()
        for sample in bootstrap_sample:
            if sample[1] in action_reward_dict:
                count, reward = action_reward_dict[sample[1]]
                action_reward_dict[sample[1]] = (
                    count + 1, reward + sample[2])
            else:
                action_reward_dict[sample[1]] = (1, sample[2])
        return action_reward_dict


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
