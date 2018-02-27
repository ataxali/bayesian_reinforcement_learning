

class HistoryManager(object):
    def __init__(self):
        self.history = list()
        self.action_set = dict()

    def get_history(self):
        return self.history

    def add(self, observation):
        # each observation must be <orig_state, action, reward, new_state>
        if not isinstance(observation, tuple):
            observation = tuple(observation)
        if not len(observation) == 4:
            raise Exception("<orig_state, action, reward, new_state>")
        self.history.append(observation)
        if observation[1] in self.action_set:
            self.action_set[observation[1]] += 1
        else:
            self.action_set[observation[1]] = 1


class ThompsonSampler(object):
    def __init__(self, history_manager):
        self.history_manager = history_manager

