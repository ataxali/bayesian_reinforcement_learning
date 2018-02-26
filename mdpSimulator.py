

class MDPSimulator(object):
    """Abstract class for MDP Simulator"""
    def sim(self, state, action):
        """ Must return (new_state, reward) tuple """
        raise NotImplementedError("Unimplemented method!")