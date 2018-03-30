import world


class MDPSimulator(object):
    """Abstract class for MDP Simulator"""
    def sim(self, state, action):
        """ Must return <orig_state, action, reward, new_state> tuple """
        raise NotImplementedError("Unimplemented method!")

    def get_valid_actions(self, root, actions):
        raise NotImplementedError("Unimplemented method!")


class WorldSimulator(MDPSimulator):
    WORLD_SIM_CACHE = dict()
    WORLD_VALID_ACTIONS_CACHE = dict()

    def __init__(self, specials, do_render=False, use_cache=False):
        # perhaps init threadpool here
        self.specials = specials
        self.do_render = do_render
        self.use_cache = use_cache

    def __run(self, sim_world, sim_state, sim_action):
        # maze doesnt need current state to simulate
        # sim_world has initialized agent position
        r = -sim_world.score
        if sim_action == sim_world.actions[0]:
            sim_world.try_move_idx(0)
        elif sim_action == sim_world.actions[1]:
            sim_world.try_move_idx(1)
        elif sim_action == sim_world.actions[2]:
            sim_world.try_move_idx(2)
        elif sim_action == sim_world.actions[3]:
            sim_world.try_move_idx(3)
        else:
            return
        s2 = sim_world.player
        r += sim_world.score
        return r, s2

    def sim(self, state, action):
        if self.use_cache:
            if (tuple(state), action, tuple(self.specials)) in WorldSimulator.WORLD_SIM_CACHE:
                # print("Skipped simulation for cached result")
                return WorldSimulator.WORLD_SIM_CACHE[(tuple(state), action, tuple(self.specials))]

        init_x, init_y = self.get_x_y(state)
        sim_world = world.World(self.do_render, init_x, init_y, specials=self.specials)
        sim_r, sim_n_s = self.__run(sim_world, state, action)
        self.specials = sim_world.specials.copy()
        if self.do_render: sim_world.destroy()
        # return values are: <orig_state, action, reward, new_state>
        if self.use_cache:
            WorldSimulator.WORLD_SIM_CACHE[(tuple(state), action, tuple(self.specials))] = (state, action, sim_r, sim_n_s)
        # print("Sim Result: ", state, action, sim_r, sim_n_s)
        return state, action, sim_r, sim_n_s

    def get_x_y(self, state):
        return state[0], state[1]

    def get_valid_actions(self, root, actions):
        valid_actions = []
        init_x, init_y = self.get_x_y(root)
        for action in actions:
            sim_world = world.World(self.do_render, init_x, init_y)
            sim_r, sim_n_s = self.__run(sim_world, root, action)
            if not list(sim_n_s) == list(root):
                valid_actions.append(action)
        if self.do_render: sim_world.destroy()
        return valid_actions


