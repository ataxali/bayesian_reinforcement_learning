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

    def __init__(self, do_render=False, use_cache=True):
        # perhaps init threadpool here
        self.do_render = do_render
        self.use_cache = use_cache

    def __run(self, sim_world, sim_state, sim_action):
        # maze doesnt need current state to simulate
        # sim_world has initialized agent position
        r = -sim_world.score
        if sim_action == sim_world.actions[0]:
            sim_world.try_move(0, -1)
        elif sim_action == sim_world.actions[1]:
            sim_world.try_move(0, 1)
        elif sim_action == sim_world.actions[2]:
            sim_world.try_move(-1, 0)
        elif sim_action == sim_world.actions[3]:
            sim_world.try_move(1, 0)
        else:
            return
        s2 = sim_world.player
        r += sim_world.score
        return r, s2

    def sim(self, state, action):
        if self.use_cache:
            if (tuple(state), action) in WorldSimulator.WORLD_SIM_CACHE:
                # print("Skipped simulation for cached result")
                return WorldSimulator.WORLD_SIM_CACHE[(tuple(state), action)]

        init_x, init_y = self.get_x_y(state)
        sim_world = world.World(self.do_render, init_x, init_y)
        sim_r, sim_n_s = self.__run(sim_world, state, action)
        if self.do_render: sim_world.destroy()
        # return values are: <orig_state, action, reward, new_state>
        if self.use_cache:
            WorldSimulator.WORLD_SIM_CACHE[(tuple(state), action)] = (state, action, sim_r, sim_n_s)
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


# MDP simulator for snake game
# this game has currently been deprecated from the project
# simulator code is left, as the game may be added back in the future
#
# import snake
#
# class SnakeSimulator(MDPSimulator):
#
#     def __init__(self, do_render=False):
#         # perhaps init threadpool here
#         self.do_render = do_render
#
#     def __run(self, sim_world, sim_state, sim_action):
#         # maze doesnt need current state to simulate
#         # sim_world has initialized agent position
#         if sim_action == "up":
#             return sim_world.on_up(None)
#         elif sim_action == "left":
#             return sim_world.on_left(None)
#         elif sim_action == "down":
#             return sim_world.on_down(None)
#         elif sim_action == "right":
#             return sim_world.on_right(None)
#         else:
#             raise Exception("Unexpected snake game input...")
#
#     def sim(self, state, action):
#         init_x, init_y = self.get_x_y(state)
#         sim_world = snake.Application(self.do_render, init_x, init_y)
#         x = self.__run(sim_world, state, action)
#         sim_r, sim_n_s = self.__run(sim_world, state, action)
#         if self.do_render: sim_world.destroy()
#         # return values are: <orig_state, action, reward, new_state>
#         # print("Sim Result: ", state, action, sim_r, sim_n_s)
#         return state, action, sim_r, sim_n_s
#
#     def get_x_y(self, state):
#         return state[0], state[1]
#
#     def get_valid_actions(self, root, actions):
#         valid_actions = []
#         init_x, init_y = self.get_x_y(root)
#         for action in actions:
#             sim_world = snake.Application(self.do_render, init_x, init_y)
#             sim_r, sim_n_s = self.__run(sim_world, root, action)
#             if not list(sim_n_s) == list(root):
#                 valid_actions.append(action)
#         if self.do_render: sim_world.destroy()
#         return valid_actions
