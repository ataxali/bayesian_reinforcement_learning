import world
import threading
import time


class MDPSimulator(object):
    """Abstract class for MDP Simulator"""
    def sim(self, state, action):
        """ Must return <orig_state, action, reward, new_state> tuple """
        raise NotImplementedError("Unimplemented method!")


class WorldSimulator(MDPSimulator):

    def __init__(self, do_render=False):
        # perhaps init threadpool here
        self.do_render = do_render

    def sim(self, state, action):

        def run(sim_world, sim_state, sim_action):
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

        init_x, init_y = self.get_x_y(state)
        sim_world = world.World(self.do_render, init_x, init_y)
        sim_r, sim_n_s = run(sim_world, state, action)
        if self.do_render: sim_world.destroy()
        # return values are: <orig_state, action, reward, new_state>
        # print("Sim Result: ", state, action, sim_r, sim_n_s)
        return state, action, sim_r, sim_n_s

    def get_x_y(self, state):
        return state[0], state[1]



