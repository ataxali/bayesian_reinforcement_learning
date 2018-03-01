import world
import threading
import time
import random

# def run():
#     pass
#
# t = threading.Thread(target=run)
# t.daemon = True
# t.start()
# world.start_sim(1, 3)

from mdpSimulator import WorldSimulator
from bayesSparse import SparseTreeEvaluator
from thompsonSampling import HistoryManager, BootstrapHistoryManager, ThompsonSampler

terminal_state_win = [world.static_specials[1][0], world.static_specials[1][1]]
terminal_state_loss = [world.static_specials[0][0], world.static_specials[0][1]]


def test_world_simulator():
    w = WorldSimulator()
    # specials at (4,0) and (4,1)
    print(w.sim([3, 3], "up")) # ([3, 3], 'up', -0.040000000000000036, (3, 2))
    print(w.sim([3, 2], "up")) # ([3, 2], 'up', -0.040000000000000036, (3, 1))
    print(w.sim([3, 1], "right")) # ([3, 1], 'right', -1.0, (4, 1))
    print(w.sim([3, 0], "right")) # ([3, 0], 'right', 1.0, (4, 0))
    print(w.sim([3, 2], "left")) # ([3, 0], 'right', 1.0, (4, 0))


def sparse_tree_tester():
    t0 = time.time()
    simulator = WorldSimulator(use_cache=True)
    root_state = [0, 4]
    action_set = ["up", "down", "left", "right"]
    print(simulator.get_valid_actions(root_state, action_set))

    simulator = WorldSimulator(use_cache=True)
    action_set = ["up", "down", "left", "right"]
    horizon = 6
    branch_factor = 5
    ste = SparseTreeEvaluator(simulator, root_state, action_set, horizon)
    ste.evaluate()
    print(ste)
    print(random.choice(ste.lookahead_tree.node.value[0]))
    t1 = time.time()
    print("Runtime:", t1-t0)


def thompson_sampler_tester():
    action_set = ["up", "down", "left", "right"]
    branching_factor = 2
    history = HistoryManager(action_set)
    #tsampler = ThompsonSampler(history, branching_factor)

    w = WorldSimulator()
    move_1 = w.sim([0, 4], "up")
    history.add(move_1)

    move_2 = w.sim([0, 3], "right")
    history.add(move_2)

    move_3 = w.sim([1, 3], "up")
    history.add(move_3)
    print(history.get_action_count_reward_dict())
    #print(tsampler.get_action_set())


def bootstrap_history_tester():
    action_set = ["up", "down", "left", "right"]
    branching_factor = 2
    history = BootstrapHistoryManager(action_set, 0.5)

    w = WorldSimulator()
    move_1 = w.sim([0, 4], "up")
    history.add(move_1)

    move_2 = w.sim([0, 3], "right")
    history.add(move_2)

    move_3 = w.sim([1, 3], "up")
    history.add(move_3)

    print(history.get_action_count_reward_dict())


def sparse_tree_model_tester():
    ###### Model Variables #####
    root_state = [0, 4]
    horizon = 5
    episode_length = 0  # number of moves before posterior distributions are reset
    action_set = ["up", "down", "left", "right"]
    history_manager = HistoryManager(action_set)
    # history_manager = BootstrapHistoryManager(action_set, 0.5) 
    # thompson_sampler = ThompsonSampler(history_manager, use_rewards=True, use_constant_boundary=0.5)
    thompson_sampler = None
    ############################

    t0 = time.time()
    original_root = root_state
    simulator = WorldSimulator(use_cache=True)
    prev_root = None
    total_move_count = 0
    episode_move_count = 0
    move_pool = []

    def eval_sparse_tree(sim, root_s, actions, horizon, tsampler=None):
        ste = SparseTreeEvaluator(sim, root_s, actions, horizon, tsampler)
        ste.evaluate()
        print(ste)
        optimal_action_index = random.choice(ste.lookahead_tree.node.value[0])
        possible_actions = list(sim.get_valid_actions(root_s, actions))
        print("Possible actions: ", possible_actions)
        optimal_action = possible_actions[optimal_action_index]
        print("Optimal action:", str(optimal_action), ":", optimal_action_index)
        print("Tree size: ", ste.lookahead_tree.get_tree_size())
        return optimal_action, optimal_action_index, possible_actions, ste

    while True:
        # check if end of training episode
        if episode_length and episode_move_count > 1 and episode_move_count % episode_length == 0:
            print('>> End of Training Episode <<')
            history_manager.reset_history()
            episode_move_count = 0

        print("Evaluating tree at ", root_state)
        optimal_action, optimal_action_index, possible_actions, ste = \
            eval_sparse_tree(simulator, root_state, action_set, horizon, thompson_sampler)
        orig_state, action, new_reward, new_state = simulator.sim(root_state, optimal_action)

        while list(new_state) == prev_root:
            # loop breaker
            print("Policy loop detected...")
            if len(possible_actions) == 0:
                raise Exception("Whoops, this is a really bad place")
            possible_actions.pop(optimal_action_index)
            optimal_action, optimal_action_index, possible_actions, ste = \
                eval_sparse_tree(simulator, root_state, possible_actions, horizon)
            orig_state, action, new_reward, new_state = simulator.sim(root_state, optimal_action)

        prev_root = root_state
        root_state = list(new_state)
        episode_move_count += 1
        total_move_count += 1
        print("Moving to ", root_state, "...")

        history_manager.add((orig_state, action, new_reward, new_state))

        move_pool.append(optimal_action)

        if root_state == terminal_state_win:
            print("Agent Won in ", total_move_count, " moves!")
            print("Time Taken: ", time.time()-t0)
            break
        if root_state == terminal_state_loss:
            print("Agent Lost in ", total_move_count, " moves!")
            print("Time Taken: ", time.time()-t0)
            break
    world.World(init_x=original_root[0], init_y=original_root[1], move_pool=move_pool)
sparse_tree_model_tester()



#bootstrap_history_tester()
#thompson_sampler_tester()
