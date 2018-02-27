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
from thompsonSampling import HistoryManager, ThompsonSampler

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
    tsampler = ThompsonSampler(history, branching_factor)

    w = WorldSimulator()
    move_1 = w.sim([0, 4], "up")
    history.add(move_1)

    move_2 = w.sim([0, 3], "right")
    history.add(move_2)

    move_3 = w.sim([1, 3], "up")
    history.add(move_3)

    print(tsampler.get_action_set())


def sparse_tree_model_tester():
    root_state = [2, 6]
    action_set = ["up", "down", "left", "right"]
    simulator = WorldSimulator(use_cache=True)
    horizon = 5
    branch_factor = 5
    prev_root = None
    move_count = 0

    def eval_sparse_tree(sim, root_s, actions, horizon):
        ste = SparseTreeEvaluator(sim, root_s, actions, horizon)
        ste.evaluate()
        print(ste)
        optimal_action_index = random.choice(ste.lookahead_tree.node.value[0])
        possible_actions = list(sim.get_valid_actions(root_s, actions))
        print("Possible actions: ", possible_actions)
        optimal_action = possible_actions[optimal_action_index]
        print("Optimal action:", str(optimal_action), ":", optimal_action_index)
        return optimal_action, optimal_action_index, possible_actions, ste


    while True:
        print("Evaluating tree at ", root_state)
        optimal_action, optimal_action_index, possible_actions, ste = \
            eval_sparse_tree(simulator, root_state, action_set, horizon)
        _, _, _, new_state = simulator.sim(root_state, optimal_action)

        while list(new_state) == prev_root:
            # loop breaker
            print("Policy loop detected...")
            if len(possible_actions) == 0:
                raise Exception("Whoops, this is a really bad place")
            possible_actions.pop(optimal_action_index)
            optimal_action, optimal_action_index, possible_actions, ste = \
                eval_sparse_tree(simulator, root_state, possible_actions, horizon)
            _, _, _, new_state = simulator.sim(root_state, optimal_action)

        prev_root = root_state
        root_state = list(new_state)
        move_count += 1
        print("Moving to ", root_state, "...")

        if root_state == terminal_state_win:
            print("Agent Won in ", move_count, " moves!")
            break
        if root_state == terminal_state_loss:
            print("Agent Lost in ", move_count, " moves!")
            break


sparse_tree_model_tester()