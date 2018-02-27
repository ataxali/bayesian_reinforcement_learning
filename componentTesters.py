import world
import threading
import time

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
    root_state = [3, 0]
    action_set = ["up", "down", "left", "right"]
    horizon = 5
    branch_factor = 5
    ste = SparseTreeEvaluator(simulator, root_state, action_set, horizon, branch_factor)
    ste.evaluate()
    t1 = time.time()
    print(ste)
    print("Runtime:", t1-t0)

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

