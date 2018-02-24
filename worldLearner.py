import world
import threading
import time

discount = 0.3
actions = world.actions
states = []
Q = {}
for i in range(world.x):
    for j in range(world.y):
        states.append((i, j))

for state in states:
    temp = {}
    for action in actions:
        temp[action] = 0.1
        world.set_cell_score(state, action, temp[action])
    Q[state] = temp

for (i, j, c, w) in world.specials:
    for action in actions:
        Q[(i, j)][action] = w
        world.set_cell_score((i, j), action, w)


def do_action(action):
    s = world.player
    r = -world.score
    if action == actions[0]:
        world.try_move(0, -1)
    elif action == actions[1]:
        world.try_move(0, 1)
    elif action == actions[2]:
        world.try_move(-1, 0)
    elif action == actions[3]:
        world.try_move(1, 0)
    else:
        return
    s2 = world.player
    r += world.score
    return s, action, r, s2


def max_Q(s):
    val = None
    act = None
    for a, q in Q[s].items():
        if val is None or (q > val):
            val = q
            act = a
    return act, val


def inc_Q(s, a, alpha, inc):
    Q[s][a] *= 1 - alpha
    Q[s][a] += alpha * inc
    world.set_cell_score(s, a, Q[s][a])


def run():
    global discount
    time.sleep(1)
    alpha = 1
    t = 1
    while True:
        # Pick the right action
        s = world.player
        max_act, max_val = max_Q(s)
        (s, a, r, s2) = do_action(max_act)

        # Update Q
        max_act, max_val = max_Q(s2)
        inc_Q(s, a, alpha, r + discount * max_val)

        # Check if the game has restarted
        t += 1.0
        if world.has_restarted():
            world.restart_game()
            time.sleep(0.01)
            t = 1.0

        # Update the learning rate
        alpha = pow(t, -0.1)

        # MODIFY THIS SLEEP IF THE GAME IS GOING TOO FAST.
        time.sleep(0.1)


t = threading.Thread(target=run)
t.daemon = True
t.start()
world.start_game()