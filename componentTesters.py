import world
import threading
import time
import random
import inputReader
import logger
import numpy as np
from mdpSimulator import WorldSimulator
from bayesSparse import SparseTreeEvaluator
from historyManager import HistoryManager, BootstrapHistoryManager
from thompsonSampling import ThompsonSampler
from gpPosterior import GPPosterior
from sklearn.gaussian_process.kernels import ExpSineSquared
from matplotlib import pyplot as plt, colors
import pickle
import sys


terminal_state_win = [world.static_specials[2][0], world.static_specials[2][1]]
terminal_state_loss_0 = [world.static_specials[0][0], world.static_specials[0][1]]
terminal_state_loss_1 = [world.static_specials[1][0], world.static_specials[1][1]]


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
    simulator = WorldSimulator()
    root_state = [0, 4]
    action_set = ["up", "down", "left", "right"]
    print(simulator.get_valid_actions(root_state, action_set, specials=[], walls=[]))

    simulator = WorldSimulator()
    action_set = ["up", "down", "left", "right"]
    horizon = 6
    branch_factor = 5
    history_manager = HistoryManager(action_set)
    ste = SparseTreeEvaluator(simulator, root_state, action_set, history_manager, horizon)
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
    history = BootstrapHistoryManager(action_set, 0.5)

    w = WorldSimulator()
    move_1 = w.sim([0, 4], "up")
    history.add(move_1 + (1,))

    move_2 = w.sim([0, 3], "right")
    history.add(move_2 + (2, ))

    move_3 = w.sim([1, 3], "up")
    history.add(move_3 + (3, ))

    print(history.history)


def gp_posterior_tester(log):
    origin_state = [6, 6]
    root_state = origin_state
    time = 0
    action_set = ["up", "down", "left", "right"]
    orig_specials = world.static_specials.copy()
    orig_walls = world.static_walls.copy()
    simulator = WorldSimulator()
    history_manager = HistoryManager(action_set)
    kernel = ExpSineSquared(length_scale=1, periodicity=1.0,
                            periodicity_bounds=(2, 100),
                            length_scale_bounds=(1, 50))
    gp = GPPosterior(history_manager=history_manager, kernel=kernel, log=None)
    # warmup no logging, just gp training
    # for i in range(1000):
    #     next_move = np.random.choice(action_set)
    #     state, action, sim_r, sim_n_s = simulator.sim(root_state, next_move)
    #     history_manager.add((root_state, action, sim_r, sim_n_s, time))
    #     root_state = sim_n_s
    #     time += 1
    #     if abs(sim_r) > 1:
    #         print("Restarting game", sim_r, time)
    #         root_state = origin_state
    #         time = 0
    #         simulator.specials = orig_specials.copy()
    #         gp.update_posterior()
    # time = 0
    #
    # t = np.atleast_2d(np.linspace(0, 1000, 1000)).T
    # x_preds, y_preds = gp.predict(t)
    # cmap_x = ['m', 'c', 'k', 'g']
    # cmap_y = ['r', 'b', 'y', 'teal']
    # total_x_obs = 0
    # total_y_obs = 0
    # for obs in gp.x_obs:
    #     total_x_obs += len(obs)
    # for obs in gp.y_obs:
    #     total_y_obs += len(obs)
    # print(">>> There are " + str(len(x_preds[0])) + " X gaussian procs for " + str(total_x_obs) + " obs <<<")
    # print(">>> There are " + str(len(y_preds[0])) + " Y gaussian procs for " + str(total_y_obs) + " obs <<<")
    # for i, preds in enumerate(x_preds[0]):
    #     plt.plot(t, preds, cmap_x[i]+":", label='x_predictions')
    #     plt.plot(list(map(lambda x: x[0], gp.x_obs[i])),
    #              list(map(lambda x: x[1], gp.x_obs[i])), cmap_x[i]+"*", markersize=10)
    # for i, preds in enumerate(y_preds[0]):
    #     plt.plot(t, preds, cmap_y[i]+":", label='y_predictions')
    #     plt.plot(list(map(lambda y: y[0], gp.y_obs[i])),
    #              list(map(lambda y: y[1], gp.y_obs[i])), cmap_y[i]+"*", markersize=10)
    #     plt.xlim(0, 100)
    # plt.show(block=True)
    # end of warmup

    def predict(time, type):
        x_preds, y_preds = gp.predict(time)
        for x_pred in x_preds[0]:
            for y_pred in y_preds[0]:
                msg = "add" + type + str(int(round(x_pred[0]))) + "," + str(
                    int(round(y_pred[0])))
                logger.log(msg, logger=log)

    for i in range(1000):
        next_move = np.random.choice(action_set)
        state, action, sim_r, sim_n_s, ns = simulator.sim(root_state, next_move, specials=orig_specials, walls=orig_walls)
        orig_specials = ns
        if list(sim_n_s) == list(state):
            if state not in gp.static_states:
                if next_move == "up":
                    logger.log('addw' + str(sim_n_s[0]) + "," + str(sim_n_s[1] - 1), logger=log)
                    gp.update_static_states([sim_n_s[0], sim_n_s[1] - 1])
                elif next_move == "down":
                    logger.log('addw' + str(sim_n_s[0]) + "," + str(sim_n_s[1] + 1),logger=log)
                    gp.update_static_states([sim_n_s[0], sim_n_s[1] + 1])
                elif next_move == "left":
                    logger.log('addw' + str(sim_n_s[0] - 1) + "," + str(sim_n_s[1]),logger=log)
                    gp.update_static_states([sim_n_s[0] - 1, sim_n_s[1]])
                elif next_move == "right":
                    logger.log('addw' + str(sim_n_s[0] + 1) + "," + str(sim_n_s[1]),logger=log)
                    gp.update_static_states([sim_n_s[0] + 1, sim_n_s[1]])
        history_manager.add((root_state, action, sim_r, sim_n_s, time))
        logger.log('clr', logger=log)
        predict(time - 1, "c")
        predict(time + 1, "c")
        predict(time, "r")
        logger.log(next_move, logger=log)
        root_state = sim_n_s
        time += 1
        if abs(sim_r) > 1:
            print("Restarting game", sim_r, time)
            root_state = origin_state
            time = 0
            # simulator.specials = orig_specials.copy()
            logger.log("reset", logger=log)
            gp.update_posterior()

    with open('gp.out', 'wb') as output:
        pickle.dump(gp, output, pickle.HIGHEST_PROTOCOL)


def plot_gp(filename):
    gp = pickle.load(open(filename, "rb"))
    kernel = ExpSineSquared(length_scale=2, periodicity=3.0,
                            periodicity_bounds=(2, 10),
                            length_scale_bounds=(1, 10))
    #gp = GPPosterior(history_manager=gp_orig.history_manager, kernel=gp_orig.kernel, log=None)
    #gp.update_posterior()
    #gp.static_states = gp_orig.static_states
    t = np.atleast_2d(np.linspace(0, 50, 50)).T
    x_preds, y_preds = gp.predict(t)
    cmap_x = ['m', 'c', 'k', 'g']
    cmap_y = ['r', 'b', 'y', 'k']
    total_x_obs = 0
    total_y_obs = 0
    for obs in gp.x_obs:
        total_x_obs += len(obs)
    for obs in gp.y_obs:
        total_y_obs += len(obs)
    print(">>> There are " + str(len(x_preds[0])) + " X gaussian procs for " + str(total_x_obs) + " obs <<<")
    print(">>> There are " + str(len(y_preds[0])) + " Y gaussian procs for " + str(total_y_obs) + " obs <<<")
    #for i, preds in enumerate(x_preds[0]):
    #    plt.plot(t, preds, cmap_x[i]+":", label='x_predictions')
    #    plt.plot(list(map(lambda x: x[0], gp.x_obs[i])),
    #             list(map(lambda x: x[1], gp.x_obs[i])), cmap_x[i]+"*", markersize=10)
    #    plt.fill(np.concatenate([t, t[::-1]]), np.concatenate([preds - 1.9600 * x_preds[1][i],
    #                                                           (preds + 1.9600 * x_preds[1][i])[::-1]]),
    #             alpha=.25, fc='k', ec='None', label='95% confidence interval')
    for i, preds in enumerate(y_preds[0]):
        plt.plot(t, preds, cmap_y[i]+":", label='y_predictions')
        plt.plot(list(map(lambda y: y[0], gp.y_obs[i])),
                 list(map(lambda y: y[1], gp.y_obs[i])), cmap_y[i]+"*", markersize=10)
        plt.fill(np.concatenate([t, t[::-1]]), np.concatenate([preds - 1.9600 * y_preds[1][i],
                                                               (preds + 1.9600 * y_preds[1][i])[::-1]]),
                 alpha=.25, fc='k', ec='None', label='95% confidence interval')
    plt.show(block=True)


def sparse_tree_model_tester(arg_dict):
    ###### Model Variables #####
    root_state = [0, 3]
    goal_state = [9, 6]
    goal_reward = 10
    loss_penalty = -10
    original_root = root_state.copy()
    horizon = 10
    if 'ep_len' in arg_dict and int(arg_dict['ep_len']):
        print("Setting episode length:", arg_dict['ep_len'], "...")
        episode_length = int(arg_dict['ep_len'])
    else:
        episode_length = 0  # number of games before posterior distributions are reset
    action_set = ["up", "down", "left", "right"]
    episode_move_limit = 100
    history_manager = HistoryManager(action_set)
    if 'bootstrap' in arg_dict:
        print("Setting history manager to Bootstrapped...")
        history_manager = BootstrapHistoryManager(action_set, 0.25)
    if episode_length:
        ts_history_manager = HistoryManager(action_set)
    else:
        ts_history_manager = history_manager
    thompson_sampler = None
    if 'prune' in arg_dict:
        move_wght = float(arg_dict['move_weight'])
        if not move_wght:
            raise Exception("Cannot start thompson sampler without move weight!")
        print("Creating thompson sampler, with move weight", move_wght, "...")
        
        thompson_sampler = ThompsonSampler(ts_history_manager, use_constant_boundary=0.5,
                                           move_weight=move_wght, move_discount=0.5,
                                           num_dirch_samples=100)
    discount_factor = 0.5
    ############################
    batch_id = arg_dict['batch_id']
    test_name = arg_dict['name']
    move_limit = int(arg_dict['move_limit'])
    root_path = arg_dict['root_path']
    simulator = WorldSimulator()
    true_specials = world.static_specials.copy()
    true_walls = world.static_walls.copy()
    total_move_count = 0
    game_move_count = 0
    episode_count = 0
    running_score = 0
    log = None
    #log = logger.DataLogger("./input_test_ts_es1.txt", replace=True)
    kernel = ExpSineSquared(length_scale=2, periodicity=3.0,
                            periodicity_bounds=(2, 10),
                            length_scale_bounds=(1, 10))
    gp = GPPosterior(history_manager=history_manager, kernel=kernel, log=None)
    ############################
    # used for testing purposes
    ############################
    # gp = pickle.load(open("gp_ts_new2.out", "rb"))
    #history_manager = pickle.load(open("hm_ts_es10.out", "rb"))
    #history_manager.history = gp.history_manager.history
    #history_manager.action_count_reward_dict = gp.history_manager.action_count_reward_dict
    #history_manager.state_count_dict = gp.history_manager.state_count_dict
    #history_manager.total_rewards = gp.history_manager.total_rewards
    #history_manager.action_set = gp.history_manager.action_set
    #history_manager = gp.history_manager
    #gp = GPPosterior(history_manager=gp_orig.history_manager, kernel=kernel, log=None)
    #gp.update_posterior()
    #gp.history_manager = history_manager

    def eval_sparse_tree(sim, root_s, actions, horizon, tsampler=None):
        ste = SparseTreeEvaluator(sim, root_s, actions, horizon,
                                  history_manager=history_manager,
                                  thompson_sampler=tsampler,
                                  discount_factor=discount_factor,
                                  state_posterior=gp,
                                  goal_state=goal_state,
                                  goal_reward=goal_reward,
                                  loss_penalty=loss_penalty)
        ste.evaluate(game_move_count)
        print(ste)
        optimal_action_index = random.choice(ste.lookahead_tree.node.value[0])
        possible_actions = ste.lookahead_tree.actions
        print("Possible actions: ", possible_actions)
        optimal_action = possible_actions[optimal_action_index]
        print("Optimal action:", str(optimal_action), ":", optimal_action_index)
        print("Tree size: ", ste.lookahead_tree.get_tree_size())
        return optimal_action, optimal_action_index, possible_actions, ste

    while True:
        print("Evaluating tree at ", root_state)
        # belief based
        optimal_action, optimal_action_index, possible_actions, ste = \
            eval_sparse_tree(simulator, root_state, action_set, horizon, thompson_sampler)
        # real world
        orig_state, action, new_reward, new_state, new_specials = simulator.sim(root_state, optimal_action,
                                                                  specials=true_specials,
                                                                  walls=true_walls)
        # record change in true specials after move
        true_specials = new_specials

        # prev_root = root_state.copy()
        root_state = list(new_state)
        print("Moving to ", root_state, "...")
        print("Move count:", total_move_count)
        print("Game move count:", game_move_count)

        history_manager.add((orig_state, action, new_reward, new_state, game_move_count))
        if episode_length:
            ts_history_manager.add((orig_state, action, new_reward, new_state, game_move_count))

        running_score += new_reward
        print("Score:", running_score)

        # check for walls
        if list(new_state) == list(orig_state):
            if tuple(new_state) not in gp.static_states:
                if action == "up":
                    logger.log(
                        'addw' + str(new_state[0]) + "," + str(new_state[1] - 1),
                        logger=log)
                    gp.update_static_states([new_state[0], new_state[1] - 1])
                elif action == "down":
                    logger.log(
                        'addw' + str(new_state[0]) + "," + str(new_state[1] + 1),
                        logger=log)
                    gp.update_static_states([new_state[0], new_state[1] + 1])
                elif action == "left":
                    logger.log(
                        'addw' + str(new_state[0] - 1) + "," + str(new_state[1]),
                        logger=log)
                    gp.update_static_states([new_state[0] - 1, new_state[1]])
                elif action == "right":
                    logger.log(
                        'addw' + str(new_state[0] + 1) + "," + str(new_state[1]),
                        logger=log)
                    gp.update_static_states([new_state[0] + 1, new_state[1]])

        # update belief game
        def predict(time, type):
            x_preds, y_preds = gp.predict(time)
            for x_pred in x_preds[0]:
                for y_pred in y_preds[0]:
                    if [int(round(x_pred[0])), int(round(y_pred[0]))] in ste.ignored_specials:
                        print("Ignoring special for belief world", int(round(x_pred[0])), int(round(y_pred[0])))
                    else:
                        msg = "add" + type + str(int(round(x_pred[0]))) + "," + str(int(round(y_pred[0])))
                        logger.log(msg, logger=log)

        logger.log('clr', logger=log)
        predict(game_move_count - 1, "c")
        predict(game_move_count + 1, "c")
        predict(game_move_count, "r")
        logger.log(action, logger=log)

        total_move_count += 1
        game_move_count += 1

        if total_move_count == move_limit:
            with open(root_path + "/" + test_name + batch_id + '.out', 'wb') as output:
                pickle.dump(gp, output, pickle.HIGHEST_PROTOCOL)
            return

        # check terminal conditions
        #if abs(new_reward) > 1 or (episode_length and (game_move_count > episode_move_limit)):
        if abs(new_reward) > 1 or (game_move_count > episode_move_limit):
            episode_count += 1
            if new_reward > 0:
                print("Agent Won in ", game_move_count, " moves!")
                sys.stdout.flush()
            print("Restarting game", new_reward, game_move_count)
            root_state = original_root.copy()
            true_specials = world.static_specials.copy()
            #if not (episode_length and (game_move_count > episode_move_limit)):
            if not (game_move_count > episode_move_limit):
                gp.update_posterior()
            game_move_count = 0
            logger.log("reset", logger=log)
            # check if end of training episode
            if episode_length and episode_count >= 1 and episode_count % episode_length == 0:
                print('>> End of Episode <<')
                ts_history_manager.reset_history()
                episode_count = 0

        sys.stdout.flush()

                #if thompson_sampler:
        #    with open('ts_test_ts_es1_bs.out', 'wb') as output:
        #        pickle.dump(thompson_sampler, output, pickle.HIGHEST_PROTOCOL)

        #if episode_length:
        #    with open('ts_hm_test_ts_es1_bs.out', 'wb') as output:
        #        pickle.dump(ts_history_manager, output, pickle.HIGHEST_PROTOCOL)

        #with open('hm_test_ts_es1_bs.out', 'wb') as output:
        #    pickle.dump(history_manager, output, pickle.HIGHEST_PROTOCOL)


#############
# gp tester #
#############
#fake_history_logger = logger.DataLogger("./fake_history.txt", replace=True)
#gp_posterior_tester(fake_history_logger)


##################
# move re-player #
##################
def launch_belief_world():
    world.World(init_x=0, init_y=6, input_reader=key_handler, specials=[(9, 0, "green", 10, "NA")],
         do_belief=True, walls=[])

def launch_real_world():
    world.World(init_x=0, init_y=6, input_reader=key_handler)

#log = logger.ConsoleLogger()
#key_handler = inputReader.KeyInputHandler(log)
#file_tailer = inputReader.FileTailer("./complete_models/input_test_ts_es1.txt", key_handler, log)
#t = threading.Thread(target=launch_belief_world)
#t.daemon = True
#t.start()


plot_gp("bayes_opt4.out.")

arg_dict = dict()
args = sys.argv
for arg in args:
    if "=" in arg:
        arg_dict[arg.split("=")[0]] = arg.split("=")[1]

sparse_tree_model_tester(arg_dict)


