import world
import random
import logger
from mdpSimulator import WorldSimulator
from bayesSparse import SparseTreeEvaluator
from historyManager import HistoryManager, BootstrapHistoryManager
from thompsonSampling import ThompsonSampler
from gpPosterior import GPPosterior
from sklearn.gaussian_process.kernels import ExpSineSquared
import pickle
import sys
import os


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
    is_testing = False
    if arg_dict["testing_file"]:
        is_testing = True

    def update_world_root(new_root):
        world.static_specials[4] = (new_root[0], new_root[1], "green", 10, "NA")

    def new_goal_state():
        new_y = random.randint(0, 6)
        while new_y == 5:
            new_y = random.randint(0, 6)
        nonlocal goal_state
        goal_state = [9, new_y]
        print(">> New Goal State:", goal_state, "<<")
        update_world_root(goal_state)

    if is_testing:
        new_goal_state()

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
    kernel = ExpSineSquared(length_scale=2, periodicity=3.0,
                            periodicity_bounds=(2, 10),
                            length_scale_bounds=(1, 10))
    gp = GPPosterior(history_manager=history_manager, kernel=kernel, log=None)
    ############################
    # used for testing purposes
    ############################
    if is_testing:
        gp = pickle.load(open(arg_dict["testing_file"], "rb"))
        history_manager.history = gp.history_manager.history
        history_manager.action_count_reward_dict = gp.history_manager.action_count_reward_dict
        history_manager.state_count_dict = gp.history_manager.state_count_dict
        history_manager.total_rewards = gp.history_manager.total_rewards
        history_manager.action_set = gp.history_manager.action_set
        print(">> Loaded trained model from," + arg_dict["testing_file"] + "<<")


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
            if not is_testing:
                with open(root_path + "/" + test_name + batch_id + '.out', 'wb') as output:
                    pickle.dump(gp, output, pickle.HIGHEST_PROTOCOL)
            return

        # check terminal conditions
        if abs(new_reward) > 1 or (game_move_count > episode_move_limit):
            episode_count += 1
            if new_reward > 0:
                print("Agent Won in ", game_move_count, " moves!")
                sys.stdout.flush()
            print("Restarting game", new_reward, game_move_count)
            if is_testing:
                new_goal_state()
            root_state = original_root.copy()
            true_specials = world.static_specials.copy()
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

###############################
# Required script parameters  #
###############################
# name (string)
# batch_id (int)
# move_limit (int)
# root_path (directory path)

###############################
# Optional script parameters  #
###############################
# prune (T/F)
# bootstrap (T/F)
# ep_len (int)
# testing (directory path)

arg_dict = dict()
args = sys.argv
for arg in args:
    if "=" in arg:
        arg_dict[arg.split("=")[0]] = arg.split("=")[1]

if arg_dict["testing"]:
    for filename in os.listdir(arg_dict["testing"]+"\\"):
        arg_dict["testing_file"] = arg_dict["testing"] + "\\" +filename
        sparse_tree_model_tester(arg_dict)
else:
    sparse_tree_model_tester(arg_dict)