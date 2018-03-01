# Pseudocode
#
# GrowSparseTree (node, branchfactor, horizon)
#     If node.depth = horizon; return
#     If node.type = “decision”
#     For each a ∈ A
#         child = (“outcome”, depth, node.belstate, a)
#         GrowSparseTree (child, branchfactor, horizon)
#     If node.type = “outcome”
#         For i = 1...branchfactor
#         [rew,obs] = sample(node.belstate, node.act)
#         post = posterior(node.belstate, obs)
#         child = (“decision”, depth+1, post, [rew,obs])
#         GrowSparseTree (child, branchfactor, horizon)
#
# EvaluateSubTree (node, horizon)
#     If node.children = empty
#         immed = MaxExpectedValue(node.belstate)
#         return immed * (horizon - node.depth)
#     If node.type = “decision”
#         return max(EvaluateSubTree(node.children))
#     If node.type = “outcome”
#         values = EvaluateSubTree(node.children)
#         return avg(node.rewards + values)

import enum
from mdpSimulator import MDPSimulator
import numpy as np

print_debug = False
NodeType = enum.Enum("NodeType", "Outcome Decision")


class SparseTree(object):

    class Node(object):
        def __init__(self, type, depth, state, value):
            self.type = type
            self.depth = depth
            self.state = state
            self.value = value

        def __str__(self):
            return "[" + str(self.type) + ":" + str(self.depth) + ":" + str(self.value) + "]"

    def __init__(self, node, parent):
        self.node = node
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def append_val_to_parent(self, value):
        self.parent.node.value.append(value)

    def __str__(self):
        children_str = "{"
        for child in self.children:
            children_str += " " + str(child.node)
        children_str += "}"
        return str(self.node) + " -> " + children_str

    def get_tree_size(self):
        if len(self.children) == 0:
            return 1
        return 1 + sum(map(lambda child: child.get_tree_size(), self.children))


class SparseTreeEvaluator(object):

        def __init__(self, mdp_simulator, root_state, action_set, horizon,
                     history_manager, thompson_sampler=None, discount_factor=0.05,
                     use_constant_boundary=0.5):
            if not isinstance(mdp_simulator, MDPSimulator):
                raise Exception('Sparse tree evaluator needs MDP Simulator!')
            self.simulator = mdp_simulator
            self.root_state = root_state
            self.action_set = action_set
            self.horizon = horizon
            self.lookahead_tree = None
            self.thompson_sampler = thompson_sampler
            self.discount_factor = discount_factor
            self.history_manager = history_manager
            self.use_constant_boundary = use_constant_boundary

        def evaluate(self):
            root_node = SparseTree.Node(NodeType.Decision, 0, self.root_state, [])
            lookahead_tree = SparseTree(root_node, None)
            self.__grow_sparse_tree(lookahead_tree)
            self.__eval_sparse_tree(lookahead_tree)
            self.lookahead_tree = lookahead_tree

        def __str__(self):
            children_str = "{"
            for child in self.lookahead_tree.children:
                children_str += " " + str(child.node)
            children_str += "}"
            return str(self.lookahead_tree.node) + " -> " + children_str

        def __grow_sparse_tree(self, lookahead_tree):
            if (lookahead_tree.node.depth >= self.horizon) and (lookahead_tree.node.type == NodeType.Decision):
                # leaves of sparse tree should be outcome nodes
                return

            if lookahead_tree.node.type == NodeType.Decision:
                for action in self.__get_actions(lookahead_tree):
                    orig_state, child_action, child_reward, child_state = \
                        self.simulator.sim(lookahead_tree.node.state, action)
                    if list(child_state) == list(orig_state):
                        continue
                    child = SparseTree(SparseTree.Node(NodeType.Outcome, lookahead_tree.node.depth,
                                                       child_state, [child_reward]), lookahead_tree)
                    lookahead_tree.add_child(child)
                    if print_debug: print("Added outcome child depth",  child)
                    self.__grow_sparse_tree(child)

            if lookahead_tree.node.type == NodeType.Outcome:
                for state in self.__get_states(lookahead_tree):
                    child = SparseTree(SparseTree.Node(NodeType.Decision, lookahead_tree.node.depth+1,
                                                       state, []), lookahead_tree)
                    lookahead_tree.add_child(child)
                    if print_debug: print("Added decision child depth", child)
                    self.__grow_sparse_tree(child)

        def __eval_sparse_tree(self, lookahead_tree):
            for child in lookahead_tree.children:
                self.__eval_sparse_tree(child)

            if lookahead_tree.node.type == NodeType.Outcome:
                reward_avg = sum(lookahead_tree.node.value) / float(len(lookahead_tree.node.value))
                if len(lookahead_tree.children) == 0:
                    depth_factor = max(self.horizon, lookahead_tree.node.depth) - lookahead_tree.node.depth + 1
                    lookahead_tree.append_val_to_parent(reward_avg * float(depth_factor) * self.discount_factor)
                else:
                    # average present and future rewards
                    lookahead_tree.append_val_to_parent(reward_avg * self.discount_factor)

            if lookahead_tree.node.type == NodeType.Decision:
                if lookahead_tree.node.depth == 0:
                    # set sparse tree root value to
                    # (best_action_index, max_avg_reward_value_discounted)
                    max_value = max(lookahead_tree.node.value)
                    lookahead_tree.node.value = ([i for i, j in
                                                 enumerate(lookahead_tree.node.value)
                                                 if j == max_value], max_value, [lookahead_tree.node.value])
                else:
                    # maximize the averages and discount the max
                    if len(lookahead_tree.node.value):
                        present_reward = max(lookahead_tree.node.value)
                        if len(lookahead_tree.children) == 0:
                            depth_factor = max(self.horizon,
                                               lookahead_tree.node.depth) - lookahead_tree.node.depth + 1
                            lookahead_tree.append_val_to_parent(present_reward *
                                                                float(depth_factor))
                        else:
                            lookahead_tree.append_val_to_parent(present_reward)

        def __get_actions(self, root):
            if self.thompson_sampler:
                valid_actions = self.simulator.get_valid_actions(root.node.state,
                                                                 self.action_set)
                return self.thompson_sampler.get_action_set(valid_actions)
            else:
                return self.action_set

        def __get_states(self, root):
            ## complete neighbor set
            neighbors = []
            rejected_neighbors = []
            for action in self.action_set:
                n_orig_state, n_action, n_reward, n_new_state = self.simulator.sim(root.node.state, action)
                if not list(n_new_state) == list(n_orig_state):
                    posterior_result = self.__posterior_state(n_new_state, len(self.action_set))
                    if posterior_result:
                        neighbors.append(n_new_state)
                    else:
                        rejected_neighbors.append(n_new_state)
            # if neighbors: return neighbors
            # return rejected_neighbors
            return neighbors

        def __posterior_state(self, state, n_neighbors):
            state_visit_count = self.history_manager.state_count_dict.get(state, 0)
            state_miss_count = sum(self.history_manager.state_count_dict.values()) - state_visit_count
            # alpha = state_miss_count + 0.1
            alpha = n_neighbors
            beta = (state_visit_count*n_neighbors) + 1
            # print("State:", state, "alpha:", alpha, "beta:", beta)
            state_prob = np.random.beta(alpha, beta)
            if self.use_constant_boundary:
                prob_threshold = self.use_constant_boundary
            else:
                # use float(alpha)/float(alpha+beta) for mean
                # otherwise median
                prob_threshold = (alpha - float(1) / float(3)) / (
                    alpha + beta - float(2) / float(3))

            if state_prob > prob_threshold:
                return state
            else:
                return None
