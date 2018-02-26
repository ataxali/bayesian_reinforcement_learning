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

NodeType = enum.Enum("NodeType", "Outcome Decision")


class SparseTree(object):

    class Node(object):
        def __init__(self, type, depth, state, value):
            self.type = type
            self.depth = depth
            self.state = state
            self.value = value

    def __init__(self, node, parent):
        self.node = node
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def append_val_to_parent(self, value):
        self.parent.node.value.append(value)


class SparseTreeEvaluator(object):

        def __init__(self, mdp_simulator, root_state, action_set, horizon, branch_factor):
            if not isinstance(mdp_simulator, MDPSimulator):
                raise Exception('Sparse tree evaluator needs MDP Simulator!')
            self.simulator = mdp_simulator
            self.root_state = root_state
            self.actionSet = action_set
            self.horizon = horizon
            self.branch_factor = branch_factor

        def evaluate(self):
            root_node = SparseTree.Node(NodeType.Decision, 0, self.root_state, [])
            lookahead_tree = SparseTree(root_node, None)
            self.__grow_sparse_tree(lookahead_tree)
            self.__eval_sparse_tree(lookahead_tree)

        def __grow_sparse_tree(self, lookahead_tree):
            if (lookahead_tree.node.depth >= self.horizon) and (lookahead_tree.node.type == NodeType.Decision):
                # leafs of sparse tree should be outcome nodes
                return

            if lookahead_tree.node.type == NodeType.Outcome:
                for action in self.__get_actions(lookahead_tree):
                    child_state, child_reward = self.simulator.sim(lookahead_tree.Node.state, action)
                    child = SparseTree(SparseTree.Node(NodeType.Outcome, lookahead_tree.Node.depth,
                                                       child_state, [child_reward]), lookahead_tree)
                    lookahead_tree.add_child(child)
                    self.__grow_sparse_tree(child)

            if lookahead_tree.node.type == NodeType.Decision:
                for state in self.__get_states(lookahead_tree):
                    child = SparseTree(SparseTree.Node(NodeType.Decision, lookahead_tree.Node.depth+1,
                                                       state, []), lookahead_tree)
                    lookahead_tree.add_child(child)
                    self.__grow_sparse_tree(child)

        def __eval_sparse_tree(self, lookahead_tree):
            for child in lookahead_tree.children:
                self.__eval_sparse_tree(child)

            if lookahead_tree.Node.type == NodeType.Outcome:
                # average present and future rewards
                reward_avg = sum(lookahead_tree.Node.value) / float(len(lookahead_tree.Node.value))
                lookahead_tree.append_val_to_parent(reward_avg)

            if lookahead_tree.Node.type == NodeType.Decision:
                if lookahead_tree.Node.depth == 0:
                    # set sparse tree root value to
                    # (best_action_index, max_avg_reward_value_discounted)
                    max_value = max(lookahead_tree.Node.value)
                    lookahead_tree.Node.value = ([i for i, j in
                                                 enumerate(lookahead_tree.Node.value)
                                                 if j == max_value], max_value)
                else:
                    # maximize the averages and discount the max
                    present_reward = max(lookahead_tree.Node.value) * float(self.horizon - lookahead_tree.Node.depth)
                    lookahead_tree.append_val_to_parent(present_reward)

        def __get_actions(self, root):
            return self.actionSet

        def __get_states(self, root):
            ## complete neighbor set
            neighbors = []
            for action in self.__get_actions(root):
                neighbor_state, neighbor_reward = self.simulator.sim(root.Node.state, action)
                neighbors.append(neighbor_state)

