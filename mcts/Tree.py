import math
import random
import sys

from mcts.State import State


def utc_value(total_visit, node_win_score, node_visit):
    if node_visit == 0:
        return sys.maxsize
    return (node_win_score / node_visit) + 1.41 * math.sqrt(math.log(total_visit) / node_visit)


class Node:
    def __init__(self, state=None):
        if state is None:
            state = State()
        self.state = state
        self.parent = None
        self.children = []

    def find_best_node_with_utc(self):
        parent_visit = self.state.visit_count
        return max(self.children,
                   key=lambda item: utc_value(parent_visit, item.state.win_score, item.state.visit_count))

    def get_random_child_node(self):
        child_node_size = len(self.children)
        if child_node_size == 0:
            return None

        index = random.choice(range(child_node_size))
        return self.children[index]

    def get_child_with_max_score(self):
        return max(self.children, key=lambda item: item.state.win_score)


class Tree:
    def __init__(self):
        self.node = Node()
