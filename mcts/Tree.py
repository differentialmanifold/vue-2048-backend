import sys, math
from mcts.State import State


def utc_value(total_visit, node_win_score, node_visit):
    if node_visit == 0:
        return sys.maxsize
    return (node_win_score / node_visit) + 1.41 * math.sqrt(math.log(total_visit) / node_visit)


class Node:
    def __init__(self):
        self.state = State()
        self.parent = None
        self.children = []

    def find_best_node_with_utc(self):
        parent_visit = self.state.visit_count
        return max(self.children,
                   key=lambda item: utc_value(parent_visit, item.state.win_score, item.state.visit_count))


class Tree:
    def __init__(self):
        self.node = Node()
