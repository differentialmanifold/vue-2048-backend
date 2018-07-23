from mcts.State import State


class Node:
    def __init__(self):
        self.state = State()
        self.parent = None
        self.children = []


class Tree:
    def __init__(self):
        self.node = Node()
