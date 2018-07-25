from mcts.Tree import Tree, Node
from mcts.State import State


def select_promissing_node(root_node):
    node = root_node
    while len(node.children > 0):
        node = node.find_best_node_with_utc()
    return node


def expand_node(promising_node):
    board = promising_node.state.board
    can_move_dir = board.can_move_dir

    states = []
    for i in range(len(can_move_dir)):
        if can_move_dir[i]:
            new_board = board.copy()
            new_board.step(i)
            state = State(new_board)
            states.append(state)

    for i in range(len(states)):
        new_node = Node(states[i])
        new_node.parent = promising_node
        promising_node.children.append(new_node)


def simulate_random_result(node_to_explore):
    pass


def back_propogation(node_to_explore, playout_result):
    pass


class MonteCarloTreeSearch:
    def find_next_move(self, board):
        tree = Tree()
        tree.node.state.board = board

        for _ in range(1000):
            promising_node = select_promissing_node(tree.node)

            expand_node(promising_node)

            node_to_explore = promising_node
            if len(promising_node.children) > 0:
                node_to_explore = promising_node.get_random_child_node()
            playout_result = simulate_random_result(node_to_explore)

            back_propogation(node_to_explore, playout_result)

        best_node = tree.node.get_child_with_max_score()

        return best_node.pre_action
