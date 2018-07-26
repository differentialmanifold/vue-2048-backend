from mcts.Tree import Tree, Node
from mcts.State import State


def select_promissing_node(root_node):
    node = root_node
    while len(node.children) > 0:
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
    state = node_to_explore.state
    new_state = State(state.board.copy())
    done = new_state.board.has_done()
    total_score = new_state.board.total_score

    while not done:
        matrix, done, max_value, total_score, _ = new_state.random_play()
    return total_score


def back_propogation(node_to_explore, playout_result):
    temp_node = node_to_explore
    while temp_node is not None:
        temp_node.state.increment_visit()
        temp_node.state.add_score(playout_result)
        temp_node = temp_node.parent


class MonteCarloTreeSearch:
    def find_next_move(self, board):
        tree = Tree()
        tree.node.state.board = board.copy()

        for i in range(1000):
            # print('index is : {}'.format(i))
            promising_node = select_promissing_node(tree.node)

            expand_node(promising_node)

            node_to_explore = promising_node
            if len(promising_node.children) > 0:
                node_to_explore = promising_node.get_random_child_node()
            playout_result = simulate_random_result(node_to_explore)

            back_propogation(node_to_explore, playout_result)

        best_node = tree.node.get_child_with_max_score()

        return best_node.state.board.last_action
