from mcts.Tree import Tree, State


def select_promissing_node(node):
    pass


def expand_node(promising_node):
    pass


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

