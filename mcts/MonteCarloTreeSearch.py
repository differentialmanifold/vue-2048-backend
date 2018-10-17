import time
import os
import sys
from datetime import datetime

if "../" not in sys.path:
    sys.path.append("../")

from mcts.Tree import Tree, Node
from mcts.State import State
from boardWithoutTiles import BoardForTrain


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
        matrix, reward, done, max_value, total_score, _ = new_state.random_play()
    return total_score


def back_propogation(node_to_explore, playout_result):
    temp_node = node_to_explore
    while temp_node is not None:
        temp_node.state.increment_visit()
        temp_node.state.add_score(playout_result)
        temp_node = temp_node.parent


def print_matrix(_matrix):
    print('---')
    for i in range(4):
        print(_matrix[i])
    print('---')


class RecycleSolver:
    def __init__(self, recycle_type, value):
        self.recycle_type = recycle_type
        self.value = value
        self.current_milli_time = lambda: int(round(time.time() * 1000))
        self.start = self.current_milli_time()
        self.end = self.start + self.value
        self.count = 0

    def calculate(self):
        if self.recycle_type:
            return self.current_milli_time() < self.end
        self.count += 1
        return self.count <= self.value


class MonteCarloTreeSearch:
    def __init__(self, recycle_type=0, value=100):
        """
        init monte carlo rollout way
        :param recycle_type: 0 for rollout a fixed number, 1 for rollout a fixed range of time(ms)
        :param value: count value or time value(ms)
        """
        self.recycle_type = recycle_type
        self.value = value

    def find_next_move(self, matrix):
        tree = Tree()
        tree.node.state.board = BoardForTrain(matrix)
        recycle_solver = RecycleSolver(self.recycle_type, self.value)

        while recycle_solver.calculate():
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

    def complete_play(self):
        board = BoardForTrain()

        index = 0
        board.env_init()

        done = False

        while not done:
            action = self.find_next_move(board.matrix())
            matrix, reward, done, value, score, can_move_dir = board.step(action)

            index += 1

            print('****')
            print('action {}'.format(action))
            print('index {}'.format(index))
            print_matrix(matrix)
            print('score {}'.format(score))
            print('done {}'.format(done))
            print('can move dir {}'.format(can_move_dir))
            print('****')
        return value


if __name__ == "__main__":
    b = range(100, 2000, 300)

    current_path = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(current_path, 'statistic.txt')

    for time_interval in b:
        total = 0
        p512 = 0
        p1024 = 0
        p2048 = 0
        p4096 = 0
        for j in range(100):
            mcts1 = MonteCarloTreeSearch(1, time_interval)
            value = mcts1.complete_play()
            total += 1
            if value >= 512:
                p512 += 1
            if value >= 1024:
                p1024 += 1
            if value >= 2048:
                p2048 += 1
            if value >= 4096:
                p4096 += 1
        with open(file_path, 'a+') as f:
            f.write(
                '{} time {} p512 {} p1024 {} p2048 {} p4096 {} total {} \n'.format(str(datetime.now()), time_interval,
                                                                                   p512,
                                                                                   p1024, p2048, p4096, total))
