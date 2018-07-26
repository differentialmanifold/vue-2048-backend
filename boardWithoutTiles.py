import random
import numpy as np
from mcts.MonteCarloTreeSearch import MonteCarloTreeSearch


class Tile:
    def __init__(self, value=0, row=-1, column=-1):
        self.value = value
        self.row = row
        self.column = column


class Board:
    size = 4
    fourProbability = 0.1
    delta_x = [-1, 0, 1, 0]
    delta_y = [0, -1, 0, 1]

    def __init__(self):
        self.cells = [[self.add_tile() for _ in range(Board.size)] for _ in range(Board.size)]
        self.add_random_tile()
        self.set_positions()
        self.has_changed = True
        self.max_value = 0
        self.total_score = 0
        self.lost = False
        self.can_move_dir = [False, False, False, False]
        self.last_action = -1

    def copy(self):
        board_copy = Board()
        board_copy.cells = [[Tile(self.cells[i][j].value, i, j) for j in range(Board.size)] for i in range(Board.size)]
        board_copy.has_changed = self.has_changed
        board_copy.max_value = self.max_value
        board_copy.total_score = self.total_score
        board_copy.lost = self.lost
        board_copy.can_move_dir = self.can_move_dir[:]
        return board_copy

    def rotate_left(self):
        rows = len(self.cells)
        columns = len(self.cells[0])
        new_rows = columns
        new_columns = rows
        self.cells = [[self.cells[j][columns - i - 1] for j in range(new_columns)] for i in range(new_rows)]

    def add_tile(self, value=0):
        res = Tile(value)
        return res

    def move_left(self):
        has_changed = False
        for row in range(Board.size):
            current_row = [tile for tile in self.cells[row] if tile.value != 0]
            result_row = [None for _ in range(Board.size)]
            for target in range(Board.size):
                target_tile = current_row.pop(0) if len(current_row) > 0 else self.add_tile()
                if len(current_row) > 0 and current_row[0].value == target_tile.value:
                    target_tile = self.add_tile(target_tile.value)
                    tile2 = current_row.pop(0)
                    target_tile.value += tile2.value
                result_row[target] = target_tile
                has_changed |= (target_tile.value != self.cells[row][target].value)
            self.cells[row] = result_row
        return has_changed

    def set_positions(self):
        for i in range(Board.size):
            for j in range(Board.size):
                tile = self.cells[i][j]
                tile.row = i
                tile.column = j

    def add_random_tile(self):
        empty_cells = [{'row': i, 'column': j} for i in range(Board.size) for j in range(Board.size) if
                       self.cells[i][j].value == 0]
        _index = random.choice(range(len(empty_cells)))
        cell = empty_cells[_index]
        new_value = 4 if random.random() < Board.fourProbability else 2
        self.cells[cell['row']][cell['column']] = self.add_tile(new_value)

    def move(self, direction):
        # 0 -> left, 1 -> up, 2 -> right, 3 -> down
        for _ in range(direction):
            self.rotate_left()
        has_changed = self.move_left()
        for _ in range(direction, 4):
            self.rotate_left()
        if has_changed:
            self.add_random_tile()
        self.last_action = direction
        self.has_changed = has_changed
        self.set_positions()

    def can_swipe_left(self):
        """
        compare adjacent cells
        two condition can move:
        1. first cell is empty and second cell is not empty
        2. two cells not empty and value is equal
        """
        for row in range(Board.size):
            for column in range(Board.size - 1):
                first_value = self.cells[row][column].value
                second_value = self.cells[row][column + 1].value
                if first_value == 0 and second_value != 0:
                    return True
                if first_value != 0 and second_value != 0 and first_value == second_value:
                    return True
        return False

    def check_swipe_all_direction(self):
        # left, up, right, down
        for i in range(Board.size):
            self.can_move_dir[i] = self.can_swipe_left()
            self.rotate_left()

    def has_lost(self):
        return not any(self.can_move_dir)

    def has_done(self):
        return self.has_lost()

    def matrix(self):
        matrix_value = [[self.cells[row][column].value for column in range(Board.size)] for row in range(Board.size)]
        return np.array(matrix_value)

    def env_init(self):
        self.__init__()
        self.check_swipe_all_direction()
        _matrix = self.matrix()
        return _matrix, self.can_move_dir

    def step(self, _action):
        self.move(_action)

        self.check_swipe_all_direction()

        _matrix = self.matrix()
        _done = False

        if self.has_done():
            _done = True

        self.max_value = np.max(_matrix)

        self.total_score = np.sum(_matrix) / 500

        return _matrix, _done, self.max_value, self.total_score, self.can_move_dir


def print_matrix(_matrix):
    print('---')
    for i in range(4):
        print(_matrix[i])
    print('---')


if __name__ == "__main__":
    board = Board()
    mcts = MonteCarloTreeSearch()

    index = 0
    matrix, can_move_dir = board.env_init()

    print('****')
    print('index {}'.format(index))
    print_matrix(matrix)
    print('****')
    done = False

    while not done:
        # action = random.choice(range(4))
        action = mcts.find_next_move(board)
        matrix, done, value, score, can_move_dir = board.step(action)

        index += 1

        print('****')
        print('action {}'.format(action))
        print('index {}'.format(index))
        print_matrix(matrix)
        print('score {}'.format(score))
        print('done {}'.format(done))
        print('can move dir {}'.format(can_move_dir))
        print('****')
