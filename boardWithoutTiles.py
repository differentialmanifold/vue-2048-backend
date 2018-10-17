import random
import numpy as np

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


class TileForTrain:
    def __init__(self, value=0, row=-1, column=-1):
        self.value = value
        self.row = row
        self.column = column


class BoardForTrain:
    action_space = 4
    fourProbability = 0.1
    delta_x = [-1, 0, 1, 0]
    delta_y = [0, -1, 0, 1]

    def __init__(self, matrix=None, size=2):
        self.size = size

        if matrix is None:
            self.cells = [[self.add_tile() for _ in range(self.size)] for _ in range(self.size)]
            self.add_random_tile()
        else:
            self.cells = [[self.add_tile(matrix[i][j]) for j in range(self.size)] for i in
                          range(self.size)]

        self.set_positions()
        self.has_changed = True
        self.max_value = 0
        self.total_score = 0
        self.lost = False
        self.can_move_dir = self.check_swipe_all_direction()
        self.last_action = -1
        self.reward = 0
        self.num_status = 0

    def copy(self):
        board_copy = BoardForTrain()
        board_copy.cells = [[TileForTrain(self.cells[i][j].value, i, j) for j in range(self.size)] for i in
                            range(self.size)]
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
        res = TileForTrain(value)
        return res

    def move_left(self):
        has_changed = False
        for row in range(self.size):
            current_row = [tile for tile in self.cells[row] if tile.value != 0]
            result_row = [None for _ in range(self.size)]
            for target in range(self.size):
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
        for i in range(self.size):
            for j in range(self.size):
                tile = self.cells[i][j]
                tile.row = i
                tile.column = j

    def add_random_tile(self):
        empty_cells = [{'row': i, 'column': j} for i in range(self.size) for j in range(self.size) if
                       self.cells[i][j].value == 0]
        _index = random.choice(range(len(empty_cells)))
        cell = empty_cells[_index]
        new_value = 4 if random.random() < BoardForTrain.fourProbability else 2
        self.cells[cell['row']][cell['column']] = self.add_tile(new_value)
        self.reward = new_value

    def move(self, direction):
        self.reward = 0

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
        self.can_move_dir = self.check_swipe_all_direction()

    def can_swipe_left(self):
        """
        compare adjacent cells
        two condition can move:
        1. first cell is empty and second cell is not empty
        2. two cells not empty and value is equal
        """
        for row in range(self.size):
            for column in range(self.size - 1):
                first_value = self.cells[row][column].value
                second_value = self.cells[row][column + 1].value
                if first_value == 0 and second_value != 0:
                    return True
                if first_value != 0 and second_value != 0 and first_value == second_value:
                    return True
        return False

    def check_swipe_all_direction(self):
        can_move_dir = [False, False, False, False]
        # left, up, right, down
        for i in range(4):
            can_move_dir[i] = self.can_swipe_left()
            self.rotate_left()
        return can_move_dir

    def has_lost(self):
        return not any(self.can_move_dir)

    def has_done(self):
        return self.has_lost()

    def matrix(self):
        matrix_value = [[self.cells[row][column].value for column in range(self.size)] for row in
                        range(self.size)]
        return np.array(matrix_value)

    def env_init(self):
        self.__init__(size=self.size)
        _matrix = self.matrix()
        return _matrix, self.can_move_dir

    def reset(self):
        return self.env_init()

    def step(self, _action):
        self.move(_action)

        _matrix = self.matrix()
        _done = False

        if self.has_done():
            _done = True

        self.max_value = np.max(_matrix)

        self.total_score = np.sum(_matrix) / 500

        return _matrix, self.reward, _done, self.max_value, self.total_score, self.can_move_dir

    def createTransitionProbability(self):
        P = {}

        shape_length = self.power(self.size, 2)
        shape_value = shape_length + 2
        shape = []
        for i in range(shape_length):
            shape.append(shape_value)

        nS = np.prod(shape)
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            multi_index = it.multi_index
            P[multi_index] = {a: [] for a in range(4)}

            matrix_value = self.transferTupleToMatrix(multi_index)

            for direction in [LEFT, UP, RIGHT, DOWN]:
                board_for_train = BoardForTrain(matrix_value)
                board_for_train.reward = 0

                for _ in range(direction):
                    board_for_train.rotate_left()
                has_changed = board_for_train.move_left()
                for _ in range(direction, 4):
                    board_for_train.rotate_left()

                moved_matrix = board_for_train.matrix()
                if has_changed:
                    empty_cells = [{'row': i, 'column': j} for i in range(self.size) for j in
                                   range(self.size) if
                                   board_for_train.cells[i][j].value == 0]

                    for _index in range(len(empty_cells)):
                        for new_value, probability in [(4, 0.1), (2, 0.9)]:
                            board_for_train = BoardForTrain(moved_matrix)
                            cell = empty_cells[_index]
                            board_for_train.cells[cell['row']][cell['column']] = board_for_train.add_tile(new_value)
                            board_for_train.reward = new_value
                            board_for_train.set_positions()
                            board_for_train.can_move_dir = board_for_train.check_swipe_all_direction()
                            P[multi_index][direction].append(
                                (probability / len(empty_cells), self.transferMatrixToTuple(board_for_train.matrix()),
                                 board_for_train.reward,
                                 board_for_train.has_done()))
                else:
                    board_for_train.set_positions()
                    board_for_train.can_move_dir = board_for_train.check_swipe_all_direction()
                    P[multi_index][direction].append(
                        (0.0, self.transferMatrixToTuple(moved_matrix), board_for_train.reward,
                         board_for_train.has_done()))

            it.iternext()
        return P

    def power(self, a, b):
        result = 1
        for i in range(b):
            result = result * a
        return result

    def log(self, sub, value):
        if value == 0:
            return 0
        result = 0
        cal_value = 1
        while cal_value < value:
            result = result + 1
            cal_value = cal_value * sub
        if cal_value != value:
            return -1
        return result

    def transferTupleToMatrix(self, tuple_value):
        if len(tuple_value) != self.power(self.size, 2):
            raise ValueError("tuple not match matrix")

        matrix_value = np.zeros(shape=[self.size, self.size])

        for i in range(self.size):
            for j in range(self.size):
                tuple_item = tuple_value[self.size * i + j]
                if tuple_item != 0:
                    matrix_value[i][j] = self.power(2, tuple_item)
        return matrix_value

    def transferMatrixToTuple(self, matrix_value):
        tuple_arr = np.zeros(shape=matrix_value.size)
        for i in range(self.size):
            for j in range(self.size):
                tuple_arr[self.size * i + j] = self.log(2, matrix_value[i][j])
        return tuple(tuple_arr)


if __name__ == "__main__":
    board = BoardForTrain()
    print(board.matrix())
    board.move(1)
    print(board.matrix())
    board.move(2)
    print(board.matrix())
    new_board = BoardForTrain(board.matrix())
    print(new_board.matrix())

    trainsition_probability = new_board.createTransitionProbability()
    print(trainsition_probability)
