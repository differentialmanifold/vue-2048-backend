import random


class Tile:
    tile_id = 0

    def __init__(self, value=0, row=-1, column=-1):
        self.value = value
        self.row = row
        self.column = column
        self.oldRow = -1
        self.oldColumn = -1
        self.markForDeletion = False
        self.mergedInto = None
        Tile.tile_id += 1
        self.id = Tile.tile_id
        self.is_new = None
        self.has_moved = None
        self.from_row = None
        self.from_column = None
        self.to_row = None
        self.to_column = None

    def set_properties(self):
        self.is_new = self.oldRow == -1 and self.mergedInto is None
        self.from_row = self.row if self.mergedInto is not None else self.oldRow
        self.from_column = self.column if self.mergedInto is not None else self.oldColumn
        self.to_row = self.mergedInto['row'] if self.mergedInto is not None else self.row
        self.to_column = self.mergedInto['column'] if self.mergedInto is not None else self.column
        self.has_moved = (self.from_row != -1 and
                          (self.from_row != self.to_row or self.from_column != self.to_column)) or (
                             self.mergedInto is not None)


class Board:
    size = 4
    fourProbability = 0.1
    delta_x = [-1, 0, 1, 0]
    delta_y = [0, -1, 0, 1]

    def __init__(self):
        Tile.tile_id = 0
        self.tiles = []
        self.cells = [[self.add_tile() for _ in range(Board.size)] for _ in range(Board.size)]
        self.add_random_tile()
        self.set_positions()
        self.set_tiles_properties()
        self.has_changed = True
        self.won = False

    def rotate_left(self):
        rows = len(self.cells)
        columns = len(self.cells[0])
        new_rows = columns
        new_columns = rows
        self.cells = [[self.cells[j][columns - i - 1] for j in range(new_columns)] for i in range(new_rows)]

    def add_tile(self, value=0):
        res = Tile(value)
        if value != 0:
            self.tiles.append(res)
        return res

    def move_left(self):
        has_changed = False
        for row in range(Board.size):
            current_row = [tile for tile in self.cells[row] if tile.value != 0]
            result_row = [None for _ in range(Board.size)]
            for target in range(Board.size):
                target_tile = current_row.pop(0) if len(current_row) > 0 else self.add_tile()
                if len(current_row) > 0 and current_row[0].value == target_tile.value:
                    tile1 = target_tile
                    target_tile = self.add_tile(target_tile.value)
                    tile1.mergedInto = target_tile.__dict__
                    tile2 = current_row.pop(0)
                    tile2.mergedInto = target_tile.__dict__
                    target_tile.value += tile2.value
                result_row[target] = target_tile
                self.won |= (target_tile.value == 2048)
                has_changed |= (target_tile.value != self.cells[row][target].value)
            self.cells[row] = result_row
        return has_changed

    def set_positions(self):
        for i in range(Board.size):
            for j in range(Board.size):
                tile = self.cells[i][j]
                tile.oldRow = tile.row
                tile.oldColumn = tile.column
                tile.row = i
                tile.column = j
                tile.markForDeletion = False

    def set_tiles_properties(self):
        for tile in self.tiles:
            tile.set_properties()

    def add_random_tile(self):
        emptyCells = [{'row': i, 'column': j} for i in range(Board.size) for j in range(Board.size) if
                      self.cells[i][j].value == 0]
        index = random.choice(range(len(emptyCells)))
        cell = emptyCells[index]
        newValue = 4 if random.random() < Board.fourProbability else 2
        self.cells[cell['row']][cell['column']] = self.add_tile(newValue)

    def clear_old_tiles(self):
        self.tiles = [tile for tile in self.tiles if tile.markForDeletion == False]
        for tile in self.tiles:
            tile.markForDeletion = True

    def move(self, direction):
        # 0 -> left, 1 -> up, 2 -> right, 3 -> down
        self.clear_old_tiles()
        for _ in range(direction):
            self.rotate_left()
        has_changed = self.move_left()
        for _ in range(direction, 4):
            self.rotate_left()
        if has_changed:
            self.add_random_tile()
        self.has_changed = has_changed
        self.set_positions()
        self.set_tiles_properties()

    def has_lost(self):
        can_move = False
        for row in range(Board.size):
            for column in range(Board.size):
                can_move |= (self.cells[row][column].value == 0)
                for direct in range(Board.size):
                    new_row = row + Board.delta_x[direct]
                    new_column = column + Board.delta_y[direct]
                    if new_row < 0 or new_row >= Board.size or new_column < 0 or new_column >= Board.size:
                        continue
                    can_move |= (self.cells[row][column].value == self.cells[new_row][new_column].value)
                    if can_move:
                        return not can_move
        return not can_move

    def has_done(self):
        return self.won or self.has_lost()

    def matrix(self):
        matrix = [[self.cells[row][column].value for column in range(Board.size)] for row in range(Board.size)]
        return matrix

    def front_call_obj(self):
        tiles = [tile.__dict__ for tile in self.tiles]
        cells = [[tile.__dict__ for tile in row] for row in self.cells]

        return {'tiles': tiles, 'cells': cells, 'hasChanged': self.has_changed, 'won': self.won,
                'done': self.has_done(), 'hasLost': self.has_lost()}


def print_matrix(matrix):
    print('---')
    for i in range(4):
        print(matrix[i])
    print('---')


if __name__ == "__main__":
    def test_lost():
        can_move = True
        for row in range(Board.size):
            for column in range(Board.size):
                can_move |= False

                for direct in range(Board.size):
                    can_move |= False
        print(can_move)


    test_lost()
