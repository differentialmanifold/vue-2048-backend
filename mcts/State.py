import random


class State:
    def __init__(self, board=None):
        self.board = board
        self.visit_count = 0
        self.win_score = 0

    def increment_visit(self):
        self.visit_count += 1

    def add_score(self, score):
        self.win_score += score

    def play(self, action):
        return self.board.step(action)

    def random_play(self):
        can_move_dir = self.board.can_move_dir

        legal_dir = []
        for i in range(len(can_move_dir)):
            if can_move_dir[i]:
                legal_dir.append(i)

        random_action = random.choice(legal_dir)

        return self.board.step(random_action)
