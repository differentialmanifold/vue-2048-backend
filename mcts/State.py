class State:
    def __init__(self, board=None):
        self.board = board
        self.visit_count = 0
        self.win_score = 0

    def play(self, action):
        return self.board.step(action)
