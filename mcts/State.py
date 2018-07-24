class State:
    def __init__(self):
        self.board = None
        self.visit_count = 0
        self.win_score = 0
        self.legal = True

    def play(self, action):
        return self.board.step(action)
