

class BoardStats:
    def __init__(board):
        board.tile_count = 0
        board.line_count = 0

    def tick_tile_count(self, lines=None):
        if lines:
            self.tile_count += lines
        else:
            self.tile_count += 1

    def tick_line_count(self):
        self.line_count += 1

    def get_line_count(self):
        return self.line_count
