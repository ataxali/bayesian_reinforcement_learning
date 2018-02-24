import numpy as np
import logger
import threading
from input_reader import KeyInputHandler
from tile import TileFactory
from boardStats import BoardStats


class Game(threading.Thread, BoardStats):

    def __init__(self, nrows, ncols, log, input, prob_override=None, rng_seed=None):
        threading.Thread.__init__(self)
        BoardStats.__init__(self)
        self.board = np.zeros((nrows, ncols))
        self.nrows = nrows
        self.ncols = ncols
        self.log = log
        self.input = input
        self.alive = True
        self.listeners = []

        self.game_state = 0  # 0=waiting for game, 1=waiting for input
        self.tile = None
        self.tile_origin = None
        self.tile_orientation = None

        if prob_override is not None:
            self.tile_generator = TileFactory(log, probs=prob_override, seed=rng_seed)
        else:
            self.tile_generator = TileFactory(log)
        logger.log("Tetris Board Created...", logger.Level.INFO, self.log)
        # start thread in ctor
        self.start()

    def run(self):
        while self.alive:
            self.__tick()
            self.__update_score()
            self.__update_listeners()

    def kill(self):
        self.alive = False

    def register_listener(self, listener):
        self.listeners.append(listener)
        listener.update(self.board)

    def __tick(self):
        if self.game_state == 0:
            new_tile = self.tile_generator.get_next_tile()
            new_tile_origin = (0, (self.ncols // 2) - 1)
            new_tile_orientation = 0
            BoardStats.tick_tile_count(self)
            self.__update_board(new_tile, new_tile_origin, new_tile_orientation, True, 1)
        elif self.game_state == 1:
            next_key = self.input.get_next_key()
            new_tile = self.tile
            new_tile_origin = self.tile_origin
            new_tile_orientation = self.tile_orientation
            if next_key is KeyInputHandler.keys.PASS:
                new_tile_origin = (min([self.tile_origin[0] + 1, self.nrows - 1]), self.tile_origin[1])
            elif next_key is KeyInputHandler.keys.ROTATE_R:
                new_tile_orientation = (self.tile_orientation + 1) % 4
            elif next_key is KeyInputHandler.keys.ROTATE_L:
                new_tile_orientation= (self.tile_orientation + 3) % 4
            elif next_key is KeyInputHandler.keys.LEFT:
                new_tile_origin = (self.tile_origin[0], max([self.tile_origin[1] - 1, 0]))
            elif next_key is KeyInputHandler.keys.RIGHT:
                new_tile_origin = (self.tile_origin[0], min([self.tile_origin[1] + 1, self.ncols - 1]))
            else:
                pass

            if next_key is KeyInputHandler.keys.DOWN:
                new_tile_origin = (min([self.tile_origin[0] + 1, self.nrows - 1]), self.tile_origin[1])

                while self.__is_valid_move(new_tile, new_tile_origin,new_tile_orientation):
                    self.__update_board(new_tile, new_tile_origin, new_tile_orientation, True, 1)
                    new_tile_origin = (min([new_tile_origin[0] + 1, self.nrows - 1]), new_tile_origin[1])

                new_tile_origin = (new_tile_origin[0] - 1, new_tile_origin[1])
                self.__update_board(new_tile, new_tile_origin, new_tile_orientation, False, 0)
                self.tile = None
            elif self.__is_valid_move(new_tile, new_tile_origin, new_tile_orientation):
                self.__update_board(new_tile, new_tile_origin, new_tile_orientation, True, 1)
            else:
                if next_key is KeyInputHandler.keys.PASS:
                    self.game_state = 0
                    self.tile = None
                else:
                    self.game_state = 1

    def __update_board(self, tile, origin, orientation, is_active=True, game_state=1):
        if self.tile is not None:
            previous_cells = self.__get_indices(self.tile.get_coords(self.tile_orientation), self.tile_origin)
            for cell in previous_cells:
                self.board[cell[0], cell[1]] = 0

        new_cells = self.__get_indices(tile.get_coords(orientation), origin)
        cell_val = 2 if is_active else 1
        for cell in new_cells:
            self.board[cell[0], cell[1]] = cell_val

        self.tile = tile
        self.tile_origin = origin
        self.tile_orientation = orientation
        self.game_state = game_state

    def __get_indices(self, tile_shape, origin):
        return [tuple(map(sum, zip(origin, x))) for x in tile_shape]

    def __update_listeners(self):
        for listener in self.listeners:
            listener.update(self.board)

    def __is_valid_move(self, new_tile, new_origin, new_orientation):
        new_idxs = self.__get_indices(new_tile.get_coords(new_orientation), new_origin)
        row_vals = [x[0] for x in new_idxs]
        col_vals = [x[1] for x in new_idxs]
        if (max(col_vals)) >= self.ncols:
            return False
        elif (min(col_vals)) < 0:
            return False
        elif (max(row_vals)) >= self.nrows:
            return False
        for cell in new_idxs:
            if self.board[cell[0], cell[1]] == 1:
                return False
        return True

    def __update_score(self):
        # figure out complete rows
        full_rows = np.where(np.all(a=self.board, axis=1))[0]
        BoardStats.tick_tile_count(self, full_rows)
        self.board[full_rows, :] = 0
        # shift rows down after removing complete lines
        full_rows = np.append(full_rows, -1)
        full_rows.sort()
        shift_block_idxs = list(map(lambda idx: range(full_rows[idx] + 1,
                                                      full_rows[idx + 1]), range(len(full_rows) - 1)))
        shift_blocks = list(map(lambda idx: np.copy(self.board[idx, :]), shift_block_idxs))
        for idx in shift_block_idxs: self.board[idx, :] = 0
        offset = len(full_rows)-1
        for i in range(len(full_rows)-1):
            self.board[full_rows[i]+offset+1:full_rows[i+1]+offset, :] = shift_blocks[i]
            offset -= 1
        logger.log("Score..." + str(BoardStats.get_line_count(self)), logger.Level.INFO, self.log)





