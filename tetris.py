import numpy as np
import logger
import threading
from input_reader import KeyInputHandler
from tile import TileFactory


class Game(threading.Thread):

    def __init__(self, nrows, ncols, log, input, prob_override=None, rng_seed=None):
        super(Game, self).__init__()
        self.board = np.zeros((nrows, ncols))
        self.nrows = nrows
        self.ncols = ncols
        self.tile_count = 0
        self.line_count = 0
        self.log = log
        self.input = input
        self.alive = True
        self.listeners = []

        self.game_state = 0  # 0=waiting for game, 1=waiting for input
        self.tile = None
        self.tile_origin = None
        self.tile_orientation = None

        if prob_override is not None:
            self.tile_generator = TileFactory(probs=prob_override, seed=rng_seed)
        else:
            self.tile_generator = TileFactory(log)
        logger.log("Tetris Board Created...", logger.Level.INFO, self.log)
        # start thread in ctor
        self.start()

    def tick(self):
        # TODO: Use piece height and width info here and in ____is_valid_move
        if self.game_state == 0:
            new_tile = self.tile_generator.get_next_tile()
            new_tile_origin = (0, (self.ncols // 2) - 1)
            new_tile_orientation = 0
            self.__update_board(new_tile, new_tile_origin, new_tile_orientation)
            self.tile = new_tile
            self.tile_origin = new_tile_origin
            self.tile_orientation = new_tile_orientation
            self.game_state = 1
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
            elif next_key is KeyInputHandler.keys.DOWN:
                current_col = self.tile_origin[0]
                max_empty_row = 0
                for i in range(0, self.board.shape[0]):
                    if not self.board[i, current_col] == 0:
                        break
                    else:
                        max_empty_row = i
                new_tile_origin = (min([max_empty_row, self.nrows - 1]), self.tile_origin[1])
            else:
                pass
            if self.__is_valid_move(new_tile, new_tile_origin, new_tile_orientation):
                self.__update_board(new_tile, new_tile_origin, new_tile_orientation)
                self.tile = new_tile
                self.tile_origin = new_tile_origin
                self.tile_orientation = new_tile_orientation
                self.game_state == 1
            else:
                self.game_state == 0

    def __update_board(self, tile, origin, orientation):
        if self.tile is not None:
            previous_cells = self.__get_indices(self.tile.get_coords(self.tile_orientation),
                                           self.tile_origin)
            for cell in previous_cells:
                self.board[cell[0], cell[1]] = 0
        new_cells = self.__get_indices(tile.get_coords(orientation), origin)
        for cell in new_cells:
            self.board[cell[0], cell[1]] = 1

    def __get_indices(self, tile_shape, origin):
        return [tuple(map(sum, zip(origin, x))) for x in tile_shape]

    def __is_valid_move(self, new_tile, new_origin, new_orientation):
        new_idxs = self.__get_indices(new_tile.get_coords(new_orientation), new_origin)
        for cell in new_idxs:
            if self.board[cell[0], cell[1]] == 1:
                return False
        return True

    def run(self):
        while self.alive:
            self.tick()
            self.__update_listeners()

    def kill(self):
        self.alive = False

    def register_listener(self, listener):
        self.listeners.append(listener)
        listener.update(self.board)

    def __update_listeners(self):
        for listener in self.listeners:
            listener.update(self.board)


