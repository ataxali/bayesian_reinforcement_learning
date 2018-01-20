import numpy as np
import logger
import threading


class Game(threading.Thread):

    def __init__(self, nrows, ncols, log, input, listeners=[], prob_override=None, rng_seed=None):
        super(Game, self).__init__()

        self.board = np.zeros((nrows, ncols))
        self.tile_count = 0
        self.line_count = 0
        self.log = log
        self.input = input
        self.alive = True
        self.listeners = listeners

        if prob_override is not None:
            self.tile_generator = TileFactory(probs=prob_override, seed=rng_seed)
        else:
            self.tile_generator = TileFactory(log)
        logger.log("Tetris Board Created...", logger.Level.INFO, self.log)
        # start thread in ctor
        self.start()

    def tick(self):
        if self.tile_count > 0:
            self.board[0, self.tile_count] = 0
        self.board[0, self.tile_count+1] = 1
        self.tile_count += 1
        next_input = self.input.get_next_key()
        logger.log(next_input, logger.Level.DEBUG, self.log)

    def run(self):
        while self.alive:
            self.tick()
            self.__update_listeners()

    def kill(self):
        self.alive = False

    def __update_listeners(self):
        for listener in self.listeners:
            listener.update(self.board)


class Tile:

    tile_names = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']
    tile_shapes = [((0, 0), (1, 0), (2, 0), (3, 0)),  # I
                   ((0, 1), (1, 1), (2, 1), (2, 0)),  # J
                   ((0, 0), (1, 0), (2, 0), (2, 1)),  # L
                   ((0, 0), (1, 0), (0, 1), (1, 1)),  # O
                   ((1, 0), (0, 1), (1, 1), (0, 2)),  # S
                   ((0, 0), (0, 1), (0, 2), (1, 1)),  # T
                   ((0, 0), (0, 1), (1, 1), (1, 2))]  # Z

    class __metaclass__(type):
        def __getattr__(self, name):
            return self.tile_names.index(name)

    def __init__(self, name):
        if  name not in self.tile_names:
            raise Exception("Invalid Tile Name:", name)
        self.name = name
        self.shape = self.tile_shapes[self.tile_names.index(name)]


class TileFactory:

    def __init__(self, log, probs=[1/7]*7, seed=None):
        self.probs = np.cumsum(probs)
        self.rng = np.random.RandomState(seed)
        self.log = log
        logger.log("TileFactory Created With Probs:" + str(["%.2f"%p for p in probs])
                   , logger.Level.INFO, self.log)

    def __gen_idx(self):
        randn = self.rng.rand()
        for i in range(len(self.probs)):
            if randn <= self.probs[i]:
                return i
        return len(self.probs)-1

    def get_next_tile(self):
        next_tile_name = Tile.tile_names[self.__gen_idx()]
        logger.log("Generating tile: " + next_tile_name, logger.Level.DEBUG, self.log)
        return Tile(next_tile_name)

