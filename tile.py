import numpy as np
import logger


class Tile:
    #TODO: Add piece height and width info here
    tile_names = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']
    tile_shapes_o_0 = [((0, 0), (1, 0), (2, 0), (3, 0)),  # I
                       ((0, 1), (1, 1), (2, 1), (2, 0)),  # J
                       ((0, 0), (1, 0), (2, 0), (2, 1)),  # L
                       ((0, 0), (1, 0), (0, 1), (1, 1)),  # O
                       ((1, 0), (0, 1), (1, 1), (0, 2)),  # S
                       ((0, 0), (0, 1), (0, 2), (1, 1)),  # T
                       ((0, 0), (0, 1), (1, 1), (1, 2))]  # Z

    tile_shapes_o_1 = [((0, 0), (0, 1), (0, 2), (0, 3)),  # I
                       ((0, 0), (1, 0), (1, 1), (1, 2)),  # J
                       ((0, 0), (0, 1), (0, 2), (1, 0)),  # L
                       ((0, 0), (1, 0), (0, 1), (1, 1)),  # O
                       ((0, 0), (1, 0), (1, 1), (2, 1)),  # S
                       ((1, 0), (0, 1), (1, 1), (2, 1)),  # T
                       ((0, 1), (1, 0), (1, 1), (2, 0))]  # Z

    tile_shapes_o_2 = [((0, 0), (1, 0), (2, 0), (3, 0)),  # I
                       ((0, 0), (0, 1), (1, 0), (2, 0)),  # J
                       ((0, 0), (0, 1), (1, 1), (2, 1)),  # L
                       ((0, 0), (1, 0), (0, 1), (1, 1)),  # O
                       ((1, 0), (0, 1), (1, 1), (0, 2)),  # S
                       ((1, 0), (1, 1), (1, 2), (0, 1)),  # T
                       ((0, 0), (0, 1), (1, 1), (1, 2))]  # Z

    tile_shapes_o_3 = [((0, 0), (0, 1), (0, 2), (0, 3)),  # I
                       ((0, 0), (0, 1), (0, 2), (1, 2)),  # J
                       ((0, 2), (1, 0), (1, 1), (1, 2)),  # L
                       ((0, 0), (1, 0), (0, 1), (1, 1)),  # O
                       ((0, 0), (1, 0), (1, 1), (2, 1)),  # S
                       ((0, 0), (1, 0), (2, 0), (1, 1)),  # T
                       ((0, 1), (1, 0), (1, 1), (2, 0))]  # Z

    class __metaclass__(type):
        def __getattr__(self, name):
            return self.tile_names.index(name)

    def __init__(self, name):
        if  name not in self.tile_names:
            raise Exception("Invalid Tile Name:", name)
        self.name = name
        self.shape = self.tile_shapes_o_0[self.tile_names.index(name)]

    def get_coords(self, orientation):
        if orientation == 0:
            return self.tile_shapes_o_0[self.tile_names.index(self.name)]
        elif orientation == 1:
            return self.tile_shapes_o_1[self.tile_names.index(self.name)]
        elif orientation == 2:
            return self.tile_shapes_o_2[self.tile_names.index(self.name)]
        elif orientation == 3:
            return self.tile_shapes_o_3[self.tile_names.index(self.name)]
        else:
            raise Exception("Invalid orientation!")


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

