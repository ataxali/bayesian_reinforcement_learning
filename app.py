import tkinter as tk
import threading
import time
import queue
import numpy as np


# tk needs to behave like a singleton
# we use global variables to manage this and other static properties
# tk should only be initialized once (but runs on its own thread)
# we could replace below with @staticmethod
global IS_TK_INIT
global root
global canvas_count

IS_TK_INIT = False
root = None
canvas_count = 0

DRAW_MULTIPLIER = 5
DRAW_X_PADDING = 0.2
DRAW_Y_PADDING = 0.2

class GameListener(object):
    """Abstract class for input handler, given to tetris.Game"""
    def update(self, board):
        raise NotImplementedError("Unimplemented method!")


class GameGraphics(threading.Thread):
    def __init__(self):
        # should only be initialized once
        global IS_TK_INIT
        if not IS_TK_INIT:
            IS_TK_INIT = True
            threading.Thread.__init__(self)
            self.canvas_count = 0
            self.start()

    def run(self):
        # all Tk references must be inside the function that calls mainloop
        global root
        root = tk.Tk()
        root.protocol("WM_DELETE_WINDOW", self.callback)
        root.mainloop()

    def callback(self):
        root.quit()

    def add_canvas(self, w=200, h=100):
        global canvas_count
        canvas = tk.Canvas(root, width=w, height=h)
        canvas.grid(row=0, column=canvas_count)
        canvas_count += 1
        return canvas


class App(GameGraphics, GameListener):
    FRAME_DELAY_SEC = 1

    def __init__(self, nrows, ncols):
        super(App, self).__init__()
        # Graphics run on a separate thread, so we wait for thread to run
        while root is None:
            time.sleep(1)
        self.x_offset = ncols * DRAW_X_PADDING
        self.y_offset = nrows * DRAW_Y_PADDING
        self.canvas = super(App, self).add_canvas(w=ncols*DRAW_MULTIPLIER + 2*self.x_offset,
                                                  h=nrows*DRAW_MULTIPLIER + 2*self.y_offset)
        self.nrows = nrows
        self.ncols = ncols
        self.drawn_board = np.zeros((nrows, ncols))
        self.frame_q = queue.Queue()
        self.__draw_board()

    def update(self, board):
        self.__push_frame(board)
        self.draw()
        #self.drawn_board = board

    def draw(self):
        next_board = self.frame_q.get(block=True)
        self.__draw_board(next_board)
        #
        time.sleep(self.FRAME_DELAY_SEC)

    def __draw_board(self, board=None):
        if board is None:
            self.__draw_default_board()
        else:
            diff = np.subtract(self.drawn_board, board)
            #print(self.drawn_board[0,:])
            #print(board[0, :])
            fill_rows, fill_cols = np.where(diff == -1)
            self.__fill_cells(fill_rows, fill_cols)

    def __draw_default_board(self):
        for i in range(0, self.ncols):
            for j in range(0, self.nrows):
                self.canvas.create_rectangle(i*DRAW_MULTIPLIER + self.x_offset,
                                             j*DRAW_MULTIPLIER + self.y_offset,
                                             (i+1)*DRAW_MULTIPLIER + self.x_offset,
                                             (j+1)*DRAW_MULTIPLIER + self.y_offset)

    def __fill_cells(self, rows, cols):
        #print(rows, cols)
        for i in range(len(rows)):
            self.canvas.create_rectangle(cols[i] * DRAW_MULTIPLIER + self.x_offset,
                                         rows[i] * DRAW_MULTIPLIER + self.y_offset,
                                         (cols[i] + 1) * DRAW_MULTIPLIER + self.x_offset,
                                         (rows[i] + 1) * DRAW_MULTIPLIER + self.y_offset,
                                         fill='black')

    def __push_frame(self, board):
        self.frame_q.put(board)

