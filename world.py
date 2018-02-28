import tkinter as tk
import time
import threading

static_specials = [(7, 3, "red", -1), (9, 1, "green", 1)]
static_x, static_y = (10, 6)
static_walls = [(1, 1), (1, 2), (2, 1), (2, 2), (3,4), (5,3), (5,4), (5,5), (5,0)]

#static_specials = [(4, 1, "red", -1), (4, 0, "green", 1)]
#static_x, static_y = (5, 5)
#static_walls = [(1, 1), (1, 2), (2, 1), (2, 2)]


class World(object):

    def __init__(self, do_render=True, init_x=None, init_y=None):
        self.do_render = do_render
        if self.do_render: self.master = tk.Tk()

        self.triangle_size = 0.2
        self.cell_score_min = -0.2
        self.cell_score_max = 0.2
        self.Width = 50
        self.x, self.y = static_x, static_y
        self.actions = ["up", "down", "left", "right"]

        if self.do_render: self.board = tk.Canvas(self.master, width=self.x*self.Width,
                               height=self.y*self.Width)
        self.score = 1
        self.restart = False
        self.walk_reward = -0.1

        self.walls = static_walls
        self.specials = static_specials
        self.cell_scores = {}

        if do_render: self.render_grid()
        if self.do_render:
            self.master.bind("<Up>", self.call_up)
            self.master.bind("<Down>", self.call_down)
            self.master.bind("<Right>", self.call_right)
            self.master.bind("<Left>", self.call_left)

        if not all(map(lambda x: isinstance(x, int), [init_x, init_y])):
            self.player = (0, self.y - 1)
            self.origin = (0, self.y - 1)
        else:
            self.origin = (init_x, init_y)
            self.player = self.origin

        if self.do_render:
            self.board.grid(row=0, column=0)
            self.me = self.board.create_rectangle(
                self.player[0] * self.Width + self.Width * 2 / 10,
                self.player[1] * self.Width + self.Width * 2 / 10,
                self.player[0] * self.Width + self.Width * 8 / 10,
                self.player[1] * self.Width + self.Width * 8 / 10, fill="orange", width=1,
                tag="me")

        if do_render: self.master.mainloop()


    def create_triangle(self, i, j, action):
        if action == self.actions[0]:
            return self.board.create_polygon((i+0.5-self.triangle_size)*self.Width, (j+self.triangle_size)*self.Width,
                                        (i+0.5+self.triangle_size)*self.Width, (j+self.triangle_size)*self.Width,
                                        (i+0.5)*self.Width, j*self.Width,
                                        fill="white", width=1)
        elif action == self.actions[1]:
            return self.board.create_polygon((i+0.5-self.triangle_size)*self.Width, (j+1-self.triangle_size)*self.Width,
                                        (i+0.5+self.triangle_size)*self.Width, (j+1-self.triangle_size)*self.Width,
                                        (i+0.5)*self.Width, (j+1)*self.Width,
                                        fill="white", width=1)
        elif action == self.actions[2]:
            return self.board.create_polygon((i+self.triangle_size)*self.Width, (j+0.5-self.triangle_size)*self.Width,
                                        (i+self.triangle_size)*self.Width, (j+0.5+self.triangle_size)*self.Width,
                                        i*self.Width, (j+0.5)*self.Width,
                                        fill="white", width=1)
        elif action == self.actions[3]:
            return self.board.create_polygon((i+1-self.triangle_size)*self.Width, (j+0.5-self.triangle_size)*self.Width,
                                        (i+1-self.triangle_size)*self.Width, (j+0.5+self.triangle_size)*self.Width,
                                        (i+1)*self.Width, (j+0.5)*self.Width,
                                        fill="white", width=1)

    def render_grid(self):
        for i in range(self.x):
            for j in range(self.y):
                self.board.create_rectangle(i*self.Width, j*self.Width,
                                            (i+1)*self.Width, (j+1)*self.Width, fill="white", width=1)
                temp = {}
                for action in self.actions:
                    temp[action] = self.create_triangle(i, j, action)
                    self.cell_scores[(i,j)] = temp
        for (i, j, c, w) in self.specials:
            self.board.create_rectangle(i*self.Width, j*self.Width,
                                        (i+1)*self.Width, (j+1)*self.Width, fill=c, width=1)
        for (i, j) in self.walls:
            self.board.create_rectangle(i*self.Width, j*self.Width,
                                        (i+1)*self.Width, (j+1)*self.Width, fill="black", width=1)

    def set_cell_score(self, state, action, val):
        triangle = self.cell_scores[state][action]
        green_dec = int(min(255, max(0, (val - self.cell_score_min) * 255.0 / (self.cell_score_max - self.cell_score_min))))
        green = hex(green_dec)[2:]
        red = hex(255-green_dec)[2:]
        if len(red) == 1:
            red += "0"
        if len(green) == 1:
            green += "0"
        color = "#" + red + green + "00"
        self.board.itemconfigure(triangle, fill=color)

    def try_move(self, dx, dy):
        #if self.restart:
        #    self.restart_game()

        # no movement out of terminal states
        for (i, j, c, w) in self.specials:
            if self.player[0] == i and self.player[1] == j:
                self.score += w
                return

        new_x = self.player[0] + dx
        new_y = self.player[1] + dy
        self.score += self.walk_reward
        # print(self.player, self.score, new_x, new_y)
        if (new_x >= 0) and (new_x < self.x) and (new_y >= 0) and (new_y < self.y) and not ((new_x, new_y) in self.walls):
            if self.do_render:
                self.board.coords(self.me, new_x*self.Width+self.Width*2/10, new_y*self.Width+self.Width*2/10,
                              new_x*self.Width+self.Width*8/10, new_y*self.Width+self.Width*8/10)
            self.player = (new_x, new_y)
        for (i, j, c, w) in self.specials:
            if new_x == i and new_y == j:
                self.score -= self.walk_reward
                self.score += w
                self.player = (new_x, new_y)
                # if self.score > 0:
                #     print("Success! score: ", self.score)
                # else:
                #     print("Fail! score: ", self.score)
                # self.restart = True
                return

    def call_up(self, event):
        self.try_move(0, -1)

    def call_down(self, event):
        self.try_move(0, 1)

    def call_left(self, event):
        self.try_move(-1, 0)

    def call_right(self, event):
        self.try_move(1, 0)

    def restart_game(self):
        self.player = self.origin
        self.score = 1
        self.restart = False
        self.board.coords(self.me, self.player[0]*self.Width+self.Width*2/10, self.player[1]*self.Width+self.Width*2/10,
                          self.player[0]*self.Width+self.Width*8/10, self.player[1]*self.Width+self.Width*8/10)

    def has_restarted(self):
        return self.restart

    def _close(self):
        self.quit()

    def quit(self):
        self.master.destroy()

