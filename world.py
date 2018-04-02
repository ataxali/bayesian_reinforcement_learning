import tkinter as tk
import time
import threading
import numpy as np
from inputReader import KeyInputHandler

static_x_dim, static_y_dim = (10, 7)
static_time_between_moves = 0.1

# (x, y, type, reward, velocity)
# 4 cat case
static_specials = [(7, 3, "red", -10, "up"), (2, 4, "red", -10, "left"),
                   (8, 4, "red", -10, "left"), (3, 0, "red", -10, "down"),
                   (9, 6, "green", 10, "NA")]
static_walls = [(1, 1), (1, 2), (2, 2), (3, 4), (5, 2), (5, 3), (5, 6), (5, 0)]


# 2 cat case
#static_specials = [(7, 3, "red", -10, "up"), (8, 5, "red", -10, "left"), (9, 1, "green", 10, "NA")]
#static_walls = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 4), (5, 3), (5, 4), (5, 5), (5, 0)]

# deterministic case
#static_specials = [(4, 1, "red", -1), (4, 0, "green", 1)]
#static_x_dim, static_y_dim = (5, 5)
#static_walls = [(1, 1), (1, 2), (2, 1), (2, 2)]


class World(object):

    def __init__(self, do_render=True, init_x=None, init_y=None, move_pool=None,
                 input_reader=None, specials=static_specials, walls=static_walls,
                 do_restart=False, do_belief=False):
        self.do_render = do_render
        if self.do_render: self.master = tk.Tk()

        self.triangle_size = 0.2
        self.cell_score_min = -0.2
        self.cell_score_max = 0.2
        self.Width = 50
        self.x, self.y = static_x_dim, static_y_dim
        self.actions = ["up", "down", "left", "right"]
        self.do_restart = do_restart
        self.do_belief = do_belief

        if self.do_render:
            self.board = tk.Canvas(self.master, width=self.x*self.Width,
                               height=self.y*self.Width)
            self.board.pack(fill=tk.BOTH, expand=tk.YES)
        self.score = 0
        self.restart = False
        self.walk_reward = -0.1

        self.walls = walls
        self.belief_walls = []
        self.original_specials = specials.copy()
        self.specials = specials
        self.belief_states = list()
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

        if move_pool:
            t = threading.Thread(target=self.run_pooled_moves, args=(move_pool,))
            t.daemon = True
            t.start()

        if input_reader:
            t = threading.Thread(target=self.run_input_moves, args=(input_reader,))
            t.daemon = True
            t.start()

        if do_render: self.master.mainloop()

    def run_input_moves(self, input_reader):
        time.sleep(1)
        while True:
            next_key = input_reader.get_next_key()
            if next_key == KeyInputHandler.keys.UP:
                time.sleep(static_time_between_moves)
                self.call_up(None)
            elif next_key == KeyInputHandler.keys.DOWN:
                time.sleep(static_time_between_moves)
                self.call_down(None)
            elif next_key == KeyInputHandler.keys.LEFT:
                time.sleep(static_time_between_moves)
                self.call_left(None)
            elif next_key == KeyInputHandler.keys.RIGHT:
                time.sleep(static_time_between_moves)
                self.call_right(None)
            elif next_key == KeyInputHandler.keys.RESET:
                time.sleep(static_time_between_moves)
                self.restart_game()
            elif next_key[:4] == 'addr':
                self.add_belief_node(next_key[4:], "R")
            elif next_key[:4] == 'addc':
                self.add_belief_node(next_key[4:], "C")
            elif next_key[:4] == 'addw':
                self.add_belief_walls(next_key[4:])
            elif next_key[:3] == 'clrw':
                self.clear_belief_walls()
            elif next_key[:3] == 'clr':
                self.clear_belief_nodes()
            else:
                print("Unknown key input:", str(next_key))

    def clear_belief_walls(self):
        if not self.do_belief: return
        n = len(self.belief_walls)
        for i in range(n):
            self.board.delete(self.belief_walls[i])
        self.belief_walls = list()

    def add_belief_walls(self, coords):
        if not self.do_belief: return
        x_y_vals = list(map(lambda x: int(x), coords.split(',')))
        if not len(x_y_vals) == 2:
            raise Exception("Cannot add belief wall: " + str(coords))
        new_rect = self.board.create_rectangle(x_y_vals[0]*self.Width, x_y_vals[1]*self.Width,
                                               (x_y_vals[0]+1)*self.Width, (x_y_vals[1]+1)*self.Width,
                                               fill="black", width=1)
        self.board.tag_raise(self.me)
        self.walls.append((x_y_vals[0], x_y_vals[1]))
        self.belief_walls.append(new_rect)

    def clear_belief_nodes(self):
        if not self.do_belief: return
        n = len(self.belief_states)
        for i in range(n):
            self.board.delete(self.belief_states[i])
        self.belief_states = list()

    def add_belief_node(self, coords, type):
        if not self.do_belief: return
        col = None
        if type == "R":
            col = "red"
        else:
            col = "light salmon"
        x_y_vals = list(map(lambda x: int(x), coords.split(',')))
        if not len(x_y_vals) == 2:
            raise Exception("Cannot add belief coord: " + str(coords))
        for wall in self.walls:
            if x_y_vals[0] == wall[0] and x_y_vals[1] == wall[1]:
                return
        for special in self.specials:
            if x_y_vals[0] == special[0] and x_y_vals[1] == special[1]:
                return
        new_rect = self.board.create_rectangle(x_y_vals[0] * self.Width,
                                               x_y_vals[1] * self.Width,
                                               (x_y_vals[0] + 1) * self.Width,
                                               (x_y_vals[1] + 1) * self.Width, fill=col,
                                               width=1)
        self.board.tag_raise(self.me)
        self.belief_states.append(new_rect)

    def run_pooled_moves(self, move_pool):
        time.sleep(1)
        while len(move_pool) != 0:
            action = move_pool[0]
            if action == self.actions[0]:
                self.try_move(0, -1)
            elif action == self.actions[1]:
                self.try_move(0, 1)
            elif action == self.actions[2]:
                self.try_move(-1, 0)
            elif action == self.actions[3]:
                self.try_move(1, 0)
            else:
                print("Unknown move", action)
            move_pool.pop(0)
            time.sleep(static_time_between_moves)

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

    def render_reset_grid(self):
        for i in range(self.x):
            for j in range(self.y):
                self.board.create_rectangle(i*self.Width, j*self.Width,
                                            (i+1)*self.Width, (j+1)*self.Width, fill="black", width=1)

                randn = np.random.choice(range(len(self.actions)))
                for action in self.actions[0:randn]:
                    self.create_triangle(i, j, action)

    def render_grid(self):
        for i in range(self.x):
            for j in range(self.y):
                self.board.create_rectangle(i*self.Width, j*self.Width,
                                            (i+1)*self.Width, (j+1)*self.Width, fill="white", width=1)
                #temp = {}
                #for action in self.actions:
                #    temp[action] = self.create_triangle(i, j, action)
                #    self.cell_scores[(i,j)] = temp
        for (i, j, c, w, v) in self.specials:
            self.board.create_rectangle(i*self.Width, j*self.Width,
                                        (i+1)*self.Width, (j+1)*self.Width, fill=c, width=1)
        for (i, j) in self.walls:
            wall_shape = self.board.create_rectangle(i*self.Width, j*self.Width,
                                        (i+1)*self.Width, (j+1)*self.Width, fill="black", width=1)
            self.belief_walls.append(wall_shape)

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

    def update_specials(self):
        # constant specials
        # return self.specials
        red_specials = []
        green_specials = []
        updated_red_specials = []
        for special in self.specials:
            if special[2] == "red":
                red_specials.append(special)
            if special[2] == "green":
                green_specials.append(special)
        for (i, j, c, w, v) in red_specials:
            if v == "up":
                j -= 1
                if (j >= 0) and (j < self.y) and not ((i, j) in self.walls):
                    pass # pass, all good
                else:
                    v = "down"
                    j += 2
            elif v == "down":
                j += 1
                if (j >= 0) and (j < self.y) and not ((i, j) in self.walls):
                    pass  # pass, all good
                else:
                    v = "up"
                    j -= 2
            elif v == "left":
                i -= 1
                if (i >= 0) and (i < self.x) and not ((i, j) in self.walls):
                    pass  # pass, all good
                else:
                    v = "right"
                    i += 2
            elif v == "right":
                i += 1
                if (i >= 0) and (i < self.x) and not ((i, j) in self.walls):
                    pass  # pass, all good
                else:
                    v = "left"
                    i -= 2
            updated_red_specials.append((i, j, c, w, v))
        return updated_red_specials + green_specials

    def try_move_idx(self, move_idx):
        if move_idx == 0:
            self.try_move(0, -1)
        elif move_idx == 1:
            self.try_move(0, 1)
        elif move_idx == 2:
            self.try_move(-1, 0)
        elif move_idx == 3:
            self.try_move(1, 0)
        else:
            print("Unknown move index", move_idx)

    def try_move(self, dx, dy):
        # no movement out of terminal states
        for (i, j, c, w, v) in self.specials:
            if self.player[0] == i and self.player[1] == j:
                if self.do_restart:
                    print("Restarting game...")
                    self.restart_game()
                    print("Game restarted...")
                    return
                else:
                    self.score += w
                    return

        old_specials = self.specials.copy()
        self.specials = self.update_specials()
        new_x = self.player[0] + dx
        new_y = self.player[1] + dy
        self.score += self.walk_reward
        if (new_x >= 0) and (new_x < self.x) and (new_y >= 0) and (new_y < self.y) and not ((new_x, new_y) in self.walls):
            # if self.do_render:
            #     for (i, j, c, w, v) in old_specials:
            #         if c == "red":
            #             self.board.create_rectangle(i * self.Width, j * self.Width,
            #                                         (i + 1) * self.Width,
            #                                         (j + 1) * self.Width, fill='white',
            #                                         width=1)
            #     self.board.coords(self.me, new_x*self.Width+self.Width*2/10, new_y*self.Width+self.Width*2/10,
            #                   new_x*self.Width+self.Width*8/10, new_y*self.Width+self.Width*8/10)
            #     self.board.tag_raise(self.me)
            #     for (i, j, c, w, v) in self.specials:
            #         self.board.create_rectangle(i * self.Width, j * self.Width,
            #                                     (i + 1) * self.Width,
            #                                     (j + 1) * self.Width, fill=c,
            #                                     width=1)
            self.player = (new_x, new_y)

        if self.do_render:
            self.board.tag_raise(self.me)
            for (i, j, c, w, v) in old_specials:
                if c == "red":
                    self.board.create_rectangle(i * self.Width,
                                                j * self.Width,
                                                (i + 1) * self.Width,
                                                (j + 1) * self.Width,
                                                fill='white',
                                                width=1)
            self.board.coords(self.me,
                              self.player[
                                  0] * self.Width + self.Width * 2 / 10,
                              self.player[
                                  1] * self.Width + self.Width * 2 / 10,
                              self.player[
                                  0] * self.Width + self.Width * 8 / 10,
                              self.player[
                                  1] * self.Width + self.Width * 8 / 10)
            self.board.tag_raise(self.me)
            for (i, j, c, w, v) in self.specials:
                self.board.create_rectangle(i * self.Width,
                                            j * self.Width,
                                            (i + 1) * self.Width,
                                            (j + 1) * self.Width,
                                            fill=c,
                                            width=1)
            self.board.tag_raise(self.me)

        for (i, j, c, w, v) in self.specials:
            if self.player[0] == i and self.player[1] == j:
                self.score -= self.walk_reward
                self.score += w

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
        self.specials = self.original_specials.copy()
        self.restart = False
        if self.do_render:
            self.render_reset_grid()
            time.sleep(static_time_between_moves)
            self.board.delete('all')
            self.render_grid()
            self.me = self.board.create_rectangle(
                self.player[0] * self.Width + self.Width * 2 / 10,
                self.player[1] * self.Width + self.Width * 2 / 10,
                self.player[0] * self.Width + self.Width * 8 / 10,
                self.player[1] * self.Width + self.Width * 8 / 10, fill="orange", width=1,
                tag="me")
            self.board.tag_raise(self.me)
            time.sleep(static_time_between_moves)

    def has_restarted(self):
        return self.restart

    def _close(self):
        self.quit()

    def quit(self):
        self.master.destroy()

