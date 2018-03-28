import random
import string
import tkinter

HEAD_CHARACTER = 'ö'
FOOD_CHARACTERS = string.ascii_letters
FOOD_POSITIONS = [[150,150], [100, 100], [50, 50], [10, 50]]


class Application:
    TITLE = 'Snake'
    SIZE = 200, 200
    GAME_OVER_SCORE = -10
    WALK_SCORE = -0.5

    def __init__(self, master, init_x, init_y):
        self.master = master

        self.head = None
        self.head_position = [init_x, init_y]
        self.segments = []
        self.segment_positions = []
        self.food = None
        self.food_position = None
        self.direction = None
        self.moved = True
        self.food_positions = FOOD_POSITIONS.copy()

        self.running = False
        self.init()

    def init(self):
        if self.master:
            self.master.title(self.TITLE)

            self.canvas = tkinter.Canvas(self.master, width=Application.SIZE[0], height=Application.SIZE[1])
            self.canvas.grid(sticky=tkinter.NSEW)

            #self.start_button = tkinter.Button(self.master, text='Start', command=self.on_start)
            #self.start_button.grid(sticky=tkinter.EW)

            self.master.bind('w', self.on_up)
            self.master.bind('a', self.on_left)
            self.master.bind('s', self.on_down)
            self.master.bind('d', self.on_right)

            self.master.columnconfigure(0, weight=1)
            self.master.rowconfigure(0, weight=1)
            self.master.resizable(width=False, height=False)
        #self.master.geometry('%dx%d' % self.SIZE)

            self.reset()
        self.running = True
        self.start()

    def on_start(self):
        self.reset()
        if self.running:
            self.running = False
            self.start_button.configure(text='Start')
        else:
            self.running = True
            self.start_button.configure(text='Stop')
            self.start()

    def reset(self):
        self.segments.clear()
        self.segment_positions.clear()
        self.canvas.delete(tkinter.ALL)

    def start(self):
        #width = self.canvas.winfo_width()
        #height = self.canvas.winfo_height()

        width = Application.SIZE[0]
        height = Application.SIZE[1]
        if self.master:
            self.canvas.create_rectangle(5, 5, width-5, height-5)
            self.direction = random.choice('wasd')
            #head_position = [round(width // 2, -1), round(height // 2, -1)]
            self.head = self.canvas.create_text(tuple(self.head_position), text=HEAD_CHARACTER)
        #self.head_position = head_position
        self.spawn_food()
        self.tick()

    def spawn_food(self):
        #width = self.canvas.winfo_width()
        #height = self.canvas.winfo_height()
        width = Application.SIZE[0]
        height = Application.SIZE[1]

        positions = [tuple(self.head_position), self.food_position] + self.segment_positions

        #position = (round(random.randint(20, width-20), -1), round(random.randint(20, height-20), -1))
        position = self.food_positions.pop(0)
        while position in positions:
            position = self.food_positions.pop(0)

        character = random.choice(FOOD_CHARACTERS)
        if self.master:
            self.food = self.canvas.create_text(position, text=character)
        self.food_position = position
        self.food_character = character

    def tick(self):
        width = Application.SIZE[0]
        height = Application.SIZE[1]
        previous_head_position = tuple(self.head_position)

        if self.direction == 'w':
            self.head_position[1] -= 10
        elif self.direction == 'a':
            self.head_position[0] -= 10
        elif self.direction == 's':
            self.head_position[1] += 10
        elif self.direction == 'd':
            self.head_position[0] += 10

        head_position = tuple(self.head_position)
        if (self.head_position[0] < 10 or self.head_position[0] >= width-10 or
            self.head_position[1] < 10 or self.head_position[1] >= height-10 or
            any(segment_position == head_position for segment_position in self.segment_positions)):
            self.game_over()
            return

        if head_position == self.food_position:
            print("!!! SNAKE ATE A PIECE !!!")
            if self.master: self.canvas.coords(self.food, previous_head_position)
            self.segments.append(self.food)
            self.segment_positions.append(previous_head_position)
            self.spawn_food()

        if self.segments:
            previous_position = previous_head_position
            for index, (segment, position) in enumerate(zip(self.segments, self.segment_positions)):
                if self.master: self.canvas.coords(segment, previous_position)
                self.segment_positions[index] = previous_position
                previous_position = position
        if self.master:
            self.canvas.coords(self.head, head_position)
        self.moved = True

        #if self.running:
        #    self.canvas.after(50, self.tick)

    def game_over(self):
        width = Application.SIZE[0]
        height = Application.SIZE[1]

        self.running = False
        #self.start_button.configure(text='Start')
        score = len(self.segments) * 100
        if self.master:
            self.canvas.create_text((round(width // 2, -1), round(height // 2, -1)), text='Game Over! Your score was: %d' % score)

    def on_up(self, event):
        orig_length = len(self.segments)
        if self.moved and not self.direction == 's':
            self.direction = 'w'
            self.moved = False
            self.tick()
        if not self.running:
            return Application.GAME_OVER_SCORE, self.head_position
        else:
            return ((orig_length - len(self.segments))*100) + Application.WALK_SCORE, self.head_position

    def on_down(self, event):
        orig_length = len(self.segments)
        if self.moved and not self.direction == 'w':
            self.direction = 's'
            self.moved = False
            self.tick()
        if not self.running:
            return Application.GAME_OVER_SCORE, self.head_position
        else:
            return ((orig_length - len(self.segments))*100) + Application.WALK_SCORE, self.head_position

    def on_left(self, event):
        orig_length = len(self.segments)
        if self.moved and not self.direction == 'd':
            self.direction = 'a'
            self.moved = False
            self.tick()
        if not self.running:
            return Application.GAME_OVER_SCORE, self.head_position
        else:
            return ((orig_length - len(self.segments))*100) + Application.WALK_SCORE, self.head_position

    def on_right(self, event):
        orig_length = len(self.segments)
        if self.moved and not self.direction == 'a':
            self.direction = 'd'
            self.moved = False
            self.tick()
        if not self.running:
            return Application.GAME_OVER_SCORE, self.head_position
        else:
            return ((orig_length - len(self.segments))*100) + Application.WALK_SCORE, self.head_position


def main():
    root = tkinter.Tk()
    Application(root, 100, 100)
    root.mainloop()


if __name__ == '__main__':
    main()