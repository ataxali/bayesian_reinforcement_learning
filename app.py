import tkinter as tk
import threading


class App(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        # call thread's start, which invokes run()
        self.start()

    def callback(self):
        self.root.quit()

    def run(self):
        # all Tk references must be inside the function that calls mainloop
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        label = tk.Label(self.root, text="Hello World")
        label.pack()

        self.root.mainloop()
