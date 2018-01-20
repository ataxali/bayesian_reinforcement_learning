import time
import logger
import threading
import enum
import queue


class InputHandler(object):
    """Abstract class for input handler, given to tetris.Game()"""
    def get_next_key(self):
        raise NotImplementedError("Unimplemented method!")


class KeyInputHandler(InputHandler):
    keys = enum.Enum('keys', 'UP DOWN LEFT RIGHT ROTATE_L ROTATE_R PASS')

    def __init__(self, log):
        self.log = log
        self.key_q = queue.Queue()
        logger.log("Key stroke handler created...", logger.Level.INFO, log)

    # handle needs to be thread safe (a handler can be shared between threads)!
    def handle(self, line):
        # clean input
        inputs = line.lower().split()
        # put into queue, with blocking enabled
        for input in inputs:
            next_key = None
            if input == 'up':
                next_key = self.keys.UP
            elif input == 'down':
                next_key = self.keys.DOWN
            elif input == 'left':
                next_key = self.keys.LEFT
            elif input == 'right':
                next_key = self.keys.RIGHT
            elif input == 'rr':
                next_key = self.keys.ROTATE_R
            elif input == 'rl':
                next_key = self.keys.ROTATE_L
            elif input == 'pass':
                next_key = self.keys.PASS
            else:
                logger.log("Unrecognized input:" + line.strip(), logger.Level.ERROR, self.log)
            if next_key is not None:
                self.key_q.put(next_key, block=True)
                logger.log("Pushing:" + str(next_key), logger.Level.DEBUG, self.log)

    def get_next_key(self):
        # this is a blocking call, will wait until an item is on queue
        return self.key_q.get(block=True)


class FileTailer(threading.Thread):
    def __init__(self, filepath, handler, log, tail_polling_secs = 1.0):
        super(FileTailer, self).__init__()
        self.file = open(filepath, 'r')
        self.log = log
        self.alive = True
        self.handler = handler
        self.tail_polling_secs = tail_polling_secs
        logger.log("File tailer created for:" + filepath + "...",
                   logger.Level.INFO, log)
        # start the thread in ctor
        self.start()

    def tail(self):
        while self.alive:
            read_ptr = self.file.tell()
            line = self.file.readline()
            if not line:
                time.sleep(self.tail_polling_secs)
                self.file.seek(read_ptr)
            else:
                yield line

    # start() invokes run
    def run(self):
        logger.log("File tailer for " + str(self.file) + " started...",
                   logger.Level.INFO, self.log)
        for line in self.tail():
            self.handler.handle(line)
