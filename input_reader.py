import time


TAIL_POLLING_SECS = 1.0


class FileTailer:

    def __init__(self, filepath):
        self.file = open(filepath, 'r')
        self.alive = True

    def tail(self):
        while self.alive:
            read_ptr = self.file.tell()
            line = self.file.readline()
            if not line:
                time.sleep(TAIL_POLLING_SECS)
                self.file.seek(read_ptr)
            else:
                yield line

