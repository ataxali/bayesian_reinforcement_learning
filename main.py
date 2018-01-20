import sys
import app
import logger
import tetris

def main():
    # initialize loggers
    main_file_log = logger.FileLogger(name="file-main", level=logger.Level.DEBUG)
    main_console_log = logger.ConsoleLogger(name="console-main", level=logger.Level.DEBUG)
    # initialize basic game
    tetris.Game(nrows=100, ncols=30, log=[main_file_log, main_console_log])
    # initialize graphics (needs to be optional)
    app.App()


if __name__ == "__main__":
    main()
