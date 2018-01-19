import sys
import app
import logger
import tetris

def main():
    # print command line arguments
    for arg in sys.argv[0:]:
        print(arg)

    # initialize loggers
    logger.FileLogger(name="file-main", level=logger.Level.DEBUG)
    logger.ConsoleLogger(name="console-main", level=logger.Level.DEBUG)

    tetris.Game(nrows=100, ncols=30)

    app.App()


if __name__ == "__main__":
    main()
