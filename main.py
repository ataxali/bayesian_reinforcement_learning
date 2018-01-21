import app
import logger
import tetris
import input_reader


def main():
    # initialize loggers
    main_file_log = logger.FileLogger(filename=".log", name="main.file", level=logger.Level.DEBUG)
    main_console_log = logger.ConsoleLogger(name="console.main", level=logger.Level.DEBUG)
    main_data_log = logger.DataLogger(filename=".log", name="main.data")
    # initialize input handler
    keyinput = input_reader.KeyInputHandler(main_console_log)
    _ = input_reader.FileTailer("input.txt", keyinput, main_console_log)
    _ = input_reader.KeyListener(keyinput, main_console_log)
    # initialize graphics (needs to be optional)
    nrows = 24
    ncols = 10
    gui = app.App(nrows, ncols)
    # initialize basic game
    game = tetris.Game(nrows=nrows, ncols=ncols, log=[main_file_log, main_console_log], input=keyinput,
                       listeners=[gui])
    logger.log("Back to Main! :)", logger.Level.INFO, main_data_log)


if __name__ == "__main__":
    main()
