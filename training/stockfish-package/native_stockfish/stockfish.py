import time

import native_stockfish_C

OPTIONS = {
    "Threads": int,
    "Hash": int,
    "MultiPV": int,
}

class Stockfish:
    def __init__(self):
        self.__engine = native_stockfish_C.Stockfish()

    def set_position(self, fen, move_list=None):
        if move_list is None:
            self.__engine.set_position(fen)
        else:
            assert isinstance(move_list, list)
            for move in move_list:
                assert isinstance(move, str), "move must be a string"
                assert len(move) == 4, "invalid move format: " + move
            self.__engine.set_position_ml(fen, move_list)

    def set_option(self, name, value):
        assert name in OPTIONS
        assert isinstance(value, OPTIONS[name])

        if name == "Threads":
            assert value >= 1 and value <= 1024, "Threads must be between 1 and 1024"
            self.__engine.set_num_threads(value)
        elif name == "Hash":
            assert value >= 1 and value <= 33554432, "Hash must be between 1 and 33554432"
            self.__engine.set_ht_size(value)
        elif name == "MultiPV":
            assert value >= 1 and value <= 256, "MultuPV must be between 1 and 256"
            self.__engine.set_multipv(value)
        else:
            assert False, "invalid option name '" + name + "'"

    def get_evaluations(self):
        return self.__engine.get_evaluations()

    def clear_evaluations(self):
        return self.__engine.clear_evaluations()

    def search(self, think_time):
        self.go()
        t0 = time.time()
        while time.time() - t0 < think_time:
            #remaining = think_time - (time.time() - t0)
            #print(f"searching ... {round(remaining, 1)}", end="\r", flush=True)
            time.sleep(0.1)
        #print(f"searching ... 0" + " "*10)
        self.stop()

    def go(self):
        self.__engine.go()

    def stop(self):
        self.__engine.stop()

    def visualize(self):
        return self.__engine.visualize()

    def __str__(self):
        return self.__engine.tostring()
