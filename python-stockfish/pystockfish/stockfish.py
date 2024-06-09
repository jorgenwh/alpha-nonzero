import time
import pystockfish_C as _C 

OPTIONS = {
    "Threads": int,
    "Hash": int,
    "MultiPV": int,
}

DEFAULT_SEARCH_TIME = 2

class Stockfish:
    def __init__(self):
        self.engine = _C.EngineWrapper()

    def set_position(self, fen, move_list=None):
        if move_list is None:
            self.engine.set_position(fen)
        else:
            assert isinstance(move_list, list)
            for move in move_list:
                assert isinstance(move, str), "move must be a string"
                assert len(move) == 4, "invalid move format: " + move
            self.engine.set_position_ml(fen, move_list)

    def set_option(self, name, value):
        assert name in OPTIONS
        assert isinstance(value, OPTIONS[name])

        if name == "Threads":
            assert value >= 1 and value <= 1024, "Threads must be between 1 and 1024"
            self.engine.set_num_threads(value)
        elif name == "Hash":
            assert value >= 1 and value <= 33554432, "Hash must be between 1 and 33554432"
            self.engine.set_ht_size(value)
        elif name == "MultiPV":
            assert value >= 1 and value <= 256, "MultuPV must be between 1 and 256"
            self.engine.set_multipv(value)
        else:
            assert False, "invalid option name '" + name + "'"

    def get_evaluations(self):
        return self.engine.get_evaluations()

    def clear_evaluations(self):
        return self.engine.clear_evaluations()

    def nodes(self):
        return self.engine.nodes()

    def search(self, think_time=None, nodes=None):
        if think_time is not None or (think_time is None and nodes is None):
            self.go()
            time.sleep(think_time if think_time is not None else DEFAULT_SEARCH_TIME)
            self.stop()
        elif nodes is not None:
            self.go_nodes_limit(nodes)

    def go(self):
        self.engine.go()

    def go_nodes_limit(self, nodes):
        self.engine.go_nodes_limit(nodes)

    def stop(self):
        self.engine.stop()

    def rule50_count(self):
        return self.engine.rule50_count()

    def visualize(self):
        return self.engine.visualize()

    def __str__(self):
        return self.engine.tostring()
