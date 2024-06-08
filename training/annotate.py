import os
import sys
import math
from tqdm import tqdm
from utils import pickle_save
from pystockfish import Stockfish


THREADS = 5
HASH = 8192
MULTI_PV = 256
NODES_PER_ANNOTATION = 3000000


def annotate(fens: list, engine: Stockfish, data: list):
    bar = tqdm(fens, desc="annotating", bar_format="{l_bar}{bar}| update: {n_fmt}/{total_fmt} - data_points: {unit} - elapsed: {elapsed}")
    for fen in bar:
        engine.clear_evaluations()
        engine.set_position(fen)
        engine.search(nodes=NODES_PER_ANNOTATION)
        eval = engine.get_evaluations()

        for move in eval:
            score_type, score = eval[move].split(" ")
            cp = None
            mate = None

            if score_type == "mate":
                mate = int(score)
                p_win = 1 if mate > 0 else 0
            elif score_type == "cp":
                cp = int(score)
                p_win = 0.5 + 0.5 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)
            else:
                raise ValueError(f"Invalid score type: '{score_type}'")

            data.append((fen, str(move), p_win))
        bar.unit = str(len(data))


if __name__ == "__main__":
    batch_number = sys.argv[1]
    batch_size = 1000

    training_data_dir = "training_data"
    input_batch_name = f"fen_batch_{batch_number}.fen"

    f = open(os.path.join(training_data_dir, input_batch_name), "r")
    fens = [fen for fen in f]
    f.close()

    engine = Stockfish()
    engine.set_option("Threads", THREADS)
    engine.set_option("Hash", HASH)
    engine.set_option("MultiPV", MULTI_PV)

    data = []
    annotate(fens, engine, data)
    pickle_save(data, "training_data/training_batch_{batch_number}.pkl")
