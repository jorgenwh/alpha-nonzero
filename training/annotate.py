import os
import sys
import math
from tqdm import tqdm
import numpy as np
from policy_index import policy_index
from pystockfish import Stockfish


THREADS = 5
HASH = 8192
MULTI_PV = 256
NODES_PER_ANNOTATION = 2000000


def move_to_policy_index(move: str) -> int:
    return policy_index.index(move)

def eval_to_annotation(eval: dict, annotations: np.ndarray, it: int):
    for move in eval:
        score_type, score = eval[move].split(" ")
        cp = None
        mate = None

        if score_type == "mate":
            mate = int(score)
            p_win = 1 if mate > 0 else 0
        elif score_type == "cp":
            cp = int(score)
            p_win = 0.5 * 2/(1 + math.exp(-0.00368208 * cp))
        else:
            raise ValueError(f"Invalid score type: '{score_type}'")

        idx = move_to_policy_index(move)
        #print(f"Move: {move}, CP: {cp}, Mate: {mate}, P_win: {p_win}, Index: {idx}")
        annotations[it, idx] = p_win

def annotate_position(fen: str, annotations: np.ndarray, it: int, engine: Stockfish):
    engine.clear_evaluations()
    engine.set_position(fen)
    engine.search(nodes=NODES_PER_ANNOTATION)
    eval = engine.get_evaluations()
    eval_to_annotation(eval, annotations, it)

def annotate_positions(fens, annotations, engine: Stockfish):
    bar = tqdm(fens)
    for i, fen in enumerate(bar):
        annotate_position(fen, annotations, i, engine)


if __name__ == "__main__":
    batch_number = sys.argv[1]
    batch_size = 1000

    training_data_dir = "training_data"
    input_batch_name = f"fen_batch_{batch_number}.fen"
    output_batch_name = f"annotation_batch_{batch_number}.anno"

    f = open(os.path.join(training_data_dir, input_batch_name), "r")
    fens = [fen for fen in f]
    f.close()

    annotations = np.zeros((batch_size, len(policy_index)), dtype=np.float32)

    engine = Stockfish()
    engine.set_option("Threads", THREADS)
    engine.set_option("Hash", HASH)
    engine.set_option("MultiPV", MULTI_PV)

    annotate_positions(fens, annotations, engine)

    np.save(os.path.join(training_data_dir, output_batch_name), annotations)
