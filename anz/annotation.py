import os
import math
import pickle
from collections import deque
from pystockfish import Stockfish
from .constants import THREADS, HASH, NODES_PER_ANNOTATION


def annotate(fen, engine):
    engine.set_position(fen)

    if engine.rule50_count() > 99:
        return None, None

    engine.clear_evaluations()
    engine.search(nodes=NODES_PER_ANNOTATION)
    eval = engine.get_evaluations()
    move = engine.get_best_move()
    assert move in eval, f"Move '{move}' not in eval: {eval}"
    value_type, value = eval[move].split(" ")
    value = int(value)

    if value_type == "mate":
        value = 1 if value > 0 else -1
    elif value_type == "cp":
        value = (2 / (1 + math.exp(-0.00368208 * value)) - 1)
    else:
        raise ValueError(f"Invalid value type: '{value_type}'")

    return move, value

def annotate_fen_file(input_fn, output_fn, max_fens):
    engine = Stockfish()
    engine.set_option("Threads", THREADS)
    engine.set_option("Hash", HASH)
    engine.set_option("MultiPV", 1)

    print(
        f"--- Stockfish settings ---\nThreads: {THREADS}\nHash: {HASH}\nNodes per annotation: {NODES_PER_ANNOTATION}\n"
    )

    # Hacky solution for now
    data = deque()
    if os.path.exists(output_fn):
        with open(output_fn, "rb") as f:
            while 1:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error while reading already existing content in output file '{output_fn}': {e}")

    in_fp = open(input_fn, "r")
    out_fp = open(output_fn, "wb")

    if len(data) > 0:
        for dp in data:
            pickle.dump(dp, out_fp)

    for i, fen in enumerate(in_fp, start=1):
        print(fen)
        move, value = annotate(fen, engine)
        if move is not None:
            pickle.dump((fen, move, value), out_fp)
        if max_fens is not None and i >= max_fens:
            break
        #print(f"Annotating FEN {i}/{'-' if max_fens is None else max_fens}", end="\r", flush=True)
    #print(f"Annotating FEN {i}/{'-' if max_fens is None else max_fens}")

    in_fp.close()
    out_fp.close()

