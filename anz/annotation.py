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

    if move not in eval:
        return None, None

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
        f"--- Stockfish settings ---\nThreads: {THREADS:,}\nHash: {HASH:,}\nNodes per annotation: {NODES_PER_ANNOTATION:,}\n"
    )

    observed_fens = set()
    data = deque()

    # Hacky solution for now
    if os.path.exists(output_fn):
        with open(output_fn, "rb") as f:
            while 1:
                try:
                    fen, move, value = pickle.load(f)
                    observed_fens.add(fen)
                    data.append((fen, move, value))
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error while reading already existing content in output file '{output_fn}': {e}")

    with open(input_fn, "r") as in_fp:
        with open(output_fn, "wb") as out_fp:

            if len(data) > 0:
                for dp in data:
                    pickle.dump(dp, out_fp)

            annotated = 0
            skipped = 0
            for fen in in_fp:
                if fen in observed_fens:
                    skipped += 1
                    continue

                observed_fens.add(fen)
                move, value = annotate(fen, engine)

                if move is None:
                    skipped += 1
                    continue

                annotated += 1
                pickle.dump((fen, move, value), out_fp)

                if max_fens is not None and annotated >= max_fens:
                    break

                print(f"Annotating FEN {annotated}/{'-' if max_fens is None else max_fens} - skipped {skipped} FENs", end="\r", flush=True)
            print(f"Annotating FEN {annotated}/{'-' if max_fens is None else max_fens} - skipped {skipped} FENs")

