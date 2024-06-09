import os
import argparse
import math
import pickle
from pystockfish import Stockfish
from constants import THREADS, HASH, MULTI_PV, NODES_PER_ANNOTATION


def annotate(fen, engine, out_f) -> int:
    engine.set_position(fen)

    if engine.rule50_count() > 99:
        return 0

    engine.clear_evaluations()
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

        pickle.dump((fen, str(move), float(p_win)), out_f)

    return len(eval)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "-input-filename", type=str, help="File containing FEN strings", required=True)
    arg_parser.add_argument("-o", "-output-filename", type=str, help="Output pickle file", required=True)
    arg_parser.add_argument("-m", "-max-fens", type=int, default=None, help="Maximum number of FENs to annotate")
    arg_parser.add_argument("-s", "-skip", type=int, default=None, help="Number of FENs to skip before starting annotation")
    arg_parser.add_argument("-f", "-filter-duplicates", action="store_true", help="Filter duplicate FENs")
    args = arg_parser.parse_args()

    input_filename = args.i
    output_filename = args.o
    max_fens = args.m
    skip = args.s
    filter_duplicates = args.f

    assert os.path.exists(input_filename), f"File '{input_filename}' does not exist"
    assert not os.path.exists(output_filename), f"File '{output_filename}' already exists"
    assert max_fens is None or max_fens > 0, f"Invalid max_fens: '{max_fens}'"
    assert skip is None or skip >= 0, f"Invalid start: '{skip}'"

    engine = Stockfish()
    engine.set_option("Threads", THREADS)
    engine.set_option("Hash", HASH)
    engine.set_option("MultiPV", MULTI_PV)
    print(
        f"--- Stockfish settings ---\nThreads: {THREADS}\nHash: {HASH}\nMultiPV: {MULTI_PV}\nNodes per annotation: {NODES_PER_ANNOTATION}\n"
    )

    observed_fens = set()
    skipped_fens = 0
    with open(input_filename, "r") as in_f:
        if skip is not None:
            print(f"Skipping {skip} FENs")
            for _ in in_f:
                pass

        with open(output_filename, "wb") as out_f:
            dp = 0
            for i, fen in enumerate(in_f, start=1):
                if filter_duplicates:
                    if fen in observed_fens:
                        skipped_fens += 1
                        continue
                    else:
                        observed_fens.add(fen)

                print(f"Annotating FEN {i - skipped_fens}/{'-' if max_fens is None else max_fens} - Data points {dp} - skipped FENs {skipped_fens}", end="\r", flush=True)
                dp += annotate(fen, engine, out_f)
                if max_fens is not None and i - skipped_fens >= max_fens:
                    break
            print(f"Annotating FEN {i - skipped_fens}/{'-' if max_fens is None else max_fens} - Data points {dp} - skipped FENs {skipped_fens}")

