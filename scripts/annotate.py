import os
import argparse

from anz.annotation import annotate_fen_file

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "-input-filename", type=str, help="File containing FEN strings", required=True)
    arg_parser.add_argument("-o", "-output-filename", type=str, help="Output pickle file", required=True)
    arg_parser.add_argument("-m", "-max-fens", type=int, default=None, help="Maximum number of FENs to annotate")
    args = arg_parser.parse_args()

    input_fn = args.i
    output_fn = args.o
    max_fens = args.m

    assert os.path.exists(input_fn), f"File '{input_fn}' does not exist"
    assert not os.path.exists(output_fn), f"File '{output_fn}' already exists"
    assert max_fens is None or max_fens > 0, f"Invalid max_fens: '{max_fens}'"

    annotate_fen_file(input_fn, output_fn, max_fens)
