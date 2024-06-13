import os
import argparse
from anz.pgn import parse_pgn


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "-input-filename", type=str, help="PGN file", required=True)
    arg_parser.add_argument("-o", "-output-filename", type=str, help="Output fen file", required=True)
    args = arg_parser.parse_args()

    input_fn = args.i
    output_fn = args.o

    assert os.path.exists(input_fn), f"File '{input_fn}' does not exist"
    assert not os.path.exists(output_fn), f"File '{output_fn}' already exists"

    parse_pgn(input_fn, output_fn)
