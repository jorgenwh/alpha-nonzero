import argparse
import os
import pickle
from collections import deque


def add_batch(observed_dp, data, batch_fn):
    duplicates = 0
    with open(batch_fn, "rb") as f:
        while 1:
            try:
                dp = pickle.load(f)
            except EOFError:
                break
            except Exception as e:
                print(f"Unhandled error: {e}")
                exit()

            if dp in observed_dp:
                duplicates += 1
                continue

            observed_dp.add(dp)
            data.append(dp)

    return duplicates

def dump_data(data, output_fn):
    with open(output_fn, "wb") as f:
        for dp in data:
            pickle.dump(dp, f)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "-input", type=str, help="Input directory containing the annotation batches to combine", required=True)
    arg_parser.add_argument("-o", "-output", type=str, help="Output filename", required=True)
    args = arg_parser.parse_args()

    input_dir = args.i
    output_fn = args.o

    assert os.path.isdir(input_dir), f"Directory '{input_dir}' does not exist"
    assert not os.path.exists(output_fn), f"File '{output_fn}' already exists"

    batch_fns = [os.path.join(input_dir, fn) for fn in os.listdir(input_dir)]
    assert len(batch_fns) > 0, f"No files found in '{input_dir}'"

    observed_fens = set()
    data = deque()

    duplicates = 0
    for fn in batch_fns:
        duplicates = add_batch(observed_fens, data, fn)
    print(f"Found {duplicates} duplicate FENs")

    dump_data(data, output_fn)
