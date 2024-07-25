import argparse
import os
import pickle


def add_batch(observed_dp, file_ptr, batch_fn):
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

            dp_hash = hash(dp)

            if dp_hash in observed_dp:
                duplicates += 1
                continue

            observed_dp.add(dp_hash)
            pickle.dump(dp, file_ptr)


    return duplicates


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
    file_ptr = open(output_fn, "wb")

    duplicates = 0
    for i, fn in enumerate(batch_fns):
        print(f"Appending batch {i + 1}/{len(batch_fns)}: {fn}", end="\r", flush=True)
        duplicates = add_batch(observed_fens, file_ptr, fn)
    print(f"Appending batch {len(batch_fns)}/{len(batch_fns)}: Done!     ")
    print(f"Found {duplicates:,} duplicate FENs")

    file_ptr.close()
