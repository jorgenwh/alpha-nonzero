import argparse
import os
import pickle

def check_pickle_size(input_fn):
    print("Counting datapoints in pickle file...")
    num_dp = 0
    with open(input_fn, "rb") as f:
        while 1:
            try:
                _ = pickle.load(f)
            except EOFError:
                break
            except Exception as e:
                print(f"Unhandled error: {e}")
                exit()
            num_dp += 1
            if num_dp % 1000 == 0:
                print(f"Counting datapoints in pickle file... {num_dp}", end="\r", flush=True)
    print(f"Counting datapoints in pickle file... {num_dp}")
    print(f"Done")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "-input", type=str, help="Input pickle file to check the size of", required=True)
    args = arg_parser.parse_args()

    input_fn = args.i
    assert os.path.exists(input_fn), f"Input file {input_fn} does not exist"

    check_pickle_size(input_fn)
