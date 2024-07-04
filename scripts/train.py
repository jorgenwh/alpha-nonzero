import argparse
import os
from anz.training import train

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-d",
        "-data",
        type=str,
        help="File path containing pickled training data",
        required=True
    )
    arg_parser.add_argument(
        "-mt",
        "-model-type",
        type=str,
        choices=["transformer", "resnet"],
        help="Model type: 'transformer' or 'resnet'",
        required=True
    )
    arg_parser.add_argument(
        "-o",
        "-outdir",
        type=str,
        help="Path to the output directory where model checkpoints will be saved",
        required=True
    )
    arg_parser.add_argument(
        "-md",
        "-max-datapoints",
        type=int,
        default=None,
        help="Limit on the number of datapoints loaded from the training data file",
    )
    arg_parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use half precision",
        required=False
    )
    args = arg_parser.parse_args()

    data_fn = args.d
    model_type = args.mt
    output_dir = args.o
    max_datapoints = args.md
    use_fp16 = args.fp16

    assert os.path.isfile(data_fn), f"File not found: {data_fn}"
    assert max_datapoints is None or max_datapoints > 0, f"Invalid value for max_datapoints: {max_datapoints}"

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    train(data_fn, model_type, output_dir, max_datapoints, use_fp16)
