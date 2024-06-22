import argparse
import os

from anz.inference import run_inference


DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-m", 
        "-model", 
        type=str, 
        help="The path to the model parameters to load", 
        required=True
    )
    arg_parser.add_argument(
        "-mt",
        "-model-type",
        type=str,
        default="transformer",
        choices=["transformer", "resnet"],
        help="Model type: 'transformer' or 'resnet'",
        required=True
    )
    arg_parser.add_argument(
        "-f", 
        "-fen", 
        type=str, 
        default=DEFAULT_FEN,
        required=False
    )
    arg_parser.add_argument(
        "-mcts", 
        type=int, 
        default=None,
        help="Number of MCTS simulations to run. MCTS will not be used if this argument is not provided",
        required=False
    )
    arg_parser.add_argument(
        "--v",
        "--value",
        action="store_true",
        help="Use only the value head of the model to generate a direct policy",
        required=False
    )
    arg_parser.add_argument(
        "--pi",
        "--policy",
        action="store_true",
        help="Use only the policy head of the model to generate a direct policy",
        required=False
    )
    args = arg_parser.parse_args()

    model_path = args.m
    model_type = args.mt
    fen = args.f
    mcts = args.mcts
    value_only = args.v
    policy_only = args.pi

    assert os.path.exists(model_path), f"Model file '{model_path}' does not exist"
    assert model_type in ["transformer", "resnet"], f"Invalid model type: {model_type}"
    assert mcts is None or mcts >= 0, f"Invalid value for mcts: {mcts}"
    assert not (value_only and policy_only), "Cannot use both --v and --pi flags at the same time"
    assert not ((value_only or policy_only) and (mcts is not None)), "Cannot use --v or --pi flags with MCTS"

    inference_result = run_inference(model_path, model_type, fen, mcts, value_only, policy_only)
    print(inference_result)
