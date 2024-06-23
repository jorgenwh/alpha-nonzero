import argparse
import os
import matplotlib.pyplot as plt

from anz.puzzle import puzzle_evaluate_training_trajectory

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-m", 
        "-models_dir", 
        type=str, 
        help="The path to the directory containing the training checkpoint parameters", 
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
        "-p", 
        "-puzzles", 
        type=str, 
        help="The path to the file containing puzzles", 
        required=True
    )
    arg_parser.add_argument(
        "-o", 
        "-out-file", 
        type=str, 
        help="Name of the output plot file", 
        required=True
    )
    arg_parser.add_argument(
        "-np", 
        "-num-puzzles", 
        type=int, 
        default=None,
        help="Upper limit on how many puzzles to evaluate each checkpoint on", 
        required=False
    )
    arg_parser.add_argument(
        "-mcts-rollouts", 
        type=int, 
        default=None,
        help="Number of MCTS simulations to run per move. MCTS will not be used if this argument is not provided",
        required=False
    )
    arg_parser.add_argument(
        "--v",
        "--value",
        action="store_true",
        help="Use only the value head of the models to generate a direct policy",
        required=False
    )
    arg_parser.add_argument(
        "--pi",
        "--policy",
        action="store_true",
        help="Use only the policy head of the models to generate a direct policy",
        required=False
    )
    args = arg_parser.parse_args()

    models_dir = args.m
    model_type = args.mt
    puzzles_path = args.p
    out_file = args.o
    num_puzzles = args.np
    mcts_rollouts = args.mcts_rollouts
    value_only = args.v
    policy_only = args.pi

    assert os.path.isdir(models_dir), f"Model checkpoint directory '{models_dir}' does not exist"
    assert model_type in ["transformer", "resnet"], f"Invalid model type: {model_type}"
    assert os.path.exists(puzzles_path), f"Puzzles file '{puzzles_path}' does not exist"
    assert not os.path.exists(out_file), f"Output file '{out_file}' already exists"
    assert num_puzzles is None or num_puzzles > 0, f"Invalid value for num_puzzles: {num_puzzles}"
    assert mcts_rollouts is None or mcts_rollouts >= 0, f"Invalid value for mcts-rollouts: {mcts_rollouts}"
    assert not (value_only and policy_only), "Cannot use both --v and --pi flags at the same time"
    assert not ((value_only or policy_only) and (mcts_rollouts is not None)), "Cannot use --v or --pi flags with MCTS"

    num_puzzles = sum([1 for _ in open(puzzles_path, "r")]) if num_puzzles is None else num_puzzles

    evaluation_results = puzzle_evaluate_training_trajectory(
        models_dir, model_type, puzzles_path, num_puzzles, mcts_rollouts, value_only, policy_only)

    if not out_file.endswith(".png"):
        out_file += ".png"

    plt.figure()
    plt.xlabel("Checkpoint")
    plt.ylabel("Accuracy")
    plt.title(f"Puzzle Evaluation [{num_puzzles}] For Training Trajectory")
    plt.plot([result.accuracy for result in evaluation_results])
    plt.xticks([i for i in range(1, len(evaluation_results) + 1)])
    # make y-axis percentages from 0 to 1
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))
    # Force the y-axis to start at 0 and end at 1
    plt.ylim(0, 1)
    plt.savefig(out_file)

