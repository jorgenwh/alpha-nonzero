import os
import chess
import torch
from typing import Union

from .constants import DEVICE
from .helpers import load_model
from .inference import run_inference
from .types import InferenceType, PuzzleEvaluationResult

def play_puzzle(
        model: torch.nn.Module, 
        model_type: str, 
        fen: str, 
        target_sequence: list, 
        mcts_rollouts: Union[int, None], 
        value_only: bool, 
        policy_only: bool
) -> int:
    board = chess.Board(fen)
    other_turn = board.turn

    for target_move in target_sequence:
        if other_turn != board.turn:
            inference_result = run_inference(model, model_type, fen, mcts_rollouts, value_only, policy_only)
            move = inference_result.move
            assert move is not None, f"Invalid move returned from policy: {move}"
            if move != target_move:
                return 0

        board.push(chess.Move.from_uci(target_move))
        fen = board.fen()

    return 1

def puzzle_evaluate(
        model_path: str, 
        model_type: str, 
        puzzles_path: str, 
        num_puzzles: int,
        mcts_rollouts: Union[int, None], 
        value_only: bool, 
        policy_only: bool
) -> PuzzleEvaluationResult:
    model = load_model(model_path, model_type).to(DEVICE)
    solved = 0

    with open(puzzles_path, "r") as f:
        for i, puzzle in enumerate(f, start=1):
            fen, target_sequence, _ = puzzle.split(",")
            target_sequence = target_sequence.split(" ")
            solved += play_puzzle(model, model_type, fen, target_sequence, mcts_rollouts, value_only, policy_only)
            if i >= num_puzzles:
                break
            print(f"Puzzle {i:,}/{num_puzzles:,} - Solved: {solved:,} [{(solved/i)*100:.2f}%]", end="\r", flush=True)
        print(f"Puzzle {num_puzzles:,}/{num_puzzles:,} - Solved: {solved:,} [{(solved/num_puzzles)*100:.2f}%]")

    acc = solved/num_puzzles
    return PuzzleEvaluationResult(accuracy=acc, inference_type=InferenceType.UNKNOWN, mcts_rollouts=None)

def puzzle_evaluate_training_trajectory(
        models_dir: str,
        model_type: str,
        puzzles_path: str,
        num_puzzles: int,
        mcts_rollouts: Union[int, None],
        value_only: bool,
        policy_only: bool
) -> list:
    model_fns = sorted(os.listdir(models_dir))
    evaluation_results = []

    for i, model_fn in enumerate(model_fns, start=1):
        model_path = os.path.join(models_dir, model_fn)
        print(f"Evaluating model {i:,}/{len(model_fns):,}")
        result = puzzle_evaluate(model_path, model_type, puzzles_path, num_puzzles, mcts_rollouts, value_only, policy_only)
        evaluation_results.append(result)

    return evaluation_results
