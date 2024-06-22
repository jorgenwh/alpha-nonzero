import chess
import torch
from typing import Union

from anz.constants import DEVICE
from anz.helpers import (
    load_model
)
from anz.inference import (
    run_value_head_policy_inference, 
    run_policy_head_policy_inference, 
    run_mcts_policy_inference
)
from anz.types import (
    InferenceResult, 
    PuzzleEvaluationResult
)

def get_policy(
        model: torch.nn.Module, 
        model_type: str, 
        fen: str, 
        mcts: Union[int, None], 
        value_only: bool, 
        policy_only: bool
) -> InferenceResult:
    if value_only:
        return run_value_head_policy_inference(model, model_type, fen)
    if policy_only:
        return run_policy_head_policy_inference(model, model_type, fen)
    if mcts is not None:
        return run_mcts_policy_inference(model, model_type, fen, mcts)
    assert False, "No policy function selected. Choose either value-head-only [--v], policy-head-only [--pi], or MCTS [-mcts <num_simulations>]"

def play_puzzle(
        model: torch.nn.Module, 
        model_type: str, 
        fen: str, 
        target_sequence: list, 
        mcts: Union[int, None], 
        value_only: bool, 
        policy_only: bool
) -> int:
    board = chess.Board(fen)
    other_turn = board.turn

    for target_move in target_sequence:
        if other_turn != board.turn:
            inference_result = get_policy(model, model_type, fen, mcts, value_only, policy_only)
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
        num_puzzles: Union[int, None], 
        mcts: Union[int, None], 
        value_only: bool, 
        policy_only: bool
) -> PuzzleEvaluationResult:
    model = load_model(model_path, model_type).to(DEVICE)
    solved = 0
    num_puzzles = sum([1 for _ in open(puzzles_path, "r")]) if num_puzzles is None else num_puzzles

    with open(puzzles_path, "r") as f:
        for i, puzzle in enumerate(f, start=1):
            fen, target_sequence, _ = puzzle.split(",")
            target_sequence = target_sequence.split(" ")
            solved += play_puzzle(model, model_type, fen, target_sequence, mcts, value_only, policy_only)
            if i >= num_puzzles:
                break
            print(f"Puzzle {i:,}/{num_puzzles:,} - Solved: {solved:,} [{(solved/i)*100:.2f}%]", end="\r", flush=True)
        print(f"Puzzle {num_puzzles:,}/{num_puzzles:,} - Solved: {solved:,} [{(solved/num_puzzles)*100:.2f}%]")

    acc = solved/num_puzzles
    return PuzzleEvaluationResult(accuracy=acc)

