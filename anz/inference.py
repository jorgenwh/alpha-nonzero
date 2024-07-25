import chess
import torch
from typing import Union

from .mcts import MCTS
from .helpers import load_model, flip_fen_if_black_turn, flip_inference_result, model_forward_pass
from .constants import DEVICE, POLICY_INDEX
from .types import InferenceResult, InferenceType


def policy_vec_to_move(pi: torch.Tensor) -> str:
    move_idx = int(torch.argmax(torch.softmax(pi, dim=1)).item())
    move = POLICY_INDEX[move_idx]
    return move

def run_value_head_policy_inference(model: torch.nn.Module, model_type: str, fen: str) -> InferenceResult:
    board = chess.Board(fen)
    best_move = None
    best_value = None

    for move in board.legal_moves:
        board.push(move)
        fen = board.fen()
        _, v = model_forward_pass(model, model_type, fen)
        value = v.item()
        if best_value is None or value > best_value:
            best_move = str(move)
            best_value = value
        board.pop()

    return InferenceResult(
        fen=fen, move=best_move, top5=None, value=None, inference_type=InferenceType.VALUE_ONLY, mcts_rollouts=None)

def run_policy_head_policy_inference(model: torch.nn.Module, model_type: str, fen: str) -> InferenceResult:
    pi, _ = model_forward_pass(model, model_type, fen)
    move = policy_vec_to_move(pi)
    return InferenceResult(
        fen=fen, move=move, value=None, top5=None, inference_type=InferenceType.POLICY_ONLY, mcts_rollouts=None)

def run_mcts_policy_inference(
        model: torch.nn.Module, 
        model_type: str, 
        fen: str, 
        rollouts: int, 
        verbose: bool = False
) -> InferenceResult:
    mcts = MCTS(model, model_type)
    inference_result = mcts.go(fen, rollouts, verbose=verbose)
    return inference_result

def run_raw_inference(model: torch.nn.Module, model_type: str, fen: str) -> InferenceResult:
    pi, v = model_forward_pass(model, model_type, fen)
    move = policy_vec_to_move(pi)
    value = v.item()
    return InferenceResult(
        fen=fen, move=move, value=value, top5=None, inference_type=InferenceType.RAW, mcts_rollouts=None)

def run_inference(
        model: Union[torch.nn.Module, str], 
        model_type: str, 
        fen: str, 
        mcts_rollouts: Union[int, None], 
        value_only: bool, 
        policy_only: bool,
        verbose: bool = False
) -> InferenceResult:
     
    model = load_model(model, model_type).to(DEVICE) if isinstance(model, str) else model.to(DEVICE)
    corrected_fen = flip_fen_if_black_turn(fen)

    if value_only:
        result = run_value_head_policy_inference(model, model_type, corrected_fen)
    elif policy_only:
        result = run_policy_head_policy_inference(model, model_type, corrected_fen)
    elif mcts_rollouts is not None:
        result = run_mcts_policy_inference(model, model_type, corrected_fen, mcts_rollouts, verbose=verbose)
    else:
        result = run_raw_inference(model, model_type, corrected_fen)

    if corrected_fen != fen:
        return flip_inference_result(result)
    return result

