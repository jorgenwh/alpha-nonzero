import chess
import torch
from typing import Tuple, Union

from anz.helpers import load_model, fen2vec, flip_fen_if_black_turn
from anz.constants import DEVICE, POLICY_INDEX
from anz.types import InferenceResult


def forward_pass(model: torch.nn.Module, model_type: str, fen: str) -> Tuple[torch.Tensor, torch.Tensor]:
    fen_vec = fen2vec(fen, model_type).unsqueeze(0).to(DEVICE)
    pi, v = model(fen_vec)
    return pi, v

def policy_vec_to_move(pi: torch.Tensor) -> str:
    move_idx = int(torch.argmax(torch.softmax(pi, dim=1)).item())
    move = POLICY_INDEX[move_idx]
    return move

def value_vec_to_move(v: torch.Tensor) -> float:
    return v.item()

def run_value_head_policy_inference(model: torch.nn.Module, model_type: str, fen: str) -> InferenceResult:
    board = chess.Board(fen)
    best_move = None
    best_value = None

    for move in board.legal_moves:
        board.push(move)
        fen = board.fen()
        _, v = forward_pass(model, model_type, fen)
        value = value_vec_to_move(v)
        if best_value is None or value > best_value:
            best_move = str(move)
            best_value = value
        board.pop()

    return InferenceResult(move=best_move, value=None)

def run_policy_head_policy_inference(model: torch.nn.Module, model_type: str, fen: str) -> InferenceResult:
    pi, _ = forward_pass(model, model_type, fen)
    move = policy_vec_to_move(pi)
    return InferenceResult(move=move, value=None)

def run_mcts_policy_inference(model: torch.nn.Module, model_type: str, fen: str, mcts: int) -> InferenceResult:
    raise NotImplementedError("MCTS policy inference not implemented")

def run_raw_inference(model: torch.nn.Module, model_type: str, fen: str) -> InferenceResult:
    pi, v = forward_pass(model, model_type, fen)
    move = policy_vec_to_move(pi)
    value = value_vec_to_move(v)
    return InferenceResult(move=move, value=value)

def run_inference(model_path: str, model_type: str, fen: str, mcts: Union[int, None], value_only: bool, policy_only: bool) -> InferenceResult:
    model = load_model(model_path, model_type).to(DEVICE)
    corrected_fen = flip_fen_if_black_turn(fen)

    if value_only:
        print(f"Running value-head-only policy inference for FEN: {fen} with model: {model_path}")
        return run_value_head_policy_inference(model, model_type, corrected_fen)
    if policy_only:
        print(f"Running single-pass policy-head-only inference for FEN: {fen} with model: {model_path}")
        return run_policy_head_policy_inference(model, model_type, corrected_fen)
    if mcts is not None:
        print(f"Running MCTS [{mcts}] policy inference for FEN: {fen} with model: {model_path}")
        return run_mcts_policy_inference(model, model_type, corrected_fen, mcts)
    else:
        print(f"Running single-pass raw inference for FEN: {fen} with model: {model_path}")
        return run_raw_inference(model, model_type, corrected_fen)

