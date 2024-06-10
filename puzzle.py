import chess
import torch

from model import Transformer
from utils import bins_to_p_wins, tokenize_move, fen_to_fixed_length_fen
from constants import (
    VOCAB_SIZE,
    BLOCK_SIZE,
    NUM_BINS,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    DEVICE,
    CHAR_TO_IDX,
)

def get_network_move(network, fen):
    flfen = fen_to_fixed_length_fen(fen)
    tokenized_position = torch.zeros(1, BLOCK_SIZE, dtype=torch.int64)
    for i, c in enumerate(flfen):
        tokenized_position[0, i] = CHAR_TO_IDX[c]
    tokenized_position = tokenized_position.to(DEVICE)

    board = chess.Board(fen)
    is_white = board.turn
    legal_moves = board.generate_legal_moves()

    policy = {}
    for move in legal_moves:
        tokenized_position[0, -1] = tokenize_move(str(move))
        output = network(tokenized_position)
        value = bins_to_p_wins(output).item()
        policy [str(move)] = value

    sorted_moves = [k for k, _ in sorted(policy.items(), key=lambda item: item[1], reverse=True)]
    return sorted_moves[0] if is_white else sorted_moves[-1]

def play_puzzle(network, fen, target_sequence):
    board = chess.Board(fen)
    other_turn = board.turn

    for target_move in target_sequence:
        if other_turn != board.turn:
            network_move = get_network_move(network, fen)
            if network_move != target_move:
                return 0

        board.push(chess.Move.from_uci(target_move))
        fen = board.fen()

    return 1

def puzzle_evaluate(network, puzzle_filename):
    attempted = 0
    solved = 0
    with open(puzzle_filename, "r") as f:
        for puzzle in f:
            fen, target_sequence, rating = puzzle.split(",")
            target_sequence = target_sequence.split(" ")
            solved += play_puzzle(network, fen, target_sequence)
            attempted += 1
            print(f"Puzzle {attempted}/10000 - Solved: {solved} [{(solved/attempted)*100:.2f}%]", end="\r", flush=True)
        print(f"Puzzle 10000/10000 - Solved: {solved} [{(solved/attempted)*100:.2f}%]")
    return solved/attempted


if __name__ == "__main__":
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        output_size=NUM_BINS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS
    )
    model.load_state_dict(torch.load("models/model_checkpoint_367000.pt"))
    model.eval()
    model = model.to(DEVICE)

    p = puzzle_evaluate(model, "lichess_puzzles.csv")
    print(f"Solved {p*100:.2f}% of the puzzles")
