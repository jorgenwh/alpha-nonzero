import sys
import torch
import chess

from model import Transformer
from utils import fen_to_fixed_length_fen, bins_to_p_wins, tokenize_move
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

    fen = sys.argv[1]
    flfen = fen_to_fixed_length_fen(fen)
    tokenized_position = torch.zeros(1, BLOCK_SIZE, dtype=torch.int64)
    for i, c in enumerate(flfen):
        tokenized_position[0, i] = CHAR_TO_IDX[c]
    tokenized_position = tokenized_position.to(DEVICE)

    board = chess.Board(fen)
    legal_moves = board.generate_legal_moves()

    net_eval = {}
    for move in legal_moves:
        tokenized_position[0, -1] = tokenize_move(str(move))

        output = model(tokenized_position)
        value = bins_to_p_wins(output).item()

        net_eval[str(move)] = value


    for move in sorted(net_eval, key=net_eval.get, reverse=True):
        print(f"{move}: {round(net_eval[move], 4)}")
