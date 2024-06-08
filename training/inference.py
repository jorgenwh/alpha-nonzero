import sys
import torch
import chess

from model import Transformer
from policy_index import policy_index
from utils import fen_to_fixed_length_fen, VOCAB_SIZE, CHAR_TO_IDX


BLOCK_SIZE     = 76
D_MODEL        = 512
N_HEADS        = 8
N_BLOCKS       = 6
BATCH_SIZE     = 64
POLICY_SIZE    = len(policy_index)
DEVICE         = "cuda:0"
LEARNING_RATE  = 0.001
TRAINING_ITERS = 300


if __name__ == "__main__":
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        policy_size=POLICY_SIZE,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_blocks=N_BLOCKS,
        device=DEVICE,
    ).to(DEVICE)
    model.load_state_dict(torch.load("models/model_checkpoint_28000.pt"))
    model.eval()

    fen = sys.argv[1]
    flfen = fen_to_fixed_length_fen(fen)
    position = torch.tensor([CHAR_TO_IDX[c] for c in flfen]).reshape(1, -1).to(DEVICE)

    # get valid moves
    board = chess.Board(fen)
    legal_moves = board.generate_legal_moves()
    valid_move_vec = torch.zeros(len(policy_index), dtype=torch.float32)
    for move in legal_moves:
        idx = policy_index.index(str(move))
        valid_move_vec[idx] = 1
    valid_move_vec = valid_move_vec.to(DEVICE)

    output = model(position)
    output *= valid_move_vec
    print(policy_index[int(torch.argmax(output).item())])
