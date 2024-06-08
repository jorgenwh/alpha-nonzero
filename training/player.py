import chess
import sys
import torch
from annotate import annotate_single_position
from policy_index import policy_index
from model import Transformer
from utils import fen_to_fixed_length_fen, CHAR_TO_IDX, VOCAB_SIZE

def stockfish_play(fen: str):
    annotation = annotate_single_position(fen)
    annotation = torch.tensor(annotation)
    assert annotation.dtype == torch.float32
    assert annotation.shape == (1, len(policy_index))
    return annotation

def neural_play(fen: str):
    flfen = fen_to_fixed_length_fen(fen)
    position = torch.LongTensor([CHAR_TO_IDX[c] for c in flfen]).unsqueeze(0)
    assert position.shape == (1, 76)
    assert position.dtype == torch.int64
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        policy_size=len(policy_index),
        block_size=76,
        d_model=512,
        n_heads=8,
        n_blocks=6,
        device='cpu',
    )
    model.load_state_dict(torch.load("models/model_checkpoint_28000.pt"))
    output = model(position)
    assert output.dtype == torch.float32
    assert output.shape == (1, len(policy_index))

    board = chess.Board(fen)
    legal_moves = [m for m in board.legal_moves]
    mask = torch.zeros_like(output)
    mask[:,:] = torch.inf
    for m in legal_moves:
        idx = policy_index.index(str(m))
        mask[0, idx] = 0
    output += mask
    return output


if __name__ == "__main__":
    fen = sys.argv[1]

    engine = "net"
    assert engine in ["sf", "net"], "Invalid engine. Choose 'stockfish' or 'neural'."

    if engine == "sf":
        policy = stockfish_play(fen)
        engine = "STOCKFISH"
    elif engine == "net":
        policy = neural_play(fen)
        engine = "NEURAL"

    top3_indices = torch.topk(policy, k=3, largest=False).indices[0]
    top3 = [policy_index[int(i)] for i in top3_indices]
    print(f"[engine={engine}] fen: {fen}\ntop-3: {top3}")

