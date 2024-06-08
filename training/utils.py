import pickle
import numpy as np
import torch
import chess
from policy_index import policy_index


VOCAB          = sorted(list(set(c for c in "PpRrNnBbQqKkabcdefgh12345678wb.09")))
VOCAB_SIZE     = len(VOCAB) + len(policy_index)
BLOCK_SIZE     = 77
POLICY_SIZE    = len(policy_index)

CHAR_TO_IDX    = {c:i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR    = {i:c for i, c in enumerate(VOCAB)}


def fen_to_fixed_length_fen(fen: str) -> str:
    # ----------------------------------------------------------
    # Fixed-length fen description:

    # 64  square/piece tokens
    # 1   turn token
    # 4   castling tokens
    # 2   en passant tokens
    # 2   halfmove clock tokens
    # 3   fullmove clock tokens

    # 64 + 1 + 4 + 2 + 2 + 3 = 76
    # Why does Google DeepMind say 77 in their paper? ¯\_(ツ)_/¯
    # ----------------------------------------------------------
    flfen = ""

    board = chess.Board(fen)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            flfen += str(piece)
        else:
            flfen += "."

    turn = "w" if board.turn else "b"
    flfen += turn

    white_kingside_castling = "K" if board.has_kingside_castling_rights(chess.WHITE) else "."
    white_queenside_castling = "Q" if board.has_queenside_castling_rights(chess.WHITE) else "."
    black_kingside_castling = "k" if board.has_kingside_castling_rights(chess.BLACK) else "."
    black_queenside_castling = "q" if board.has_queenside_castling_rights(chess.BLACK) else "."
    castling = white_kingside_castling + white_queenside_castling + black_kingside_castling + black_queenside_castling
    flfen += castling

    ep_sq = board.ep_square
    if ep_sq is not None:
        flfen += chess.SQUARE_NAMES[ep_sq]
    else:
        flfen += ".."

    halfmove_clock = str(board.halfmove_clock)
    if len(halfmove_clock) == 1:
        flfen += "." + halfmove_clock
    elif len(halfmove_clock) == 2:
        flfen += halfmove_clock
    else:
        raise ValueError(f"halfmove clock is not 1 or 2 digits long. fen: {fen}")

    fullmove_clock = str(board.fullmove_number)
    if len(fullmove_clock) == 1:
        flfen += ".." + fullmove_clock
    elif len(fullmove_clock) == 2:
        flfen += "." + fullmove_clock
    elif len(fullmove_clock) == 3:
        flfen += fullmove_clock
    else:
        raise ValueError("fullmove clock is not 1, 2 or 3 digits long")

    assert len(flfen) == 76, f"fixed-length fen is not 76 characters long: {flfen}"
    return flfen

def _prepare_batch(batch_num):
    f = open(f"training_data/fen_batch_{batch_num}.fen")
    fens = [fen for fen in f]
    flfens = [fen_to_fixed_length_fen(fen) for fen in fens]
    f.close()

    positions = torch.zeros((len(flfens), 76), dtype=torch.int64)
    for i, flfen in enumerate(flfens):
        positions[i] = torch.tensor([CHAR_TO_IDX[c] for c in flfen])
    annotations = torch.tensor(np.load(f"training_data/annotation_batch_{batch_num}.anno.npy"))

    assert positions.dtype == torch.int64
    assert annotations.dtype == torch.float32

    return positions, annotations

def prepare_training_data(sbn, ebn):
    batches = (ebn - sbn) + 1
    print(f"Preparing batches... batch 0/{batches}", end="\r", flush=True)

    positions = torch.zeros(size=(batches*1000, 76), dtype=torch.int64)
    annotations = torch.zeros(size=(batches*1000, 1968), dtype=torch.float32)

    for idx, i in enumerate(range(sbn, ebn+1)):
        pos, annos = _prepare_batch(i)
        s = idx*1000
        e = (idx + 1)*1000
        positions[s:e] = pos
        annotations[s:e] = annos
        print(f"Preparing batches... batch {idx+1}/{batches}", end="\r", flush=True)
    print(f"Preparing batches... batch {batches}/{batches}")

    return positions, annotations

def pickle_save(data: list, fn: str):
    with open(fn, "wb") as f:
        pickle.dump(data, f)

def pickle_load(fn: str):
    with open(fn, "rb") as f:
        data = pickle.load(f)
    return data


class AverageMeter():
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{round(self.avg, 4)}"

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fixed_length_fen = fen_to_fixed_length_fen(fen)

    print(f"original fen: {fen}")
    print(f"fixed-length fen: {fixed_length_fen}")
