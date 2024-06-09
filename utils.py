from collections import deque
import pickle
import numpy as np
import torch
import chess
from constants import POLICY_INDEX


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

def p_wins_to_bins(p_wins, num_bins):
    assert num_bins in [16, 32, 64, 128, 256], "num_bins must be one of [16, 32, 64, 128, 256]"

    bin_vecs = torch.zeros((len(p_wins), num_bins), dtype=torch.float32)
    for i, p_win in enumerate(p_wins):
        bin_idx = round(p_win * (num_bins-1))
        bin_vecs[i, bin_idx] = 1.0

    return bin_vecs

def get_batch(batch_num):
    data = pickle_load(f"training_data/training_batch_{batch_num}.pkl")

    positions = deque()
    targets = deque()

    for (fen, move, p_win) in data:
        p_win = float(p_win)
        assert isinstance(fen, str)
        assert isinstance(move, str)
        assert isinstance(p_win, float)

        pos_vec = torch.zeros(BLOCK_SIZE, dtype=torch.int64)
        flfen = fen_to_fixed_length_fen(fen)
        for i, c in enumerate(flfen):
            pos_vec[i] = CHAR_TO_IDX[c]
        pos_vec[-1] = policy_index.index(move)

        positions.append(pos_vec)
        targets.append(p_win)

    return positions, targets
    

def prepare_training_data(sbn, ebn, num_bins, max_data_points=5000):
    batches = (ebn - sbn) + 1
    print(f"Preparing batches... batch 0/{batches}", end="\r", flush=True)

    positions = deque(maxlen=max_data_points)
    targets = deque(maxlen=max_data_points)

    for idx, i in enumerate(range(sbn, ebn+1)):
        ps, ts = get_batch(i)
        positions.extend(ps)
        targets.extend(ts)
        if len(positions) >= max_data_points:
            break
        print(f"Preparing batches... batch {idx+1}/{batches} - DP: {len(positions)}/{max_data_points}", end="\r", flush=True)
    print(f"Preparing batches... batch -/{batches} - DP: {len(positions)}/{max_data_points}")

    positions = torch.stack(list(positions))
    #targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
    targets = p_wins_to_bins(targets, num_bins)

    assert positions.dtype == torch.int64
    assert targets.dtype == torch.float32
    assert len(positions.shape) == 2
    assert positions.shape[1] == BLOCK_SIZE
    #assert targets.shape == (len(positions), 1)
    assert targets.shape == (len(positions), num_bins)
    
    return positions, targets

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
    #fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    #fixed_length_fen = fen_to_fixed_length_fen(fen)

    #print(f"original fen: {fen}")
    #print(f"fixed-length fen: {fixed_length_fen}")

    values = np.linspace(0, 1, 16)
    bins = p_wins_to_bins(values, 16)
    print(values)
    print(bins)


