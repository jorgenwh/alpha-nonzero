import os
import pickle
import torch
import chess
from constants import BLOCK_SIZE, CHAR_TO_IDX, VOCAB, POLICY_INDEX


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

def bins_to_p_wins(bins):
    return torch.argmax(bins, dim=1).float() / (bins.shape[1] - 1)

def tokenize_move(move: str) -> int:
    return len(VOCAB) + POLICY_INDEX.index(move)

def fetch_training_data(fn, num_bins, max_data_points=None):
    assert os.path.isfile(fn), f"File not found: {fn}"

    inputs = []
    targets = []

    dp = 0
    print("Fetching training data...")
    with open(fn, "rb") as f:
        while 1:
            try:
                fen, move, target = pickle.load(f)
            except EOFError:
                break
            except Exception as e:
                print(f"Unhandled error: {e}")
                exit()

            assert isinstance(fen, str), f"fen is not a string: {fen}"
            assert isinstance(move, str), f"move is not a string: {move}"
            assert isinstance(target, float), f"target is not a float: {target}"

            flfen = fen_to_fixed_length_fen(fen.strip())

            tokenized_input = torch.zeros(BLOCK_SIZE, dtype=torch.int64)
            for i, c in enumerate(flfen):
                tokenized_input[i] = CHAR_TO_IDX[c]
            tokenized_input[-1] = tokenize_move(move)

            inputs.append(tokenized_input)
            targets.append(target)
            dp += 1

            if max_data_points is not None and dp >= max_data_points:
                break

            print(f"Parsing data point {dp}/{'-' if max_data_points is None else max_data_points}", end="\r", flush=True)
        print(f"Parsing data point {dp}/{'-' if max_data_points is None else max_data_points}")

    inputs = torch.stack(inputs)
    targets = p_wins_to_bins(targets, num_bins)

    assert inputs.shape == (dp, BLOCK_SIZE), f"inputs shape {inputs.shape} is not correct"
    assert inputs.dtype == torch.int64, f"inputs dtype {inputs.dtype} is not correct"
    assert targets.shape == (dp, num_bins), f"targets shape {targets.shape} is not correct"
    assert targets.dtype == torch.float32, f"targets dtype {targets.dtype} is not correct"
            
    return inputs, targets


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
    x, y = fetch_training_data("data/training_data.pkl", num_bins=32, max_data_points=None)
    print(x.shape)
    print(x.dtype)
    print(y.shape)
    print(y.dtype)
