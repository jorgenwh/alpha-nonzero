import os
import chess


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

def fetch_training_data(fn, max_data_points=None):
    assert os.path.isfile(fn), f"File not found: {fn}"

    inputs = []
    targets = []

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

