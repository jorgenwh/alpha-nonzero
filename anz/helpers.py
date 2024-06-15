import os
from anz.constants import BLOCK_SIZE, BOARD_CONV_CHANNELS, CHAR_TO_IDX
import torch
import chess


MODEL_TYPES = ["transformer", "convolutional"]

PIECE_TO_BOARD_CHANNEL = {
    "P": 0, 
    "N": 1, 
    "B": 2, 
    "R": 3, 
    "Q": 4, 
    "K": 5, 
    "p": 6, 
    "n": 7, 
    "b": 8, 
    "r": 9, 
    "q": 10, 
    "k": 11
}
W_KINGSIDE_CASTLING_BOARD_CHANNEL   = 12
W_QUEENSIDE_CASTLING_BOARD_CHANNEL  = 13
B_KINGSIDE_CASTLING_BOARD_CHANNEL   = 14
B_QUEENSIDE_CASTLING_BOARD_CHANNEL  = 15
EN_PASSANT_BOARD_CHANNEL            = 16
HALFMOVE_CLOCK_BOARD_CHANNEL        = 17
FULLMOVE_CLOCK_BOARD_CHANNEL        = 18


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

def fens2vec_transformer(fens: list) -> torch.Tensor:
    B = len(fens)
    flfens = [fen_to_fixed_length_fen(fen) for fen in fens]
    output = torch.zeros(size=(B, BLOCK_SIZE), dtype=torch.int64)

    for i, fen in enumerate(flfens):
        for j, c in enumerate(fen):
            output[i,j] = CHAR_TO_IDX[c]

    return output

def fens2vec_convolutional(fens: list):
    B = len(fens)
    output = torch.zeros(size=(B, BOARD_CONV_CHANNELS, 8, 8), dtype=torch.float32)

    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            
            if piece is None:
                continue

            channel = PIECE_TO_BOARD_CHANNEL[str(piece)]
            rank = square % 8
            file = square // 8

            output[i, channel, file, rank] = 1

        if board.has_kingside_castling_rights(chess.WHITE):
            output[i, W_KINGSIDE_CASTLING_BOARD_CHANNEL, :, :] = 1
        if board.has_queenside_castling_rights(chess.WHITE):
            output[i, W_QUEENSIDE_CASTLING_BOARD_CHANNEL, :, :] = 1
        if board.has_kingside_castling_rights(chess.BLACK):
            output[i, B_KINGSIDE_CASTLING_BOARD_CHANNEL, :, :] = 1
        if board.has_queenside_castling_rights(chess.BLACK):
            output[i, B_QUEENSIDE_CASTLING_BOARD_CHANNEL, :, :] = 1

        ep_square = board.ep_square
        if ep_square is not None:
            file = ep_square // 8
            rank = ep_square % 8
            output[i, EN_PASSANT_BOARD_CHANNEL, file, rank] = 1

        halfmove_clock = board.halfmove_clock
        fullmove_clock = board.fullmove_number

        output[i, HALFMOVE_CLOCK_BOARD_CHANNEL, :, :] = halfmove_clock
        output[i, FULLMOVE_CLOCK_BOARD_CHANNEL, :, :] = fullmove_clock

    return output

def fens2vec(fens: list, model_type: str = "transformer") -> torch.Tensor:
    assert model_type in MODEL_TYPES, f"model_type must be one of {MODEL_TYPES}, not '{model_type}'"

    if model_type == "transformer":
        return fens2vec_transformer(fens)
    if model_type == "convolutional":
        return fens2vec_convolutional(fens)

    assert False, f"Invalid model_type: '{model_type}'"

def fetch_training_data(fn):
    assert os.path.isfile(fn), f"File not found: {fn}"

    fens = []

    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        if not board.turn:
            fens[i] = board.mirror().transform(chess.flip_horizontal).fen()

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

