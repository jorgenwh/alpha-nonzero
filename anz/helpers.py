import psutil
import chess
import torch

from anz.policy_index import policy_index
from anz.constants import BLOCK_SIZE, BOARD_CONV_CHANNELS, CHAR_TO_IDX, POLICY_SIZE

DTYPE_BYTE_SIZES = {
    torch.float32   : 4,
    torch.int64     : 8
}

MODEL_TYPES = ["transformer", "resnet"]

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

MIRROR_RANK_MAP = {
    "1": "8",
    "2": "7",
    "3": "6",
    "4": "5",
    "5": "4",
    "6": "3",
    "7": "2",
    "8": "1"
}
MIRROR_FILE_MAP = {
    "a": "h",
    "b": "g",
    "c": "f",
    "d": "e",
    "e": "d",
    "f": "c",
    "g": "b",
    "h": "a"
}


def flip_chess_move(move: str) -> str:
    assert len(move) == 4 or len(move) == 5, f"Invalid move format: '{move}'"
    if len(move) == 4:
        from_file, from_rank, to_file, to_rank, promotion = move[0], move[1], move[2], move[3], ""
    else:
        from_file, from_rank, to_file, to_rank, promotion = move[0], move[1], move[2], move[3], move[4]

    from_file = MIRROR_FILE_MAP[from_file]
    from_rank = MIRROR_RANK_MAP[from_rank]
    to_file = MIRROR_FILE_MAP[to_file]
    to_rank = MIRROR_RANK_MAP[to_rank]

    return from_file + from_rank + to_file + to_rank + promotion

def allocate_zero_tensor(size, dtype) -> torch.Tensor:
    available_bytes = psutil.virtual_memory().available
    needed_bytes = DTYPE_BYTE_SIZES[dtype]
    if isinstance(size, int):
        needed_bytes *= size
    else:
        for s in size:
            needed_bytes *= s

    if needed_bytes > available_bytes:
        raise MemoryError(f"Insufficient memory to allocate tensor of size {size} ({needed_bytes} bytes needed)")

    return torch.zeros(size, dtype=dtype)


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

def fen2vec_transformer(fen: str) -> torch.Tensor:
    vec = allocate_zero_tensor((BLOCK_SIZE), torch.int64)
    flfen = fen_to_fixed_length_fen(fen)

    for i, c in enumerate(flfen):
        vec[i] = CHAR_TO_IDX[c]

    return vec

def fen2vec_resnet(fen: str) -> torch.Tensor:
    vec = allocate_zero_tensor((BOARD_CONV_CHANNELS, 8, 8), torch.float32)
    board = chess.Board(fen)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        
        if piece is None:
            continue

        channel = PIECE_TO_BOARD_CHANNEL[str(piece)]
        rank = square % 8
        file = square // 8

        vec[channel, file, rank] = 1

    if board.has_kingside_castling_rights(chess.WHITE):
        vec[W_KINGSIDE_CASTLING_BOARD_CHANNEL, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        vec[W_QUEENSIDE_CASTLING_BOARD_CHANNEL, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        vec[B_KINGSIDE_CASTLING_BOARD_CHANNEL, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        vec[B_QUEENSIDE_CASTLING_BOARD_CHANNEL, :, :] = 1

    ep_square = board.ep_square
    if ep_square is not None:
        file = ep_square // 8
        rank = ep_square % 8
        vec[EN_PASSANT_BOARD_CHANNEL, file, rank] = 1

    halfmove_clock = board.halfmove_clock
    fullmove_clock = board.fullmove_number

    vec[HALFMOVE_CLOCK_BOARD_CHANNEL, :, :] = halfmove_clock
    vec[FULLMOVE_CLOCK_BOARD_CHANNEL, :, :] = fullmove_clock

    return vec

def fen2vec(fen: str, model_type: str = "transformer") -> torch.Tensor:
    assert model_type in MODEL_TYPES, f"model_type must be one of {MODEL_TYPES}, not '{model_type}'"

    if model_type == "transformer":
        return fen2vec_transformer(fen)
    if model_type == "resnet":
        return fen2vec_resnet(fen)

    assert False, f"Invalid model_type: '{model_type}'"

def move2vec(move: str) -> torch.Tensor:
    index = policy_index.index(move)
    vec = allocate_zero_tensor((POLICY_SIZE), torch.float32)
    vec[index] = 1
    return vec

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

