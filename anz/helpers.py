import psutil
import chess
import torch
import copy
from collections import OrderedDict
from typing import Union, Tuple, List

from .models import Transformer, ResNet
from .types import InferenceResult
from .constants import *


DTYPE_BYTE_SIZES = {
    torch.bfloat16   : 2,
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


def load_model(model_path: str, model_type: str) -> torch.nn.Module:
    model = None
    if model_type == "transformer":
        model = Transformer()
    if model_type == "resnet":
        model = ResNet()
    assert model is not None, f"Model type {model_type} not supported"

    try:
        checkpoint = torch.load(model_path)
    except Exception as e:
        raise Exception(f"Error loading model data from {model_path}: {e}")

    model_state_dict = checkpoint["model_state_dict"]
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k
        if name.startswith("_orig_mod."):
            name = name[10:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    return model

def flip_fen(fen: str) -> str:
    board = chess.Board(fen)
    castling_fen = ""
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_fen += "K"
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_fen += "Q"
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_fen += "k"
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_fen += "q"
    if castling_fen == "":
        castling_fen = "-"
    board = board.mirror().transform(chess.flip_horizontal)
    flipped_fen_no_castling_rights = board.fen()
    before_castling_rights = flipped_fen_no_castling_rights.split(" ")[:2]
    after_castling_rights = flipped_fen_no_castling_rights.split(" ")[3:]
    return " ".join(before_castling_rights) + " " + castling_fen + " " + " ".join(after_castling_rights)

def flip_fen_if_black_turn(fen: str) -> str:
    board = chess.Board(fen)
    if board.turn == chess.BLACK:
        return flip_fen(fen)
    return fen

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

def flip_inference_result(result: InferenceResult) -> InferenceResult:
    flipped_result = InferenceResult(
        fen=flip_fen(result.fen), 
        move=None, 
        value=None, 
        top5=None, 
        inference_type=result.inference_type, 
        mcts_rollouts=result.mcts_rollouts
    ) 
    if result.move is not None:
        flipped_result.move = flip_chess_move(result.move)
    if result.value is not None:
        flipped_result.value = -result.value
    if result.top5 is not None:
        flipped_result.top5 = [(flip_chess_move(move), value) for move, value in result.top5]
    return flipped_result

def allocate_zero_tensor(
        size: Union[int, Tuple], 
        dtype: torch.dtype
) -> torch.Tensor:
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

def fens2vec_transformer(fens: List[str]) -> torch.Tensor:
    vec = allocate_zero_tensor((len(fens), BLOCK_SIZE), torch.int64)

    for i, fen in enumerate(fens):
        board = chess.Board(fen)

        for j, square in enumerate(chess.SQUARES):
            piece = board.piece_at(square)
            if piece is not None:
                vec[i, j] = CHAR_TO_IDX[str(piece)]
            else:
                vec[i, j] = EMPTY_TOKEN

        turn = "w" if board.turn else "b"
        vec[i, 64] = WL_TOKEN if turn else BL_TOKEN

        vec[i, 65] = KU_TOKEN if board.has_kingside_castling_rights(chess.WHITE) else EMPTY_TOKEN
        vec[i, 66] = QU_TOKEN if board.has_queenside_castling_rights(chess.WHITE) else EMPTY_TOKEN
        vec[i, 67] = KL_TOKEN if board.has_kingside_castling_rights(chess.BLACK) else EMPTY_TOKEN
        vec[i, 68] = QL_TOKEN if board.has_queenside_castling_rights(chess.BLACK) else EMPTY_TOKEN

        ep_sq = board.ep_square
        if ep_sq is not None:
            sq = chess.SQUARE_NAMES[ep_sq]
            vec[i, 69] = CHAR_TO_IDX[sq[0]]
            vec[i, 70] = CHAR_TO_IDX[sq[1]]
        else:
            vec[i, 69:71] = EMPTY_TOKEN

        halfmove_clock = str(board.halfmove_clock)
        if len(halfmove_clock) == 1:
            vec[i, 71] = EMPTY_TOKEN
            vec[i, 72] = CHAR_TO_IDX[halfmove_clock]
        elif len(halfmove_clock) == 2:
            vec[i, 71] = CHAR_TO_IDX[halfmove_clock[0]]
            vec[i, 72] = CHAR_TO_IDX[halfmove_clock[1]]
        else:
            raise ValueError(f"halfmove clock is not 1 or 2 digits long. fen: {fen}")

        fullmove_clock = str(board.fullmove_number)
        if len(fullmove_clock) == 1:
            vec[i, 73] = EMPTY_TOKEN
            vec[i, 74] = EMPTY_TOKEN
            vec[i, 75] = CHAR_TO_IDX[fullmove_clock]
        elif len(fullmove_clock) == 2:
            vec[i, 73] = EMPTY_TOKEN
            vec[i, 74] = CHAR_TO_IDX[fullmove_clock[0]]
            vec[i, 75] = CHAR_TO_IDX[fullmove_clock[1]]
        elif len(fullmove_clock) == 3:
            vec[i, 73] = CHAR_TO_IDX[fullmove_clock[0]]
            vec[i, 74] = CHAR_TO_IDX[fullmove_clock[1]]
            vec[i, 75] = CHAR_TO_IDX[fullmove_clock[2]]
        else:
            raise ValueError("fullmove clock is not 1, 2 or 3 digits long")

    return vec

def fens2vec_resnet(fens: List[str], fp16: bool = False) -> torch.Tensor:
    size = (len(fens), BOARD_CONV_CHANNELS, 8, 8)
    dtype = torch.bfloat16 if fp16 else torch.float32
    vec = allocate_zero_tensor(size, dtype)

    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            channel = PIECE_TO_BOARD_CHANNEL[str(piece)]
            rank = square % 8
            file = square // 8

            vec[i, channel, file, rank] = 1

        vec[i, W_KINGSIDE_CASTLING_BOARD_CHANNEL, :, :] = int(board.has_kingside_castling_rights(chess.WHITE))
        vec[i, W_QUEENSIDE_CASTLING_BOARD_CHANNEL, :, :] = int(board.has_queenside_castling_rights(chess.WHITE))
        vec[i, B_KINGSIDE_CASTLING_BOARD_CHANNEL, :, :] = int(board.has_kingside_castling_rights(chess.BLACK))
        vec[i, B_QUEENSIDE_CASTLING_BOARD_CHANNEL, :, :] = int(board.has_queenside_castling_rights(chess.BLACK))

        ep_square = board.ep_square
        if ep_square is not None:
            file = ep_square // 8
            rank = ep_square % 8
            vec[i, EN_PASSANT_BOARD_CHANNEL, file, rank] = 1

        vec[i, HALFMOVE_CLOCK_BOARD_CHANNEL, :, :] = board.halfmove_clock
        vec[i, FULLMOVE_CLOCK_BOARD_CHANNEL, :, :] = board.fullmove_number

    return vec

def fens2vec(fens: List[str], model_type: str = "transformer", fp16: bool = False) -> torch.Tensor:
    assert model_type in MODEL_TYPES, f"model_type must be one of {MODEL_TYPES}, not '{model_type}'"

    if model_type == "transformer":
        return fens2vec_transformer(fens)
    if model_type == "resnet":
        return fens2vec_resnet(fens, fp16)

    assert False, f"Invalid model_type: '{model_type}'"

def model_forward_pass(
        model: torch.nn.Module, 
        model_type: str, 
        fen: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    fen_vec = fens2vec([fen], model_type, fp16=False).to(DEVICE)
    with torch.no_grad():
        pi, v = model(fen_vec)
    return pi, v

def get_torch_model_size(model: torch.nn.Module) -> int:
    # Thank you to ptrblck
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    return param_size + buffer_size


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

