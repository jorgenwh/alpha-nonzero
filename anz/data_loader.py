import os
import pickle
import chess
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union, List

from .constants import BATCH_SIZE, POLICY_INDEX
from .helpers import MODEL_TYPES, fens2vec, allocate_zero_tensor, flip_chess_move


class AlphaZeroDataset(Dataset):
    def __init__(self, 
                 fens: List[str],
                 moves: torch.Tensor, 
                 values: torch.Tensor, 
                 model_type: str, 
                 use_fp16: bool = False
    ):
        self.fens = fens
        self.moves = moves
        self.values = values
        self.model_type = model_type
        self.use_fp16 = use_fp16

        assert len(self.fens) == len(self.moves) == len(self.values)
        assert model_type in MODEL_TYPES, f"model_type must be one of {MODEL_TYPES}, not '{model_type}'"
        if use_fp16:
            assert self.values.dtype == torch.bfloat16, f"Dataset dtype does not match value dtype: {self.values.dtype}"

    def __len__(self):
        return len(self.fens) // BATCH_SIZE

    def __getitem__(self, index):
        i0 = index * BATCH_SIZE
        i1 = i0 + BATCH_SIZE
    
        positions = fens2vec(self.fens[i0:i1], self.model_type, self.use_fp16)
        pis = self.moves[i0:i1]
        vs = self.values[i0:i1]

        return positions, pis, vs


def get_dataset(
        fn: str, 
        model_type: str, 
        max_datapoints: Union[int, None], 
        use_fp16: bool = False
) -> AlphaZeroDataset:
    assert os.path.isfile(fn), f"File not found: {fn}"

    size = 0
    if max_datapoints is None:
        with open(fn, "rb") as in_fp:
            while 1:
                try:
                    _ = pickle.load(in_fp)
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error while reading file '{fn}': {e}")
                    #exit(1)
                    continue
                size += 1
    else:
        size = max_datapoints

    fens = []
    moves = allocate_zero_tensor(size, torch.int64)
    values = allocate_zero_tensor(size, torch.bfloat16 if use_fp16 else torch.float32)

    with open(fn, "rb") as in_fp:
        i = 0
        while 1:
            if i % 1000 == 0:
                print(f"Reading datapoint {i:,}/{size:,}", end="\r", flush=True)

            try:
                fen, move, value = pickle.load(in_fp)
            except EOFError:
                break
            except Exception as e:
                print(f"Error while reading file '{fn}': {e}")
                #exit(1)
                continue

            fen = fen.strip()
            board = chess.Board(fen)
            if not board.turn:
                fen = board.mirror().transform(chess.flip_horizontal).fen()
                move = flip_chess_move(move)
                value = -value

            fens.append(fen)
            moves[i] = POLICY_INDEX.index(move)
            values[i] = value
            i += 1

            if i >= size:
                break

        print(f"Reading datapoint {i:,}/{size:,}")

    return AlphaZeroDataset(fens, moves, values, model_type)

def get_data_loader(fn: str, model_type: str, max_datapoints: Union[int, None], use_fp16: bool = False) -> DataLoader:
    dataset = get_dataset(fn, model_type, max_datapoints, use_fp16)
    data_loader = DataLoader(dataset)
    return data_loader

