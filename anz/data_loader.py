import time
import os
import pickle
import chess
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union

from .constants import BATCH_SIZE
from .helpers import MODEL_TYPES, fen2vec, move2vec, allocate_zero_tensor

class AlphaZeroDataset(Dataset):
    def __init__(self, fens: list, moves: list, values: list, model_type: str):
        self.fens = fens
        self.moves = moves
        self.values = values
        self.model_type = model_type
        assert len(self.fens) == len(self.moves) == len(self.values)
        assert model_type in MODEL_TYPES, f"model_type must be one of {MODEL_TYPES}, not '{model_type}'"

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, index):
        position = fen2vec(self.fens[index], self.model_type)
        pi = move2vec(self.moves[index])
        v = allocate_zero_tensor((1), torch.float32)
        v[0] = self.values[index]
        return position, pi, v


def get_dataset(fn: str, model_type: str, max_datapoints: Union[int, None]) -> AlphaZeroDataset:
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
                    exit(1)
                size += 1
    else:
        size = max_datapoints

    fens = []
    moves = []
    values = []

    with open(fn, "rb") as in_fp:
        i = 1
        while 1:
            print(f"Reading datapoint {i:,}/{size:,}", end="\r", flush=True)
            try:
                fen, move, value = pickle.load(in_fp)
                fen = fen.strip()
                board = chess.Board(fen)
                if not board.turn:
                    fen = board.mirror().transform(chess.flip_horizontal).fen()
                    value = -value

                fens.append(fen)
                moves.append(move)
                values.append(value)
                i += 1

                if i >= size:
                    break

            except EOFError:
                break
            except Exception as e:
                print(f"Error while reading file '{fn}': {e}")
                exit(1)

        print(f"Reading datapoint {i:,}/{size:,}")

    return AlphaZeroDataset(fens, moves, values, model_type)

def get_data_loader(fn: str, model_type: str, max_datapoints: Union[int, None]) -> DataLoader:
    dataset = get_dataset(fn, model_type, max_datapoints)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return data_loader

