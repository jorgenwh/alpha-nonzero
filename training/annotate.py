import math
import chess
from tqdm import tqdm
from stockfish import Stockfish
from policy_index import policy_index


def move_to_policy_index(move: str) -> int:
    return policy_index.index(move)

def eval_to_annotation(eval: list()) -> list():
    annotation = [0] * len(policy_index)

    for d in eval:
        move = d["Move"]
        cp = d["Centipawn"]
        mate = d["Mate"]
        p_win = 0.5 * 2/(1 + math.exp(-0.004 * cp)) if mate is None else 1
        idx = move_to_policy_index(move)
        #print(f"Move: {move}, CP: {cp}, Mate: {mate}, P_win: {p_win}, Index: {idx}")
        annotation[idx] = p_win

    return annotation

def annotate_position(stockfish: Stockfish, fen: str) -> list():
    stockfish.set_fen_position(fen)
    legal_moves = [str(move) for move in chess.Board(fen).legal_moves]
    evals = stockfish.get_top_moves(len(legal_moves))
    annotation = eval_to_annotation(evals)
    return annotation


def annotate_positions(fens: list()) -> list():
    stockfish_path = "/home/jorgen/projects/alpha-nonzero/training/stockfish/stockfish-ubuntu-x86-64-avx2"
    stockfish_params = {
        "Threads": 4,
        #"Hash": 8192,   # MB of hash table
        #"UCI_LimitStrength": 'true',
        #"UCI_Elo": 2500,
    }
    stockfish = Stockfish(stockfish_path, parameters=stockfish_params)
    annotations = []
    bar = tqdm(fens)
    for fen in bar:
        annotation = annotate_position(stockfish, fen)
        annotations.append(annotation)
    return annotations


if __name__ == "__main__":
    f = open("data/fens.fen", "r")
    fens = []
    for fen in f:
        fens.append(fen)
        if len(fens) >= 100:
            break
    f.close()

    annotations = annotate_positions(fens)