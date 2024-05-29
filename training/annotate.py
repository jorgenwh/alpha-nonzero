import math
import chess
import chess.engine
from tqdm import tqdm
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

def annotate_position(engine, fen):
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(time=0.5))
    print(info)

    #annotation = eval_to_annotation(info["score"])
    annotation = None
    return annotation

def annotate_positions(fens):
    stockfish_path = "/home/jorgen/projects/alpha-nonzero/training/stockfish/stockfish-ubuntu-x86-64-avx2"
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    print(engine.options)

    exit()
    annotations = []
    bar = tqdm(fens)
    for fen in bar:
        annotation = annotate_position(engine, fen)
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
