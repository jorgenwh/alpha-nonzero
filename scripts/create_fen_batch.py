import chess

in_fp = open("fens/fens.fen", "r")
out_fp = open("fens/75mfens.fen", "w")

observed = set()
i = 0
for fen in in_fp:
    if fen in observed:
        continue

    observed.add(fen)
    board = chess.Board(fen)
    if board.outcome() is None:
        out_fp.write(fen)
        i += 1

    if i % 1000:
        print(f"{i}/75000000")

    if i == 75000000:
        break

in_fp.close()
out_fp.close()
