import time
import chess

def pgn_line_is_game(line: str) -> bool:
    return line[0].isnumeric()

def parse_pgn_game(line: str) -> list:
    moves = []
    data = line.split(".")[1:]

    for ply in data:
        ply = ply.strip()
        ply = ply.split(" ")
        if len(ply) == 2:
            moves.append(ply[0])
        elif len(ply) == 3:
            moves.append(ply[0])
            moves.append(ply[1])

    return moves

def extract_fen_history(moves: list) -> list:
    board = chess.Board()
    fens = [board.fen()]

    for move in moves:
        board.push_san(move)
        fens.append(board.fen())

    return fens

def parse_pgn(input_filename: str, output_filename: str) -> None:
    in_file = open(input_filename, "r")
    out_file = open(output_filename, "w")

    games_parsed = 0
    t0 = time.time()
    for line in in_file:
        if "{" in line:
            continue

        if pgn_line_is_game(line):
            moves = parse_pgn_game(line)
            fens = extract_fen_history(moves)
            for fen in fens:
                out_file.write(fen + "\n")
            games_parsed += 1

            elapsed_s = time.time() - t0
            games_per_second = games_parsed / elapsed_s

            if games_parsed % 100 == 0:
                print(f"Games parsed: {games_parsed} [{int(games_per_second)}/s]", end="\r")
    print(f"Games parsed: {games_parsed} [{int(games_per_second)}/s]")

    in_file.close()
    out_file.close()


