import chess

default_chess_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b Kkq - 0 1"

from anz.helpers import flip_fen

#print(flip_fen(default_chess_fen))

#board = chess.Board(default_chess_fen)
#print(board.fen())

#board.set_castling_fen("Qkq")
#print(board.fen())
#print()
print(default_chess_fen)
print(flip_fen(default_chess_fen))
