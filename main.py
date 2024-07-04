import chess

fen = "rnb1kb1r/pp3ppp/2p1p3/6B1/8/5N2/PqP2PPP/RN1QKB1R w KQkq - 0 8"

fens = [fen]

from anz.helpers import fen_batch_to_vector_transformer, fen2vec

fens_vector = fen_batch_to_vector_transformer(fens)
print(fens_vector.dtype)
print(fens_vector.shape)

fens_vector2 = fen2vec(fen, "transformer")
fens_vector2.reshape(1, -1)
print(fens_vector2.dtype)
print(fens_vector2.shape)

print(fens_vector == fens_vector2)
print(fens_vector)
print(fens_vector2)
