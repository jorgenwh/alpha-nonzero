from native_stockfish import Stockfish

FEN = "rnbqk1r1/pp3p1p/4pn1Q/2p5/3Pp3/P1P5/2P2PPP/R1B1KBNR w KQq - 0 1"
THINK_TIME = 1

stockfish = Stockfish()
stockfish.set_option("Threads", 1)
stockfish.search(THINK_TIME)


#stockfish.set_option("Threads", 12)
#stockfish.search(THINK_TIME)
