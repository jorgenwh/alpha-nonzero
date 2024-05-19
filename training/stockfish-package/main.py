import time
from native_stockfish import get_stockfish

stockfish = get_stockfish()

stockfish.set_position("r1bq1b1r/p2k1ppp/n1p1pn2/3p4/3P4/4PN1P/PPP2PP1/RNBQK2R w KQ - 0 8")
stockfish.visualize()

stockfish.go()
time.sleep(5)
stockfish.stop()

v = stockfish.get_score()
print(v)
