import os

NUM_BATCHES = 5000
BATCH_SIZE = 1000

if not os.path.isdir("training_data"):
    os.mkdir("training_data")

f = open("data/fens.fen", "r")
batch = 1

fen_batch = []
i = 1
for fen in f:
    print(f"batch: {batch}  ", end="\r")
    fen_batch.append(fen)
    if len(fen_batch) == BATCH_SIZE:
        out_f = open(f"training_data/fen_batch_{batch}.fen", "w")
        for out_fen in fen_batch:
            out_f.write(out_fen)
        out_f.close()

        fen_batch.clear()
        i = 1
        batch += 1

    i += 1
    if batch >= NUM_BATCHES:
        break

print(f"batch: {batch}  ")

f.close()
