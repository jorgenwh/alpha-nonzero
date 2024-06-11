
in_fp = open("fens/fens.fen", "r")
out_fp = open("fens/100mfens.fen", "w")

observed = set()
i = 0

for fen in in_fp:
    if fen not in observed:
        out_fp.write(fen)
        observed.add(fen)
        i += 1

        if i % 100 == 0:
            print(f"{i}/100000000 len(observed)={len(observed)}", end="\r", flush=True)

        if i == 100_000_000:
            break

print(f"100000000/100000000 len(observed)={len(observed)}")

in_fp.close()
out_fp.close()
