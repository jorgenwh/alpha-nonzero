
import pickle

with open("data/training_data.pkl", "rb") as f:
    while True:
        inp = input("~")
        if inp == "q":
            break
        try:
            print(pickle.load(f))
        except EOFError:
            break
