import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS

from anz.constants import DEVICE
from anz.mcts import MCTS
from anz.helpers import load_model


LOC_MODEL       = None
LOC_MODEL_TYPE  = None

LOC_MCTS        = None

LOC_FEN         = None
LOC_ROLLOUTS    = None


app = Flask(__name__)
CORS(app)


@app.route("/get", methods=["GET"])
def get():
    global LOC_MCTS
    global LOC_FEN
    global LOC_ROLLOUTS

    if LOC_MCTS is None or LOC_FEN is None or LOC_ROLLOUTS is None:
        return jsonify({"success": False})

    try:
        inference_result = LOC_MCTS.go(fen=LOC_FEN, rollouts=LOC_ROLLOUTS)
        move = inference_result.move
        value = inference_result.value
        top5 = inference_result.top5
        print(f"GET: move='{inference_result.move}' value='{inference_result.value}' top5='{inference_result.top5}'")

        return jsonify({
            "success": True, 
            "move": move, 
            "value": value,
            "top5": top5})
    except Exception as e:
        print(f"GET: failed with error: {e}")
        return jsonify({"success": False})


@app.route("/post", methods=["POST"])
def post():
    global LOC_FEN
    global LOC_ROLLOUTS

    try:
        data = request.get_json()

        fen = data["fen"]
        rollouts = int(data["rollouts"])

        LOC_FEN = fen
        LOC_ROLLOUTS = rollouts

        print(f"POST: fen='{fen}' rollouts='{rollouts}'")

        return jsonify({"success": True}) 
    except Exception as e:
        print(f"POST: failed with error: {e}")
        return jsonify({"success": False})


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-m", "-model", 
        type=str, 
        help="The path to the model parameters to load", 
        required=True
    )
    arg_parser.add_argument(
        "-mt", "-model-type",
        type=str,
        choices=["transformer", "resnet"],
        help="Model type: 'transformer' or 'resnet'",
        required=True
    )
    args = arg_parser.parse_args()

    LOC_MODEL       = load_model(args.m, args.mt).to(DEVICE)
    LOC_MODEL_TYPE  = args.mt
    LOC_MCTS        = MCTS(model=LOC_MODEL, model_type=LOC_MODEL_TYPE)

    app.run(debug=True)
