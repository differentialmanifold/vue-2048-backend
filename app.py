from flask import Flask, json, request, Response
from flask_cors import CORS
from board import Board
from mcts.MonteCarloTreeSearch import MonteCarloTreeSearch

app = Flask(__name__)
CORS(app)

board = Board()
monte_carlo_tree_search = MonteCarloTreeSearch()


@app.route('/init')
def init():
    board.env_init()
    return transfer_response(board.front_call_obj())


@app.route('/step/<int:direct>')
def step(direct):
    board.step(direct)
    return transfer_response(board.front_call_obj())


@app.route('/autostep')
def autostep():
    direct = monte_carlo_tree_search.find_next_move(board.matrix())
    board.step(direct)
    return transfer_response(board.front_call_obj())


def transfer_response(obj, status_code=200):
    predict_results = json.dumps(obj, ensure_ascii=False)
    return Response(
        response=predict_results,
        mimetype="application/json; charset=UTF-8",
        status=status_code
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
