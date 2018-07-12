from flask import Flask, json, request, Response
from flask_cors import CORS
from board import Board

app = Flask(__name__)
CORS(app)

board = Board()


@app.route('/init')
def init():
    board.__init__()
    return transfer_response(board.front_call_obj())


@app.route('/step/<int:direct>')
def step(direct):
    board.move(direct)
    return transfer_response(board.front_call_obj())


def transfer_response(obj, status_code=200):
    predict_results = json.dumps(obj, ensure_ascii=False)
    return Response(
        response=predict_results,
        mimetype="application/json; charset=UTF-8",
        status=status_code
    )


if __name__ == '__main__':
    app.run()
