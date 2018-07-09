from flask import Flask, json, request, Response

app = Flask(__name__)


@app.route('/')
def hello_world():
    board = Board()

    # return 'Hello World!'
    return transfer_response(board.__dict__)


def transfer_response(obj, status_code=200):
    predict_results = json.dumps(obj, ensure_ascii=False)
    return Response(
        response=predict_results,
        mimetype="application/json; charset=UTF-8",
        status=status_code
    )


class Board:
    def __init__(self):
        self.a = 'a'
        self.b = 'b'

    def create(self):
        print('create')


if __name__ == '__main__':
    app.run()
