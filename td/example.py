from collections import defaultdict
import tensorflow as tf
import numpy as np

from tensorBoard.tesorBoardPlot import TensorBoardPlot


def actions():
    return np.zeros(4)


Q = defaultdict(actions)

Q['a'][3] = 100

print(Q)

import pickle

q_learning_scope = 'q_learning'

with open(q_learning_scope + '.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(Q, f, pickle.HIGHEST_PROTOCOL)

with open(q_learning_scope + '.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    datas = pickle.load(f)

print(datas)
