import os
from keras.models import load_model


class ModelSave:
    def __init__(self, scope="estimator"):
        self.scope = scope

        model_dir = os.path.expanduser('~/Developer/python/model_save/model_{}'.format(scope))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model_path = os.path.join(model_dir, 'model_{}.h5'.format(scope))

    def exist(self):
        return os.path.isfile(self.model_path)

    def save(self, model):
        model.save(self.model_path)

    def load(self):
        return load_model(self.model_path)
