import tensorflow as tf
import numpy as np

import os


class TensorBoardPlot:
    def __init__(self, scope="estimator"):
        self.scope = scope
        summary_dir = os.path.join(os.path.expanduser('~/Developer/python/tensorboard'), 'summaries_{}'.format(scope))
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        self.summary_writer = tf.summary.FileWriter(summary_dir)

    def add_value(self, key, value, step):
        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=value, node_name=key, tag=key)
        self.summary_writer.add_summary(episode_summary, step)
        self.summary_writer.flush()
