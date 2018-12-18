import sys
import itertools
import os
import time
import argparse
import numpy as np
import keras
from collections import namedtuple
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, InputLayer, Flatten, Input, Lambda
from keras.optimizers import Adam

if "../" not in sys.path:
    sys.path.append("../")
from boardWithoutTiles import BoardForTrain
from tensorBoard.tesorBoardPlot import TensorBoardPlot
from model_save.model_save import ModelSave


class Estimator:
    """Q-Value Estimator neural network.
    """

    def __init__(self, model_size, scope):
        self.model_size = model_size
        self.model_save = ModelSave(scope=scope)
        self.model = None
        if self.model_save.exist():
            self.model = self.model_save.load()
        else:
            self.model = self._build_model()

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        main_input = Input(shape=(self.model_size, self.model_size), name='main_input')
        x = Flatten()(main_input)
        x = Dense(256, activation='relu', name='first_layer')(x)
        x = Dense(256, activation='relu', name='second_layer')(x)
        value_function_output = Dense(4, name='value_function_output')(x)

        action_input = Input(shape=(4,), name='action_input')

        def get_action_value(tensors):
            value_function_output_tensor = tensors[0]
            action_input_tensor = tensors[1]

            product_tensor = value_function_output_tensor * action_input_tensor

            return keras.backend.sum(product_tensor, axis=-1, keepdims=True)

        q_function_output = Lambda(get_action_value, name='q_function_output')([value_function_output, action_input])

        model = keras.Model(inputs=[main_input, action_input], outputs=q_function_output)

        model.summary()

        model.compile(loss='mean_squared_error',
                      optimizer=Adam())

        return model

    def save(self):
        self.model_save.save(self.model)

    def predict(self, s):
        """
        Predicts action values.
        Args:
          s: State input of shape [batch_size, 4, 4]
        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        main_input = self.model.get_layer('main_input').input
        value_function_output = self.model.get_layer('value_function_output').output

        get_value_function = keras.backend.function(inputs=[main_input], outputs=[value_function_output])

        return get_value_function([s])[0]

    def predict_single(self, s):
        """
        Predicts with single state.
        :param s: State input of shape [4, 4]
        :return: Tensor of shape [NUM_VALID_ACTIONS] containing the estimated action values.
        """
        return self.predict(np.expand_dims(s, axis=0))[0]

    def update(self, s, a, y):
        """
        Updates the estimator towards the given targets.
        Args:
          s: State input of shape [batch_size, 4, 4]
          a: Chosen actions of shape [batch_size, 4]
          y: Targets of shape [batch_size]
        Returns:
          The calculated loss on the batch.
        """
        loss = self.model.train_on_batch(x={'main_input': s, 'action_input': a}, y={'q_function_output': y})
        return loss

    def update_single(self, s, a, y):
        """
        Updates the estimator towards the given targets.
        :param s: State input of shape [4, 4]
        :param a: Chosen actions of shape (4,)
        :param y: Targets of shape ()
        :return: The calculated loss on the batch
        """
        s = np.expand_dims(s, axis=0)
        a = np.expand_dims(a, axis=0)
        y = np.expand_dims(y, axis=0)
        return self.update(s, a, y)


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        estimator: a estimator that predict state -> action-values.
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation, available_direction):
        dir_arr = np.array(available_direction).astype(int)

        A = (np.ones(nA, dtype=float) * dir_arr) * epsilon / np.sum(dir_arr)
        best_action = np.argmax(np.array(estimator.predict_single(observation)) * dir_arr)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env,
               args,
               replay_memory_size=50000,
               replay_memory_init_size=5000,
               batch_size=32,
               discount_factor=1.0,
               alpha=0.5,
               epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: environment.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    """
    q_learning_scope = args['scope'] + str(args['size'])

    estimator = Estimator(model_size=args['size'], scope=q_learning_scope)

    total_step = 0

    tensorBoardPlot = TensorBoardPlot(scope=q_learning_scope)

    # The policy we're following
    policy = make_epsilon_greedy_policy(estimator, epsilon, env.action_space)

    # The replay memory
    replay_memory = []
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    # Populate the replay memory with initial experience
    # print("Populating replay memory...")
    # matrix, can_move_dir = env.reset()
    # state = transferMatrixToState(matrix, env)
    # for i in range(replay_memory_init_size):
    #     action_probs = policy(state, can_move_dir)
    #     action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    #     next_matrix, reward, done, _, _, next_can_move_dir = env.step(action)
    #     next_state = transferMatrixToState(next_matrix, env)
    #     replay_memory.append(Transition(state, action, reward, next_state, done))
    #     if done:
    #         matrix, can_move_dir = env.reset()
    #         state = transferMatrixToState(matrix, env)
    #     else:
    #         state = next_state
    #         can_move_dir = next_can_move_dir
    # print("Init replay finished")

    for i_episode in itertools.count(start=total_step):
        # Reset the environment and pick the first action
        state, can_move_dir = env.reset()

        # One step in the environment
        for t in itertools.count():

            # Take a step
            action_probs = policy(state, can_move_dir)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            action_array = np.zeros(shape=(4,))
            action_array[action] = 1

            next_state, reward, done, _, _, next_can_move_dir = env.step(action)

            # # If our replay memory is full, pop the first element
            # if len(replay_memory) == replay_memory_size:
            #     replay_memory.pop(0)
            #
            # # Save transition to replay memory
            # replay_memory.append(Transition(state, action, reward, next_state, done))
            #
            # # TD Update
            # # Sample a minibatch from the replay memory
            # samples = random.sample(replay_memory, batch_size)
            # for sample in samples:
            #     best_next_action = np.argmax(Q[sample.next_state])
            #     td_target = sample.reward + discount_factor * Q[sample.next_state][best_next_action]
            #     td_delta = td_target - Q[sample.state][sample.action]
            #     Q[sample.state][sample.action] += alpha * td_delta

            next_q_value = 0.0
            if not done:
                next_value_function = estimator.predict_single(next_state)
                best_next_action = np.argmax(next_value_function)
                next_q_value = next_value_function[best_next_action]

            td_target = reward + discount_factor * next_q_value
            loss = estimator.update_single(state, action_array, td_target)

            # state_i = np.expand_dims(state, axis=0)
            # a_i = np.expand_dims(action_array, axis=0)
            # q_value = estimator.model.predict([state_i, a_i])
            # print(
            #     'in episode {} step {} loss is {} value function is {} q_value is {} action is {}'.format(i_episode, t,
            #                                                                                               loss,
            #                                                                                               estimator.predict_single(
            #                                                                                                   state),
            #                                                                                               q_value,
            #                                                                                               action_array))
            #
            # print('first layer weights is:')
            # weights_1 = estimator.model.get_layer('first_layer').get_weights()[0]
            # print(weights_1.shape)
            # print(np.sum(weights_1))
            # print(weights_1[0][:5])
            # print('second layer weights is:')
            # weights_2 = estimator.model.get_layer('second_layer').get_weights()[0]
            # print(weights_2.shape)
            # print(np.sum(weights_2))
            # print(weights_2[0][:5])
            # print('value function layer weights is:')
            # value_function_output_layer = estimator.model.get_layer('value_function_output')
            # weights_3 = value_function_output_layer.get_weights()[0]
            # print(weights_3.shape)
            # print(np.sum(weights_3))
            # print(weights_3[0][:5])

            state = next_state
            can_move_dir = next_can_move_dir

            if done:
                break

        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % int(args['outputInterval']) == 0:
            print('----------')
            print("Episode {}.".format(i_episode + 1))
            test_Q(tensorBoardPlot, env, estimator, i_episode)

        if (i_episode + 1) % (int(args['outputInterval']) * 10) == 0:
            print('start saving')
            current_time = time.time()
            estimator.save()
            print('save cost {} second'.format(time.time() - current_time))


def test_Q(tensorBoardPlot, env, estimator, step):
    board_without_tiles = BoardForTrain(size=int(args['size']))
    print('args size is {}'.format(args['size']))

    policy = make_epsilon_greedy_policy(estimator, 0, board_without_tiles.action_space)

    score_map = {}
    max_value_map = {}

    test_episodes = 100
    for i in range(test_episodes):
        board_without_tiles = BoardForTrain(size=int(args['size']))
        while not board_without_tiles.has_done():
            state = board_without_tiles.matrix()
            action = np.argmax(policy(state, board_without_tiles.can_move_dir))
            board_without_tiles.move(action)
        matrix = board_without_tiles.matrix()
        score = np.sum(matrix)
        max_value = np.max(matrix)
        if score in score_map:
            score_map[score] = score_map[score] + 1
        else:
            score_map[score] = 1

        if max_value in max_value_map:
            max_value_map[max_value] = max_value_map[max_value] + 1
        else:
            max_value_map[max_value] = 1
    print(score_map)
    print(max_value_map)

    weight_score = 0
    for key in score_map:
        weight_score = weight_score + key * (score_map[key] / test_episodes)

    weight_max_value = 0
    for key in max_value_map:
        weight_max_value = weight_max_value + key * (max_value_map[key] / test_episodes)

    tensorBoardPlot.add_value('weight_score', weight_score, step)
    tensorBoardPlot.add_value('weight_max_value', weight_max_value, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--size', help='size of matrix, 2x2,3x3,4x4', default=2)
    parser.add_argument('--outputInterval', help='interval to print test value', default=100)
    parser.add_argument('--scope', help='scope for saving in redis and tensorboard', default='2048_fa_primal')
    args = vars(parser.parse_args())

    board_without_tiles = BoardForTrain(size=int(args['size']))

    q_learning(board_without_tiles, args)
