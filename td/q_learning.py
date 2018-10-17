import sys
import itertools
import os
import pickle
import argparse
import numpy as np
from collections import defaultdict

if "../" not in sys.path:
    sys.path.append("../")
from boardWithoutTiles import BoardForTrain
from tensorBoard.tesorBoardPlot import TensorBoardPlot

q_learning_scope = 'q_learning'


def actions():
    return np.zeros(4)


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation, available_direction, is_print=False):
        dir_arr = np.array(available_direction).astype(int)

        if np.sum(np.array(Q[observation])) == 0:
            if is_print:
                print("observation {} is not trained".format(observation))
            return dir_arr / np.sum(dir_arr)

        A = (np.ones(nA, dtype=float) * dir_arr) * epsilon / np.sum(dir_arr)
        best_action = np.argmax(np.array(Q[observation]) * dir_arr)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, args, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: environment.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        Q is the optimal action-value function, a dictionary mapping state -> action values.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    # if os.path.isfile(q_learning_scope + '.pickle'):
    #     with open(q_learning_scope + '.pickle', 'rb') as f:
    #         Q = pickle.load(f)
    # else:
    #     Q = defaultdict(actions)
    Q = defaultdict(actions)

    tensorBoardPlot = TensorBoardPlot(scope=q_learning_scope, summaries_dir=os.path.dirname(os.path.abspath(__file__)))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space)

    for i_episode in itertools.count():
        # Reset the environment and pick the first action
        matrix, can_move_dir = env.reset()
        state = env.transferMatrixToTuple(matrix)

        # One step in the environment
        episode_length = 0
        for t in itertools.count():

            # Take a step
            action_probs = policy(state, can_move_dir)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_matrix, reward, done, _, _, next_can_move_dir = env.step(action)
            next_state = env.transferMatrixToTuple(next_matrix)

            episode_length = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            state = next_state
            can_move_dir = next_can_move_dir

            if done:
                break

        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % int(args['outputInterval']) == 0:
            print('----------')
            print("Episode {}.".format(i_episode + 1))
            test_Q(tensorBoardPlot, Q, i_episode)

            # with open(q_learning_scope + '.pickle', 'wb') as f:
            #     pickle.dump(Q, f, pickle.HIGHEST_PROTOCOL)

    return Q


def test_Q(tensorBoardPlot, Q, step):
    board_without_tiles = BoardForTrain(size=int(args['size']))

    policy = make_epsilon_greedy_policy(Q, 0, board_without_tiles.action_space)

    score_map = {}
    max_value_map = {}

    test_episodes = 1000
    for i in range(test_episodes):
        board_without_tiles = BoardForTrain(size=int(args['size']))
        while not board_without_tiles.has_done():
            status = board_without_tiles.transferMatrixToTuple(board_without_tiles.matrix())
            action = np.argmax(policy(status, board_without_tiles.can_move_dir))
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

    q_length = len(Q)
    print("Q length is {}".format(q_length))
    tensorBoardPlot.add_value('Q_length', q_length, step)
    tensorBoardPlot.add_value('weight_score', weight_score, step)
    tensorBoardPlot.add_value('weight_max_value', weight_max_value, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--size', help='size of matrix, 2x2,3x3,4x4', default=2)
    parser.add_argument('--outputInterval', help='interval to print test value', default=100)
    args = vars(parser.parse_args())

    board_without_tiles = BoardForTrain(size=int(args['size']))

    Q = q_learning(board_without_tiles, args)
