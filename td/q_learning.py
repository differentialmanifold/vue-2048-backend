import sys
import itertools
import os
import time
import argparse
import numpy as np
from collections import namedtuple

if "../" not in sys.path:
    sys.path.append("../")
from boardWithoutTiles import BoardForTrain
from tensorBoard.tensorBoardPlot import TensorBoardPlot
from my_redis.my_redis import MyRedis


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


def transferMatrixToState(matrix, env, Q):
    state = env.transferMatrixToTuple(matrix)
    if state not in Q:
        Q[state] = np.zeros(4)
    return state


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

    Returns:
        Q is the optimal action-value function, a dictionary mapping state -> action values.
    """
    q_learning_scope = args['scope']

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    my_redis = MyRedis(q_learning_scope)
    saved_obj = my_redis.restore_td()

    if saved_obj is None:
        Q = dict()
        total_step = 0
    else:
        Q = saved_obj['q']
        total_step = saved_obj['step'] + 1

    tensorBoardPlot = TensorBoardPlot(scope=q_learning_scope, summaries_dir=args['summaries_dir'])

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space)

    # The replay memory
    replay_memory = []
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    # Populate the replay memory with initial experience
    # print("Populating replay memory...")
    # matrix, can_move_dir = env.reset()
    # state = transferMatrixToState(matrix, env, Q)
    # for i in range(replay_memory_init_size):
    #     action_probs = policy(state, can_move_dir)
    #     action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    #     next_matrix, reward, done, _, _, next_can_move_dir = env.step(action)
    #     next_state = transferMatrixToState(next_matrix, env, Q)
    #     replay_memory.append(Transition(state, action, reward, next_state, done))
    #     if done:
    #         matrix, can_move_dir = env.reset()
    #         state = transferMatrixToState(matrix, env, Q)
    #     else:
    #         state = next_state
    #         can_move_dir = next_can_move_dir
    # print("Init replay finished")

    # added part saved to redis
    add_saved_obj = dict()

    for i_episode in itertools.count(start=total_step):
        # Reset the environment and pick the first action
        matrix, can_move_dir = env.reset()
        state = transferMatrixToState(matrix, env, Q)

        # One step in the environment
        episode_length = 0
        for t in itertools.count():

            # Take a step
            action_probs = policy(state, can_move_dir)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_matrix, reward, done, _, _, next_can_move_dir = env.step(action)
            next_state = transferMatrixToState(next_matrix, env, Q)

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

            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            add_saved_obj[state] = Q[state]

            state = next_state
            can_move_dir = next_can_move_dir

            if done:
                break

        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % int(args['outputInterval']) == 0:
            print('----------')
            print("Episode {}.".format(i_episode + 1))
            test_Q(tensorBoardPlot, env, Q, i_episode)

        if (i_episode + 1) % (int(args['outputInterval']) * 10) == 0:
            print('start saving with {} state'.format(len(add_saved_obj)))
            current_time = time.time()
            saved_obj = {'q': add_saved_obj, 'step': i_episode}
            my_redis.store_td(saved_obj)
            add_saved_obj = dict()
            print('save cost {} second'.format(time.time() - current_time))

    return Q


def test_Q(tensorBoardPlot, env, Q, step):
    board_without_tiles = BoardForTrain(size=int(args['size']))

    policy = make_epsilon_greedy_policy(Q, 0, board_without_tiles.action_space)

    score_map = {}
    max_value_map = {}

    test_episodes = 1000
    for i in range(test_episodes):
        board_without_tiles = BoardForTrain(size=int(args['size']))
        while not board_without_tiles.has_done():
            status = transferMatrixToState(board_without_tiles.matrix(), env, Q)
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
    parser.add_argument('--scope', help='scope for saving in redis and tensorboard', default='tencent2')
    parser.add_argument('--summaries_dir', help='directory for tensorboard',
                        default=os.path.expanduser('~/tensorboard'))
    args = vars(parser.parse_args())

    board_without_tiles = BoardForTrain(size=int(args['size']))

    Q = q_learning(board_without_tiles, args)
