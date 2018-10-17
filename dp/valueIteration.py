import sys
import numpy as np

if "../" not in sys.path:
    sys.path.append("../")
from boardWithoutTiles import BoardForTrain
from dp import policyEvaluate


def value_iteration(transition_probability, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        transition_probability: transition_probability envrionment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider
            V: The value to use as an estimator

        Returns:
            A vector containing the expected value of each action.
        """
        A = np.zeros(4)
        for a in range(4):
            for prob, next_state, reward, done in transition_probability[state][a]:
                A[a] += prob * (reward + discount_factor * V.get(next_state, 0))
        return A

    V = {}
    for status in transition_probability.keys():
        V[status] = 0

    index = 0
    while True:
        index = index + 1
        print('index is {}'.format(index))
        # Stopping condition
        delta = 0
        # Update each state...
        for s in transition_probability.keys():
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            V[s] = best_action_value
            # Check if we can stop
        if delta < theta:
            break

    # Start with a random policy
    policy = policyEvaluate.create_random_policy(transition_probability)
    for s in transition_probability.keys():
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s][best_action] = 1.0

    return policy, V


if __name__ == '__main__':
    board_without_tiles = BoardForTrain()

    transition_probability = board_without_tiles.createTransitionProbability()

    policy, V = value_iteration(transition_probability, discount_factor=0.99)

    print('policy is:')
    print(policy)
    print('value function is:')
    print(V)

    print(board_without_tiles.matrix())
    while not board_without_tiles.has_done():
        status = board_without_tiles.transferMatrixToTuple(board_without_tiles.matrix())
        action = np.argmax(policy[status])
        board_without_tiles.move(action)
        print('---move action {} ---'.format(action))
        print(board_without_tiles.matrix())

    score_map = {}

    for i in range(1000):
        board_without_tiles = BoardForTrain()
        while not board_without_tiles.has_done():
            status = board_without_tiles.transferMatrixToTuple(board_without_tiles.matrix())
            action = np.argmax(policy[status])
            board_without_tiles.move(action)
            print('---move action {} ---'.format(action))
            print(board_without_tiles.matrix())
        score = board_without_tiles.transferMatrixToTuple(board_without_tiles.matrix())
        score = tuple(sorted(list(score), reverse=True))
        if score in score_map:
            score_map[score] = score_map[score] + 1
        else:
            score_map[score] = 1
    print(score_map)
