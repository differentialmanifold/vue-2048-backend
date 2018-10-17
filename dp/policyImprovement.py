import sys
import numpy as np

if "../" not in sys.path:
    sys.path.append("../")
from boardWithoutTiles import BoardForTrain
from dp import policyEvaluate


def policy_improvement(transition_probability, policy_eval_fn=policyEvaluate.policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        transition_probability: transition_probability envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy
        V is the value function for the optimal policy.

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

    # Start with a random policy
    policy = policyEvaluate.create_random_policy(transition_probability)

    iterate = 0
    while True:
        iterate = iterate + 1
        print('iterate {}'.format(iterate))
        # Evaluate the current policy
        V = policy_eval_fn(policy, transition_probability, discount_factor)

        # Will be set to false if we make any changes to the policy
        policy_stable = True

        total_status = len(transition_probability)
        diff_status = 0

        # For each state...
        for s in transition_probability.keys():
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])

            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
                diff_status = diff_status + 1
            policy[s] = np.eye(4)[best_a]

        print('diff status is {}/{}'.format(diff_status, total_status))
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V


if __name__ == '__main__':
    board_without_tiles = BoardForTrain()

    transition_probability = board_without_tiles.createTransitionProbability()

    policy, V = policy_improvement(transition_probability, discount_factor=0.99)

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
