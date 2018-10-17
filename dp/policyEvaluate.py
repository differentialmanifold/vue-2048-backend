import sys
import numpy as np

if "../" not in sys.path:
    sys.path.append("../")
from boardWithoutTiles import BoardForTrain


def policy_eval(policy, transition_probability, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: representing the policy.
        transition_probability: transition_probability represents the transition probabilities of the environment.
            transition_probability[s][a] is a list of transition tuples (prob, next_state, reward, done).
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        the value function.
    """
    # Start with a random (all 0) value function
    V = {}
    for status in transition_probability.keys():
        V[status] = 0
    index = 0
    while True:
        index = index + 1
        print('index is {}'.format(index))
        # print(V)
        delta = 0
        # For each state, perform a "full backup"
        for s in transition_probability.keys():
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in transition_probability[s][a]:
                    # Calculate the expected value. Ref: Sutton book eq. 4.6.
                    v += action_prob * prob * (reward + discount_factor * V.get(next_state, 0))
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return V


def create_random_policy(transition_probability):
    random_policy = {}
    for s in transition_probability.keys():
        random_policy[s] = np.ones(shape=4) / 4
    return random_policy


if __name__ == '__main__':
    board_without_tiles = BoardForTrain()

    transition_probability = board_without_tiles.createTransitionProbability()

    random_policy = create_random_policy(transition_probability)

    vfun = policy_eval(random_policy, transition_probability)

    print(vfun)
