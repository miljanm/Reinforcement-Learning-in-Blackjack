from environment import State
import numpy as np


__author__ = 'miljan'

# action constants
HIT = 0
STICK = 1


def calculate_mse(mc_values, sarsa_values):
    """ Given the true value function and another value function calculates Mean Squared Error

    :param mc_values: Monte Carlo value function
    :param sarsa_values: SARSA value function
    :return: MSE between MC and SARSA value functions
    """
    mse = 0
    for key in mc_values:
        mse += (sarsa_values[key] - mc_values[key]) ** 2
    return mse / len(mc_values.keys())


def get_state_features(state):
    """ Transforms a given state into a feature representation based on the given state

    :param state: current state of the game
    :return: a numpy feature vector representation
    """
    player = state.player_sum
    dealer = state.dealer_first_card

    # features ranges
    player_intervals = [(1, 4), (4, 7), (7, 10)]
    dealer_intervals = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]

    # see which features the given state corresponds to
    features = []
    for i in range(len(player_intervals)):
        for j in range(len(dealer_intervals)):
            player_range = player_intervals[i]
            dealer_range = dealer_intervals[j]
            if (player_range[0] <= player <= player_range[1]) and (dealer_range[0] <= dealer <= dealer_range[1]):
                features.append(1)
            else:
                features.append(0)

    return np.array(features)


def get_state_action_features(state_features, action):
    """ Gives a 36 feature vector, where the order depends if the action is HIT or STICK

    :param state_features: feature representation of a current state
    :param action: chosen action
    :return: 36 feature vector numpy array
    """
    if action == HIT:
        return np.append(state_features, np.zeros(18))
    else:
        return np.append(np.zeros(18), state_features)