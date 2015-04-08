import random
import utilities


__author__ = 'miljan'


# action constants
HIT = 0
STICK = 1


def epsilon_greedy(epsilon, value_function, state):
    """ Epsilon greedy policy, which returns random action with probability epsilon, highest value action otherwise.

    :param epsilon: random action probability
    :param value_function: (state, action) value function
    :param state: current state of the game
    :return: action to take
    """

    # exploration
    if random.random() < epsilon:
        return _random_action()
    # exploitation
    else:
        player = state.player_sum
        dealer = state.dealer_first_card

        value_HIT = value_function[(player, dealer, HIT)]
        value_STICK = value_function[(player, dealer, STICK)]

        if value_HIT > value_STICK:
            return HIT
        elif value_STICK > value_HIT:
            return STICK
        else:
            return _random_action()


def epsilon_greedy_lfa(epsilon, theta, state_features):
    """ Epsilon greedy policy, for linear function approximation,
    which returns random action with probability epsilon, highest value action otherwise.

    :param epsilon: random action probability
    :param value_function: [state_action_feature_vector] value function
    :param state: current state feature vector
    :return: action to take
    """

    # exploration
    if random.random() < epsilon:
        act = _random_action()
        val = utilities.get_state_action_features(state_features, act).dot(theta)
        return val, act
    # exploitation
    else:
        value_HIT = utilities.get_state_action_features(state_features, 0).dot(theta)
        value_STICK = utilities.get_state_action_features(state_features, 1).dot(theta)

        if value_HIT > value_STICK:
            return value_HIT, HIT
        elif value_STICK > value_HIT:
            return value_STICK, STICK
        else:
            act = _random_action()
            val = utilities.get_state_action_features(state_features, act).dot(theta)
            return val, act


def _random_action():
    return HIT if random.random() < 0.5 else STICK