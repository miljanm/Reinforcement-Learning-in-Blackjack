import random


__author__ = 'miljan'


# game action constants
HIT = 0
STICK = 1


def _draw_black_card():
    """ Draw a black card from a deck

    :return: a positive card in range [1, 10]
    """
    return random.randint(1, 10)


def _draw_red_card():
    """ Draw a red card from a deck

    :return: a negative card in range [1, 10]
    """
    return -random.randint(1, 10)


def _draw_card():
    """ Draws a card from the deck.

    :return: a card in range [1, 10], red with probability 1/3, otherwise black
    """
    return _draw_red_card() if random.random() < (1 / 3.0) else _draw_black_card()


def step(state, action):
    """ Returns the next game state and reward, given current state and action.

    :param state: current state of the game, instance of State class
    :param action: 0 or 1, corresponding to hit or stick, respectively
    :return: a tuple (next game state, reward)
    """

    # PLAYER TURN
    if action == HIT:
        state.player_sum += _draw_card()
        # if player is bust terminate game
        if state.player_sum > 21 or state.player_sum < 1:
            state.terminal = True
            return state, -1
        else:
            return state, None
    # DEALER TURN
    elif action == STICK:
        # this is the last action to be played
        state.terminal = True
        # dealer sticks on 17 or greater
        while state.dealer_first_card < 17:
            state.dealer_first_card += _draw_card()
            # check if the sum went negative
            if state.dealer_first_card < 1:
                return state, 1
        # find a winner
        if state.dealer_first_card > 21 or state.dealer_first_card < state.player_sum:
            return state, 1
        elif state.dealer_first_card > state.player_sum:
            return state, -1
        else:
            return state, 0
    else:
        raise Exception('Unknown game action!')


class State(object):
    def __init__(self, dealer_first_card=None, player_sum=None):
        # boolean flag noting if the game is finished
        self.terminal = False
        if dealer_first_card is not None:
            self.dealer_first_card = dealer_first_card
        else:
            self.dealer_first_card = _draw_black_card()
        if player_sum is not None:
            self.player_sum = player_sum
        else:
            self.player_sum = _draw_black_card()