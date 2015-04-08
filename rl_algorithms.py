from collections import defaultdict
from multiprocessing import Process
import environment
import policies
import plotting
import pickle
import utilities
import numpy as np


__author__ = 'miljan'


def monte_carlo(iterations=1000000, policy=policies.epsilon_greedy, n_zero=100):
    """ Performs Monte Carlo control in the Easy21 game.

    :param iterations: number of monte carlo iterations
    :param policy: exploration strategy to use
    :param n_zero: epsilon greedy constant (only applicable if epsilon greedy policy is used)
    :return: value function and the plot of the optimal value function
    """
    # (player, dealer, action) key
    value_function = defaultdict(float)
    # (player, dealer) key
    counter_state = defaultdict(int)
    # (player, dealer, action) key
    counter_state_action = defaultdict(int)
    # number of wins
    wins = 0

    print('Iterations completed:')
    for i in xrange(iterations):

        if (i % 500000) == 0:
            print(i)

        # create a new random starting state
        state = environment.State()
        # play a round
        observed_keys = []
        while not state.terminal:
            player = state.player_sum
            dealer = state.dealer_first_card

            # find an action defined by the policy
            epsilon = n_zero / float(n_zero + counter_state[(player, dealer)])
            action = policy(epsilon, value_function, state)
            observed_keys.append((player, dealer, action))

            # take a step
            [state, reward] = environment.step(state, action)

        # we have reached an end of episode
        if reward is not None:
            # update over all keys
            for key in observed_keys:
                # update counts
                counter_state[key[:-1]] += 1
                counter_state_action[key] += 1

                # update value function
                alpha = 1.0 / counter_state_action[key]
                value_function[key] += alpha * (reward - value_function[key])

        if reward == 1:
            wins += 1

    print('Wins: %.4f%%' % ((float(wins) / iterations) * 100))
    # plot the optimal value function
    plotting.plot_value_function(value_function)
    return value_function


def sarsa_lambda(l=0.9, max_episodes=1000, policy=policies.epsilon_greedy,
                 n_zero=100, gamma=1, plot_learning_curve=True, multiproc=True):
    """ Applies eligibility trace version of Sarsa to the game Easy21

    :param l: lambda parameter
    :param max_episodes: stop learning after this many episodes
    :param policy: exploration strategy to use
    :param n_zero: epsilon greedy constant (only applicable if epsilon greedy policy is used)
    :param gamma: discounting rate
    :param plot_learning_curve: whether to turn on plotting of learning curve for lambda = 0 and 1
    :param multiproc: whether to use multiprocessing when doing plots or not (warning! turn off if running multiple
        algorithms on mac or windows simultaneously)
    :return: value function after max_episodes
    """
    # (player, dealer, action) key
    value_function = defaultdict(float)
    # (player, dealer) key
    counter_state = defaultdict(int)
    # (player, dealer, action) key
    counter_state_action = defaultdict(int)
    # no. of wins to calculate the percentage of wins at the end
    wins = 0

    # learning curve plotting
    if l in {0, 1} and plot_learning_curve:
        learning_curve = []
        try:
            mc_values = pickle.load(open("Data/MC_value_function.pickle", "rb"))
        except:
            mc_values = monte_carlo(iterations=1000000)

    for episode in range(max_episodes):

        # current (player, dealer, action)
        eligibility_trace = defaultdict(float)

        # initial state, action [SA..]
        state = environment.State()
        player_current = state.player_sum
        dealer_current = state.dealer_first_card
        epsilon = n_zero / float(n_zero + counter_state[(player_current, dealer_current)])
        action_current = policy(epsilon, value_function, state)

        while not state.terminal:

            # update counts
            counter_state[(player_current, dealer_current)] += 1
            counter_state_action[(player_current, dealer_current, action_current)] += 1

            # take a step, get reward [..R..]
            [state, reward] = environment.step(state, action_current)
            if reward is None:
                reward = 0

            # follow up state, action [..SA]
            player_next = state.player_sum
            dealer_next = state.dealer_first_card
            epsilon = n_zero / float(n_zero + counter_state[(player_next, dealer_next)])
            action_next = policy(epsilon, value_function, state)

            delta = reward + gamma * value_function[(player_next, dealer_next, action_next)] - \
                value_function[(player_current, dealer_current, action_current)]

            alpha = 1.0 / counter_state_action[(player_current, dealer_current, action_current)]

            eligibility_trace[(player_current, dealer_current, action_current)] += 1

            # update the values
            for key in value_function:
                value_function[key] += alpha * delta * eligibility_trace[key]
                eligibility_trace[key] *= gamma * l

            player_current = player_next
            dealer_current = dealer_next
            action_current = action_next

        # use it later to calculate the percentage of wins
        if reward == 1:
            wins += 1

        # get the episode MSE for plotting learning curve
        if l in {0, 1} and plot_learning_curve:
            learning_curve.append((episode, utilities.calculate_mse(mc_values, value_function)))

    # plot learning curve
    if l in {0, 1} and plot_learning_curve:
        if multiproc:
            # create a new process so computation can continue after plotting
            p = Process(target=plotting.plot_learning_curve, args=(learning_curve, l,))
            p.start()
        else:
            plotting.plot_learning_curve(learning_curve, l)

    # get the percentage of wins
    print float(wins) / max_episodes
    return value_function


def linear_function_approximation(l=0.9, max_episodes=1000, policy=policies.epsilon_greedy_lfa, n_zero=100,
                                  gamma=1, plot_learning_curve=True, multiproc=True):
    """ Value function approximation using coarse coding

    :param l: lambda parameter
    :param gamma: discounting rate
    :param max_episodes: stop learning after this many episodes
    :param policy: exploration strategy to use
    :param n_zero: epsilon greedy constant (only applicable if epsilon greedy policy is used)
    :param multiproc: whether to use multiprocessing when doing plots or not (warning! turn off if running multiple
        algorithms on mac or windows simultaneously)
    :return: value function after max_episodes
    """
    # weights vector for the state_action feature vector
    theta = np.random.random(36)*0.2
    # random move probability
    epsilon = 0.05
    # step-size parameter
    alpha = 0.01

    # learning curve plotting
    if l in {0, 1} and plot_learning_curve:
        learning_curve = []
        try:
            mc_values = pickle.load(open("Data/MC_value_function.pickle", "rb"))
        except:
            mc_values = monte_carlo(iterations=1000000)

    for episode in range(max_episodes):

        # key is state_action feature vector
        eligibility_trace = np.zeros(36)

        # initial state, action [SA..], and set of features
        state = environment.State()
        # calculate features for the given state
        state_features_current = utilities.get_state_features(state)
        # get action from this state
        q_a_current, action_current = policy(epsilon, theta, state_features_current)
        # calculate final state, action feature vector
        features_current = utilities.get_state_action_features(state_features_current, action_current)

        while not state.terminal:

            # update eligibility trace (accumulating)
            eligibility_trace = np.add(eligibility_trace, features_current)

            # take a step, get reward [..R..]
            [state, reward] = environment.step(state, action_current)
            if reward is None:
                reward = 0

            # follow up state, action [..SA]
            state_features_next = utilities.get_state_features(state)
            q_a_next, action_next = policy(epsilon, theta, state_features_next)
            features_next = utilities.get_state_action_features(state_features_next, action_next)

            # calculate state value difference
            delta = reward + gamma * q_a_next - q_a_current
            # update weights
            theta = np.add(theta, alpha * delta * eligibility_trace)
            # update trace
            eligibility_trace *= gamma * l

            features_current = features_next
            action_current = action_next

        # calculate value function
        value_function = defaultdict(float)
        for player in xrange(1, 22):
            for dealer in xrange(1, 11):
                for action in [0, 1]:
                    s = environment.State(dealer, player)
                    phi = utilities.get_state_action_features(utilities.get_state_features(s), action)
                    value_function[(s.player_sum, s.dealer_first_card, action)] = phi.dot(theta)

        # get the episode MSE for plotting learning curve
        if l in {0, 1} and plot_learning_curve:
            learning_curve.append((episode, utilities.calculate_mse(mc_values, value_function)))

    # plot learning curves
    if l in {0, 1} and plot_learning_curve:
        if multiproc:
            # create a new process so computation can continue after plotting
            p = Process(target=plotting.plot_learning_curve, args=(learning_curve, l,))
            p.start()
        else:
            plotting.plot_learning_curve(learning_curve, l)

    return value_function
