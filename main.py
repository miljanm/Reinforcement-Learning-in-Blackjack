import rl_algorithms
import numpy as np
import pickle
import plotting
import utilities


__author__ = 'miljan'


def _test_monte_carlo(iterations=100000, pickle_file=False):
    """ Test the monte carlo control algorithm

    :param iterations: Number of monte carlo iterations
    :param pickle_file: whether to save the results or not for use in other algorithms
    """
    print '\nTesting Monte-Carlo control...'

    mc_value_function = rl_algorithms.monte_carlo(iterations)
    if pickle_file:
        print 'Pickling...'
        pickle.dump(mc_value_function, open("Data/MC_value_function.pickle", "wb"))
        print 'Done pickling...'


def _test_sarsa(averaging_runs=1, plot_learning_curve=False, multiproc=True):
    """ Test the SARSA algorithm

    :param averaging_runs: Number of runs to use in averaging SARSA results
    :param plot_learning_curve: Whether to plot the learning curve for lambda being 0 or 1
    """
    print '\nTesting SARSA...'

    try:
        mc_value_function = pickle.load(open("Data/MC_value_function.pickle", "rb"))
    except:
        mc_value_function = rl_algorithms.monte_carlo(iterations=1000000)
    average_mse = []
    lambda_space = np.linspace(0, 1, 11)

    # get MSE errors over all averaging runs
    for i in xrange(averaging_runs):
        mse = []
        for _lambda in lambda_space:
            sarsa_value_function = rl_algorithms.sarsa_lambda(l=_lambda, \
                                        plot_learning_curve=plot_learning_curve, multiproc=multiproc)
            mse.append(utilities.calculate_mse(mc_value_function, sarsa_value_function))
        average_mse.append(mse)
        if (i % 5) == 0:
            print i

    # average scores from n runs of SARSA and plot
    average_mse = [float(sum(col))/len(col) for col in zip(*average_mse)]
    average_lambda_mse = zip(lambda_space, average_mse)

    for entry in average_lambda_mse:
        print('--------------------')
        print('Lambda: %.1f' % entry[0])
        print('Mean Squared Error: %.4f' % entry[1])
        print('\n')

    plotting.plot_sarsa_mse(average_lambda_mse)


def _test_linear_function_approximation(averaging_runs=1, plot_learning_curve=False, multiproc=True):
    """ Test the SARSA LFA algorithm

    :param averaging_runs: Number of runs to use in averaging SARSA LFA results
    :param plot_learning_curve: Whether to plot the learning curve for lambda being 0 or 1
    """
    print '\nTesting linear function approximation...'

    mc_value_function = pickle.load(open("Data/MC_value_function.pickle", "rb"))
    average_mse = []
    lambda_space = np.linspace(0, 1, 11)

    # get MSE errors over all averaging runs
    for i in xrange(averaging_runs):
        mse = []
        for _lambda in lambda_space:
            sarsa_value_function = rl_algorithms.linear_function_approximation(l=_lambda, \
                                        plot_learning_curve=plot_learning_curve, multiproc=multiproc)
            mse.append(utilities.calculate_mse(mc_value_function, sarsa_value_function))
        average_mse.append(mse)
        if (i % 5) == 0:
            print i

    # average scores from n runs of SARSA and plot
    average_mse = [float(sum(col))/len(col) for col in zip(*average_mse)]
    average_lambda_mse = zip(lambda_space, average_mse)

    for entry in average_lambda_mse:
        print('--------------------')
        print('Lambda: %.1f' % entry[0])
        print('Mean Squared Error: %.4f' % entry[1])
        print('\n')

    plotting.plot_sarsa_mse(average_lambda_mse)


# run tests for all algorithms
def _test_all_algorithms():
    """ Runs all the algorithms implemented in rl_algorithms module
    """
    _test_monte_carlo(iterations=1000000)
    _test_sarsa(averaging_runs=1, plot_learning_curve=True, multiproc=False)
    _test_linear_function_approximation(averaging_runs=1, plot_learning_curve=True, multiproc=False)


if __name__ == '__main__':
    # _test_linear_function_approximation(averaging_runs=1, plot_learning_curve=True, multiproc=True)
    # _test_monte_carlo(iterations=5000000, pickle_file=True)
    # _test_sarsa(averaging_runs=1, plot_learning_curve=True)
    _test_all_algorithms()