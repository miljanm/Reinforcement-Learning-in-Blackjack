# do not remove, needed in add_subplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, pyplot as plt
import numpy as np
import itertools as it


__author__ = 'miljan'


# action constants
HIT = 0
STICK = 1


def plot_value_function(value_function):
    """ Plots the value function given

    :param value_function: value function to be plotted
    :return: a 3D plot of value function
    """
    # player range
    x = range(1, 22)
    # dealer range
    y = range(1, 11)
    # values matrix
    z = np.zeros((len(x), len(y)))
    for i, j in it.product(x, y):
        z[(i-1, j-1)] = max(value_function[(i, j, HIT)], value_function[(i, j, STICK)])
    # form a grid
    x, y = np.meshgrid(x, y)

    # do the plotting
    fig = plt.figure(figsize=(10.5, 7))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.plot_surface(x.T, y.T, z, rstride=1, cstride=1, cmap=cm.BrBG, antialiased=True)

    # set the plot parameters
    fig.colorbar(p)
    fig.suptitle('Monte Carlo control', fontsize=17)
    plt.xlabel("Player sum")
    plt.ylabel("Dealer showing")
    plt.xticks(range(1, 22, 2))
    plt.yticks(range(1, 11))
    ax.set_xlim(1, 22)
    ax.set_ylim(1, 11)
    ax.view_init(elev=20, azim=124)

    plt.show()


def plot_sarsa_mse(lambda_mse):
    """ Plots the Mean Squared Error for SARSA

    :param lambda_mse: MSE for each lambda used
    """
    x, y = zip(*lambda_mse)

    fig = plt.figure()
    plt.plot(x, y)

    fig.suptitle('Lambda vs Mean Squared Error', fontsize=17)
    plt.xlabel("lambda")
    plt.ylabel("MSE")

    plt.show()


def plot_learning_curve(episode_mse, l):
    """ Plots a learning curve for given episode, MSE pairs

    :param episode_mse: MSE for each episode
    :param l: lambda used for the given run
    """
    x, y = zip(*episode_mse)

    fig = plt.figure()
    plt.plot(x, y)

    fig.suptitle('Episode no. vs Mean Squared Error for lambda = ' + str(l), fontsize=17)
    plt.xlabel("Episode no.")
    plt.ylabel("MSE")

    plt.show()
