# Reinforcement-Learning-in-Blackjack
Implementation of several reinforcement learning algorithms used to play a variation of blackjack


In order to run all the algorithms just run main.py.

This will execute test_all_algorithms() function which runs MC, SARSA and Linear Function Approximation for SARSA with plots showing the results.

Details about other modules:
- environment.py - contains the step() function and the implementation of the environment
- rl_algorithms - contains MC, SARSA and Linear Function Approximation 
- plotting.py - contains functions to plot value function, SARSA and LFA results
- policies.py - place to put the policies, at the moment contains just epsilon greedy policy
- utilities.py - calculation of mean squared error and conversion of state to feature vector for LFA
