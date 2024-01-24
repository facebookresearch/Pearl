Pearl: A production ready Reinforcement learning library

This document gives an overview of Pearl file structure and organization. Reading this will be helpful for developers and end users to get an idea of how to use different components of pearl for testing different algorithmic innovations.

# Overview

As a general purpose reinforcement learning library, Pearl has been developed to include both
contextual bandit as well as reinforcement learning algorithms. Beyond different algorithm
implementations, the library has different components written in a modular way such that they can be integrated together to test different ideas from sequential decision making in an end to end manner
for different applications.

Pearl has been developed keeping in mind the concept of a reinforcement learning agent, which interacts with an environment to collect data. It has different modules to process and learn from
this data, with the goal of ultimately learning to act optimally in the given environment. We expect
Pearl to be useful for both academic researchers who want to test novel innovations to (a subset) of
agent modules as well as for industry practiotioners who want to test reinforcement learning ideas
for applications to large scale production ready systems

Different Pearl agent modules (functionalities):

* Policy Learners: algorithms which the agent can use to learn an optimal policy from a set of
environment interaction data. Examples:
    * Bandits: linear and neural bandit algorithms.
    * Reinforcement learning:
        1) Value function based methods - (deep) Q-learning, SARSA
        2) Policy optimization methods - REINFORCE, proximal policy optimization (PPO)
        3) Actor critic methods - soft actor critic (SAC), deep deterministic policy gradient (DDPG),
           Twin delayed deep deterministic policy gradient (TD3)
        4) Offline RL methods: Conservative Q-learning (CQL), Implicit Q-learning
        5) Distributional policy learning methods: Quantile regression deep Q-learning.

* Exploration algorithms:
    * Bandits: Upper confidence bound (UCB) and Thompson sampling based exploration methods.
    * Reinforcement learning:
        1) Noise based exploration: epsilon greedy (for discrete action space) and random noise
           injection (for continuous action space).
        2) Propensity based exploration for stochastic policies.
        3) Posterior sampling methods using ensemble approximation (model episetemic uncertainty
           in value function approximation).

* Replay Buffers:
    * Bandits: simple first-in-first-out (FIFO) replay buffer for environment interactions with
      bandit feedback.
    * Reinforcement learning:
        1) On-policy and off-policy FIFO replay buffers with transition tuples (state, action, reward,
           next_state, next_acton).
        2) Hindsight experience replay: specialized method for credit assignment in sparse reward
           settings.

* Safety Modules:
    * Risk sensitive safety module: specific to distributional policy learning algorithms. Value
      function estimates are computed using the estimated return distribution under different choices of risk measures - for e.g. Value at risk (Var), conditional value at risk (CVar), Mean-variance
      etc.
    * Constrained policy optimization: Hard or soft constraints based on auxiliary cost functions
      specified by the environment - constraints on policy search space. Policy learner module takes this into account while searching for the optimal policy.
    * Constrained action space module: filters action space based on environment defined safety
      safety constraints.

* History summarization module: to learn a state representation based on a sequence of past environment
interaction tuples.
    * RNN, LSTM and Attention based history summarization
