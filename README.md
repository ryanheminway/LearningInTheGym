# Navigating Reward Structures: A Comparison of Deep Q-Learning and Genetic Algorithms in Diverse Environments

## Authors: Ryan Heminway and Danya Gordin

## Project Overview

This repository stores the source code used for the CS5335 Robotics Final Project. The project is an exploration of Genetic Algorithms and Deep Q-Networks as approaches for learning in environments typically tackled via Reinforcement Learning. We use OpenAI Gymnasium environments (LunarLander-v2 and MountainCar-v0) as a testbed for the solutions. More specifically, our work investigates the effect of varying reward structures on the GA and DQN algorithms. In the end, we find that GA is a competitive alternative to DQN especically for sparse reward environments. For more details, please refer to the paper which is uncluded in the repository.

## Repository Breakdown

All implementation work was done in Python. Most of the code was implemented and executed in Google Colab, with some experiments being executed on the Northeastern Discover Computing Cluster for the ability to parallelize multiple runs at once. At the root level of the project, there are self-sufficient Colab notebooks for training and evaluating GA and DQN models on the LunarLander-v2 and MountainCar-v0 environments. Additionally, the `slurm` folder holds python scripts and resources for executing GA experiments on the Discover cluster using the Slurm resource allocation toolset. Within the `slurm` folder, there is an additional `pyenv` folder which has two formats for describing the full Python Conda environment used for the experiments. Finally, the `plot_results.py` script at the root level of the project contains useful plotting functions that process the CSV files produced by experiments.
