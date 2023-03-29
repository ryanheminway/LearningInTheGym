This repository stores the source code used for the CS5335 Robotics Final Project done by Ryan Heminway and Danya Gordin. The project is an exploration of Genetic Algorithms and Q-Networks as approaches for learning in environments typically tackled via Reinforcement Learning. We use OpenAI Gymnasium environments as a testbed for the solutions.

# File Overview

`gaTraining.py` -- Trains a GA on a given Gymnasium environment using PyGAD. Run it with `python gaTraining.py`. It will print the fitness of the best individual in the population at each generation, then display a graph of the results. Once you close the graph, it will load the best found model and run the environment with `render_mode="human"` so you can see the results.

`networks.py` -- General Neural Network definition in PyTorch. These networks should be used in both the GA and QDL settings. The `BaseNet` class is a general feed-forward linear MLP with one hidden layer. The subclasses of `MountainCarNet` and `LunarLanderNet` specify the correct input and output dimensions for their respective environments.

`requirements.txt` and `environment.yml` -- Lists of packages I used for the python environment. Two formats of the same information. I use Anaconda for my Python environment management. 