# -*- coding: utf-8 -*-
"""
GA training for OpenAI environments using PyGAD

(NOTE Ryan 3/29) Currently WIP... playing around with setup
"""
import gymnasium as gym
import pygad
from pygad import torchga
import numpy as np
import torch
from networks import BaseNet, MountainCarNet, LunarLanderNet

def fitness_func(solution, sol_idx):
    """
    Fitness function for GA will create a NN model out of the individual solution
    (vector of weights) and use that model to control the agent in the given
    environment. The fitness of individual is the total reward collected during
    the run in the environment. 

    (TODO Ryan) How long should a run be? Until completion of a single instance?
                Could also be a fixed number of timesteps?
    """
    global torch_ga, model, env

    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=solution) 
        
    # Use the current solution as the model parameters.
    model.load_state_dict(model_weights_dict)
    
    return run_in_env(model, env)
    
def run_in_env(model, env):
    """
    Do a run in the environment, and collect reward. Given model should have
    weights loaded. 
    """
    total_reward = 0
    done = False
    observation, info = env.reset(seed=42)
    while(not done):
        # Run model on observation to get activations for each action
        action_activations = model(torch.from_numpy(observation))
        # Pick action with highest activation 
        action = np.argmax(action_activations.detach().numpy()) 
        # Step in environment using that action
        observation, reward, terminated, truncated, info = env.step(action)
        # Collect reward from step
        total_reward = total_reward + reward
        if (terminated or truncated):
            done = True
    env.close()
    # Fitness is total reward
    return total_reward

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


if __name__ == '__main__':
    #env = gym.make("MountainCar-v0")
    env = gym.make("LunarLander-v2",
                   continuous = False,
                   gravity = -10.0,
                   enable_wind = False
                   )

    # Create the PyTorch model
    #model = MountainCarNet()
    model = LunarLanderNet()

    # Create an instance of the pygad.torchga.TorchGA class that will build a 
    # population where each individual is a vector representing the weights
    # and biases of the model
    # (TODO Ryan) How is weight initialization done here?
    torch_ga = torchga.TorchGA(model=model,
                               num_solutions=10)

    # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
    num_generations = 100 # Number of generations.
    num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
    initial_population = torch_ga.population_weights # Initial population of network weights

    # (TODO Ryan) What settings do we want for (1) mutation type, (2) mutation rate,
    # (3) crossover type, (4) crossover rate, (5) selection style, (6) elitism params
    ga_instance = pygad.GA(num_generations=num_generations, 
                           num_parents_mating=num_parents_mating, 
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           on_generation=callback_generation)

    ga_instance.run()

    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_fitness(title="GA For Mountain Car -- Generation vs Total Reward", linewidth=4)

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    # Fetch the parameters of the best solution.
    best_solution_weights = torchga.model_weights_as_dict(model=model,
                                                          weights_vector=solution)
    model.load_state_dict(best_solution_weights)
    #env = gym.make("MountainCar-v0", render_mode="human")
    env = gym.make("LunarLander-v2",
                   continuous = False,
                   gravity = -10.0,
                   enable_wind = False,
                   render_mode = "human")
    run_in_env(model, env)