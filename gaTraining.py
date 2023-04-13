# -*- coding: utf-8 -*-
"""
GA training for OpenAI environments using PyGAD

(NOTE Ryan Heminway 3/29) Currently WIP... playing around with setup
"""
import gymnasium as gym
import pandas as pd
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
    
    total_rew = 0
    num_games = 10
    for i in range(num_games):
        total_rew += run_in_env(model, env)[0]
    
    fitness = (total_rew / num_games)
    return fitness
    
def run_in_env(model, env):
    """
    Do a run in the environment, and collect reward. Given model should have
    weights loaded. 
    """
    total_reward = 0
    done = False
    observation, info = env.reset()
    num_steps = 0
    while(not done):
        # Run model on observation to get activations for each action
        action_activations = model(torch.from_numpy(observation))
        # Pick action with highest activation 
        action = np.argmax(action_activations.detach().numpy()) 
        # Step in environment using that action
        observation, reward, terminated, truncated, info = env.step(action)
        # Collect reward from step
        total_reward += reward
        num_steps += 1
        if (terminated or truncated):
            done = True
    env.close()
    # Fitness is total reward
    return total_reward, num_steps

def callback_generation(ga_instance):
    """
    Callback function provided to PyGAD. Executes after every generation is
    done. Used here to evaluate the state of the model throughout the course
    of training. 

    Parameters
    ----------
    ga_instance : pygad.GA instance used for training.

    Returns
    -------
    None.

    """
    global df, model, env, df_name
    
    gen = ga_instance.generations_completed
    print("Generation complete: ", gen)
    if gen % 1 == 0:
        # Grab best solution
        solution, _, _ = ga_instance.best_solution()
        best_solution_weights = torchga.model_weights_as_dict(model=model,
                                                              weights_vector=solution)
        model.load_state_dict(best_solution_weights)
        
        pop = ga_instance.pop_size[0]
        num_evals = 25
        for i in range(num_evals):
            reward, steps = run_in_env(model, env)
            df = pd.concat([df, pd.DataFrame.from_records([{'Generation': gen,
                            'Eval': i, 
                            'TotalReward': reward,
                            'Success': str((reward > 200)),
                            'NumSteps': steps,
                            'Pop': pop}])], ignore_index=True)
            
        df.to_csv(df_name)
            

        # humanEnv = gym.make("LunarLander-v2",
        #                render_mode = "human")
        # print("Got run results for gen [", gen, "]: ", run_in_env(model, humanEnv))



def train_and_eval_model(env, model, df): 
    """
    Trains a given model on a given environment, and evaluates the best solution
    after every generation. Stores results in the given dataframe. This represents
    a single training session.

    Parameters
    ----------
    env : Gymnasium environment to evaluate.
    model : Torch model to use for training and agent control.
    df : Pandas dataframe to use for reporting.

    Returns
    -------
    PyGAD GA instance resulting from training. 
    """
    # Create an instance of the pygad.torchga.TorchGA class that will build a 
    # population where each individual is a vector representing the weights
    # and biases of the model
    # (TODO Ryan) How is weight initialization done here?
    torch_ga = torchga.TorchGA(model=model,
                               num_solutions=100)

    # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
    num_generations = 100 # Number of generations.
    num_parents_mating = 2 # Number of solutions to be selected as parents in the mating pool.
    initial_population = torch_ga.population_weights # Initial population of network weights

    # (TODO Ryan) What settings do we want for (1) mutation type, (2) mutation rate,
    # (3) crossover type, (4) crossover rate, (5) selection style, (6) elitism params
    ga_instance = pygad.GA(num_generations=num_generations, 
                           num_parents_mating=num_parents_mating, 
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           on_generation=callback_generation,
                           parent_selection_type="tournament", # tournament                           keep_elitism=10, # Elites
                           K_tournament=10,
                           crossover_probability=0.0,
                           mutation_by_replacement=True,
                           mutation_percent_genes=10,
                           keep_elitism=10)
                        
    ga_instance.run()
    return ga_instance
    

if __name__ == '__main__':
    #env = gym.make("MountainCar-v0")
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    # Create the PyTorch model
    #model = MountainCarNet()

    
    training_loops = 5
    for i in range(training_loops):
        model = LunarLanderNet()
        df = pd.DataFrame(columns=['Generation', 'Eval', 'TotalReward', 'NumSteps', 'Success', 'Pop'])
        df_name = "{env}GARUN={run}.csv".format(env=env_name, run=i)
        ga_instance = train_and_eval_model(env, model, df)
        print(df.to_string())
        df.to_csv(df_name)
    


    # # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    # ga_instance.plot_fitness(title="GA For Lunar Lander -- Generation vs Total Reward", linewidth=4)

    # # Returning the details of the best solution.
    # solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    # print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    # # Fetch the parameters of the best solution.
    # best_solution_weights = torchga.model_weights_as_dict(model=model,
    #                                                       weights_vector=solution)
    # model.load_state_dict(best_solution_weights)
    # #env = gym.make("MountainCar-v0", render_mode="human")
    # env = gym.make("LunarLander-v2",
    #                render_mode = "human")
    # run_in_env(model, env)