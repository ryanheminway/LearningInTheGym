# -*- coding: utf-8 -*-
import gymnasium as gym
import pandas as pd
import pygad
from pygad import torchga
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys


## HELPER FUNCTIONS ## 

def fitness_func(ga_instance, solution, sol_idx):
    """
    Fitness function for GA will create a NN model out of the individual solution
    (vector of weights) and use that model to control the agent in the given
    environment. The fitness of individual is the total reward collected during
    the run in the environment. 

    (TODO Ryan) How long should a run be? Until completion of a single instance?
                Could also be a fixed number of timesteps?
    """
    global torch_ga, model, env, device

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
    global device
    model.to(device)
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
    global df, model, env, df_name, device
    
    gen = ga_instance.generations_completed
    #print("Generation complete: ", gen)
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
    torch_ga = torchga.TorchGA(model=model,
                               num_solutions=100)

    # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
    num_generations = 100 # Number of generations.
    num_parents_mating = 2 # Number of solutions to be selected as parents in the mating pool.
    initial_population = torch_ga.population_weights # Initial population of network weights


    ga_instance = pygad.GA(num_generations=num_generations, 
                           num_parents_mating=num_parents_mating, 
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           on_generation=callback_generation,
                           parent_selection_type="tournament", # tournament                           
                           K_tournament=10,
                           crossover_probability=0.6,
                           mutation_by_replacement=True,
                           mutation_percent_genes=10,
                           keep_elitism=10)
                        
    ga_instance.run()
    return ga_instance


## NETWORKS

class BaseNet(nn.Module):
    """
    Base Neural Network definiton that can be applied to all problems we are
    covering for the CS5335 project. Based on the problem (Gym environment)
    at hand, the input layer and output layer sizes will need to change. All
    else can remain the same. See the subclasses below which have different
    default values in the constructor, referring to the change in setup.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.hidden_size = 64
        self.network = nn.Sequential(
            nn.Linear(in_dim, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, out_dim)
        )
    
    def forward(self, x):
        out = self.network(x)
        return out
        


class MountainCarNet(BaseNet):
    """
    Neural Network definition for the Mountain Car environment in Open AI 
    Gym. The input layer size is 2 because each observation is a vector
    of 2 floating point values. The output layer size is 3 because each 
    action is one of ["0: Accelerate to the left", "1: Don’t accelerate",
                      "2: Accelerate to the right"]
    
    https://gymnasium.farama.org/environments/classic_control/mountain_car/
    """
    def __init__(self):
        super().__init__(2, 3)
        


class LunarLanderNet(BaseNet):
    """
    Neural Network definition for the Lunar Lander environment in Open AI
    Gymnasium. Input layer size is 8 because observations are a vector of
    8 floating point values. Output layer is size 4 because each action is
    one of ["0: do nothing", "1: fire left orientation engine" ,
            "2: fire main engine", "3: fire right orientation engine"]
    
    https://gymnasium.farama.org/environments/box2d/lunar_lander/
    """
    def __init__(self):
        super().__init__(8, 4)
        
        
        
# TRAINING 
def run_model(run_idx, model_path):
  global drive_path
  # Create the PyTorch model
  model = LunarLanderNet()
  df = pd.DataFrame(columns=['Generation', 'Eval', 'TotalReward', 'NumSteps', 'Success', 'Pop'])
  df_name = model_path + "{env}GARUN={run}.csv".format(env=env_name, run=run_idx)
  ga_instance = train_and_eval_model(env, model, df)
  #print(df.to_string())
  df.to_csv(df_name)
  return df.to_string()
        
        
if __name__ == '__main__':       
    # MAKE ENV
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    
    model_path = str(Path.cwd()) + "/results/" + "gaForLunar/"
    Path(model_path).mkdir(parents=True, exist_ok=True)
    
    # SETUP GPU ACCESS IF POSSIBL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Run TRAINING_LOOPS runs in parallel with multiprcessing
    run_model(0, model_path)


