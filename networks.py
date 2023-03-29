# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

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
        # Selecting hidden layer size of 8 for now
        self.hidden_size = 8
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

