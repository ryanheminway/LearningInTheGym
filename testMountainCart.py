import numpy as np
import gymnasium as gym
env = gym.make("MountainCar-v0", render_mode="human")
observation, info = env.reset(seed=42)
print("observation: ", observation)
print("as numpy: ", np.asarray(observation))
print("info: ", info)
"""
THIS FILE IS NO LONGER USED. WAS JUST FOR INITIAL TESTS

Notes to self for implementation:
    1) We should try to use a Tensorflow or PyTorch GA implementation so we can
       use the same NN in both QDL and GA. 
           -- Looks like PyGAD is our best option. 
    2) Fitness function for GA will be single run of mountain car? Should we have any special
        fitness functions? Like I would imagine maybe making a fitness function which maximized
        distance travelled may help the search.
    3) Size of NN if we're going to use same one for all runs? Maybe one hidden
       layer with size 8 or something? Since inputs will be 2-8 dimensional vectors?
"""
for i in range(200):
    #print(i)
    if (i < 30):
        action = 0
    else:
        action = 2
    #action = env.action_space.sample() # this is where you would insert your policy
    #print("action: ", action)
    observation, reward, terminated, truncated, info = env.step(action)
    print("reward: ", reward)

    if terminated or truncated:
        observation, info = env.reset()
env.close()