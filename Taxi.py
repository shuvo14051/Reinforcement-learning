import gym
import random
import numpy as np

# DeprecatedEnv: Env Taxi-v2 not found (valid versions include ['Taxi-v3'])
env = gym.make('Taxi-v3')

"""
This command will display a popup window. Since it is written within a loop,
an updated popup window will be rendered for every new action taken in each step.
"""
env.render() 

action_size = env.action_space.n
print("Action size ", action_size)
state_size = env.observation_space.n
print("State size ", state_size)

qtable = np.zeros((state_size, action_size))
print(qtable)

total_episodes = 10000       
total_test_episodes = 100     
max_steps = 99

learning_rate = 0.7     # learning rate
gamma = 0.618           #discount rate

# Exploration parameters
epsilon = 1.0                
max_epsilon = 1.0             
min_epsilon = 0.01            
decay_rate = 0.01     

for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    
    for step in range(max_steps):
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0,1)
        
        # if random > epsilon then exploitation
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
            
        # else exploration
        else:
            action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma *
                                    np.max(qtable[new_state, :]) - qtable[state, action])
        state = new_state
        if done == True:
            break
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)


env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        env.render()
        action = np.argmax(qtable[state,:])

        new_state, reward, done, info = env.step(action)

        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            break
        state = new_state