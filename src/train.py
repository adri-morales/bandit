from environment import MultiArmedBandit
from agent import BanditAgent
import pandas as pd

n_episodes = 10
n_actions = 10
max_steps = 50

env = MultiArmedBandit(n_actions)

history = []

for episode in range(n_episodes):
    obs, info = env.reset()
    agent = BanditAgent(n_actions, 0.05)
    
    for step in range(max_steps):
        
        action = agent.get_action(obs)
        
        new_obs, reward, done, info = env.step(action)
        
        agent.update(obs, action, reward, done, new_obs)
        
        obs = new_obs
        
        history.append([episode, step, reward, action])
        
history = pd.DataFrame(history, columns=['episode', 'step', 'reward', 'action'])
history.to_csv('data/results/history.csv')