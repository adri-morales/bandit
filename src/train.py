from environment import MultiArmedBandit
import sys
import pandas as pd
from tqdm import tqdm
import argparse
import ast
import yaml
import os

for subdir, _, _ in os.walk('src/agents'):
    sys.path.append(subdir)

import greedy
import egreedy
import optimistic
import ucb

parser = argparse.ArgumentParser(description='Run a reinforcement learning agent.')
# Agent options
parser.add_argument('-c', '--config', type=str, default='config.yaml')

args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

n_episodes = 1000
n_actions = 10
max_steps = 1000

agent_dict = {'greedy': greedy.BanditAgent, 
              'epsilon_greedy': egreedy.EGreedyBanditAgent,
              'ucb': ucb.UCBBanditAgent,
              'optimistic_greedy': optimistic.OptimisticGreedyBanditAgent}

agent_str = config['agent']
agent_class = agent_dict[agent_str]

del config['agent']
agent_params = config

env = MultiArmedBandit(n_actions)

history = []

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    agent = agent_class(**agent_params)
    
    for step in range(max_steps):
        
        action = agent.get_action(obs)
        
        new_obs, reward, done, info = env.step(action)
        
        agent.update(obs, action, reward, done, new_obs)
        
        obs = new_obs
        
        history.append([episode, step, reward, action, agent_str])
        
history = pd.DataFrame(history, columns=['episode', 'step', 'reward', 'action', 'agent'])
history.to_csv(f'results/{agent_str}_history.csv')