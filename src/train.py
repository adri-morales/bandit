from environment import MultiArmedBandit
import sys
import pandas as pd
from tqdm import tqdm
import argparse
import ast

sys.path.append('src/agents')
import greedy
import egreedy
import optimistic
import ucb

parser = argparse.ArgumentParser(description='Run a reinforcement learning agent.')
# Agent options
parser.add_argument('-a', '--agent', type=str, choices=['greedy', 'epsilon_greedy', 'ucb', 'optimistic_greedy'], required=True,
                    help='the agent to use for training')

# Agent parameter options
parser.add_argument('-p', '--parameters', nargs='+', required=True,
                    help='the parameters for the agent in the format <parameter_name>:<value>')

args = parser.parse_args()

# Parse agent parameters into a dictionary
agent_params = {}
if args.parameters is not None:
    for param in args.parameters:
        key, value = param.split(':')
        
        value = ast.literal_eval(value)
            
        agent_params[key] = value

n_episodes = 1000
n_actions = 10
max_steps = 1000

agent_dict = {'greedy': greedy.BanditAgent, 
              'epsilon_greedy': egreedy.EGreedyBanditAgent,
              'ucb': ucb.UCBBanditAgent,
              'optimistic_greedy': optimistic.OptimisticGreedyBanditAgent}

agent_class = agent_dict[args.agent]

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
        
        history.append([episode, step, reward, action, args.agent])
        
history = pd.DataFrame(history, columns=['episode', 'step', 'reward', 'action', 'agent'])
history.to_csv(f'data/results/{args.agent}_history.csv')