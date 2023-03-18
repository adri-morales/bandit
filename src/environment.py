import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiArmedBandit(gym.Env):
    """
    Custom environment for the multi-armed bandit problem, where the agent must choose one of N arms 
    in each round and receive a reward for that action. 
    
    Attributes:
        n_arms (int): The number of arms of the bandit.
        stationary (bool): Whether the rewards for each arm are stationary or not.
    """
    metadata = {}

    def __init__(self, n_arms, stationary=True):
        super(MultiArmedBandit, self).__init__()
        self.n_arms = n_arms
        self.stationary = stationary

        # Define the observation space (only one state)
        self.observation_space = spaces.Discrete(1)
        
        # Define the action space
        self.action_space = spaces.Discrete(self.n_arms)
        
        # Initialize arm means and state
        self.arm_means=None
        self.state = 0
    
    def _get_obs(self):
        """
        Helper function to retrieve the current observation.
        """
        return self.state
    
    def _get_info(self):
        """
        Helper function to retrieve information about the environment.
        """
        return {'means': self.arm_means}

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state and returns the initial observation.
        
        Returns:
            observation (int): The initial observation (always 0).
            info (dict): A dictionary containing information about the environment, such as the mean reward of each arm.
        """
        super().reset(seed=seed)
        
        # Generate an array of n_arms values following a normal distribution representing the mean reward of each arm
        self.arm_means = np.random.normal(0, 1, self.n_arms)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        """
        Executes an action in the environment and returns a tuple of (observation, reward, done, info).
        
        Args:
            action (int): The action to be taken by the agent (an integer between 0 and n_arms-1).
        
        Returns:
            observation (int): The next observation (always 0).
            reward (float): The reward obtained for the chosen action.
            done (bool): Whether the episode has ended or not (always False).
            info (dict): A dictionary containing information about the environment, such as the mean reward of each arm.
        """
        mean = self.arm_means[action]
        reward = np.random.normal(mean, 1)
        done = False
        info = self._get_info()
        observation = self._get_obs()
        
        return observation, reward, done, info


if __name__ == '__main__':
    # Create an instance of the environment
    env = MultiArmedBandit(10)
    
    # Reset the environment and retrieve the initial observation and information
    obs, info = env.reset()
    
    # Run 10 steps by taking random actions and print the results
    for i in range(10):
        random_action = np.random.randint(env.n_arms)
        print(random_action)
        obs, reward, done, info = env.step(random_action)
        print(obs, reward, done, info)
        print()
