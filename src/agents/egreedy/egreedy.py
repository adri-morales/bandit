from agents.greedy.greedy import BanditAgent
import numpy as np

class EGreedyBanditAgent(BanditAgent):
    
    def __init__(self, n_actions: int, lr: float, epsilon: float):
        """
        Initializes the e-greedy bandit agent with a policy, learning rate, and exploration rate.

        Args:
            n_actions (int): The number of actions the agent can take.
            lr (float): The learning rate used to update the policy.
            epsilon (float): The exploration rate used to select a random action.
        """
        super().__init__(n_actions, lr)
        self.epsilon = epsilon
        
    def get_action(self, obs: int) -> int:
        """
        Selects an action given the current policy. Selects optimal action with probability 1-epsilon
        and a random action with probability epsilon

        Args:
            obs (int): The current observation.

        Returns:
            int: The index of the selected action.
        """
        
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.argmax(self.policy)
            
        return action