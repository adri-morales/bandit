from agents.greedy.greedy import BanditAgent
import numpy as np

class OptimisticGreedyBanditAgent(BanditAgent):
    
    def __init__(self, n_actions: int, lr: float, qi: float):
        """
        Initializes the optimistic greedy bandit agent with a policy, learning rate, and initial q estimates.

        Args:
            n_actions (int): The number of actions the agent can take.
            lr (float): The learning rate used to update the policy.
            qui (float): The initial Q values for each action.
        """
        super().__init__(n_actions, lr)
        self.qi = qi
        self.policy = np.ones(10) * qi