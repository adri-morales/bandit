from agents.greedy.greedy import BanditAgent
import numpy as np

class UCBBanditAgent(BanditAgent):
    
    def __init__(self, n_actions: int, lr: float, c: float):
        super().__init__(n_actions, lr)
        self.c = c
        self.step_counter = 0
        self.action_counter = np.zeros(self.n_actions)
    
    def get_action(self, obs: int) -> int:
        """
        Selects an action based on UCB criteria.

        Args:
            obs (int): The current observation.

        Returns:
            int: The index of selected action.
        """
        
        ucb_list = []
        for i, q in enumerate(self.policy):
            
            t = self.step_counter
            na = self.action_counter[i] + 1e-6
            
            ucb = q + self.c * np.sqrt(np.log(t)/na)
            ucb_list.append(ucb)
            self.step_counter += 1
            self.action_counter[i] += 1
            
        action = np.argmax(ucb_list)
            
        return action