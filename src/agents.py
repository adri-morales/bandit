import numpy as np

class BanditAgent:
    """
    A simple bandit agent that learns to select the optimal action given a set of available actions and rewards.

    Attributes:
        policy (np.ndarray): An array containing the estimated value of each action.
        lr (float): The learning rate used to update the policy.
    """

    def __init__(self, n_actions: int, lr: float):
        """
        Initializes a new BanditAgent.

        Args:
            n_actions (int): The number of available actions.
            lr (float): The learning rate used to update the policy.
        """
        self.n_actions = n_actions
        self.policy = np.zeros(n_actions)
        self.lr = lr

    def get_action(self, obs: int) -> int:
        """
        Selects the optimal action given the current policy.

        Args:
            obs (int): The current observation.

        Returns:
            int: The index of the optimal action.
        """
        return np.argmax(self.policy)

    def update(self, obs: int, action: int, reward: float, terminated: bool, next_obs: int) -> None:
        """
        Updates the policy given the current transition.

        Args:
            obs (int): The current observation.
            action (int): The action taken in the current observation.
            reward (float): The reward received for taking the action.
            terminated (bool): Whether or not the episode has terminated.
            next_obs (int): The observation obtained after taking the action.
        """
        self.policy[action] += self.lr * (reward - self.policy[action])
        
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