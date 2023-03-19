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
        self.policy = np.random.normal(0, 1, n_actions)
        self.lr = lr

    def get_action(self, obs: np.ndarray) -> int:
        """
        Selects the optimal action given the current policy.

        Args:
            obs (np.ndarray): The current observation.

        Returns:
            int: The index of the optimal action.
        """
        return np.argmax(self.policy)

    def update(self, obs: np.ndarray, action: int, reward: float, terminated: bool, next_obs: np.ndarray) -> None:
        """
        Updates the policy given the current transition.

        Args:
            obs (np.ndarray): The current observation.
            action (int): The action taken in the current observation.
            reward (float): The reward received for taking the action.
            terminated (bool): Whether or not the episode has terminated.
            next_obs (np.ndarray): The observation obtained after taking the action.
        """
        self.policy[action] += self.lr * (reward - self.policy[action])
