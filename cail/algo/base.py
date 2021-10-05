import os
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from abc import abstractmethod
from typing import Tuple
from cail.env import NormalizedEnv


class Algorithm:
    """
    Base class for all algorithms

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    seed: int
        random seed
    gamma: float
        discount factor
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device,
            seed: int,
            gamma: float
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma

    def explore(self, state: np.array) -> Tuple[np.array, float]:
        """
        Act with policy with randomness

        Parameters
        ----------
        state: np.array
            current state

        Returns
        -------
        action: np.array
            mean action
        log_pi: float
            log(\pi(a|s)) of the action
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state: np.array) -> np.array:
        """
        Act with deterministic policy

        Parameters
        ----------
        state: np.array
            current state

        Returns
        -------
        action: np.array
            action to take
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor(state.unsqueeze_(0))
        return action.cpu().numpy()[0]

    def step(self, env: NormalizedEnv, state: np.array, t: int, step: int):
        """
        Sample one step in the environment

        Parameters
        ----------
        env: gym.wrappers.TimeLimit
            environment
        state: np.array
            current state
        t: int
            current time step in the episode
        step: int
            current total steps
        """
        pass

    @abstractmethod
    def is_update(self, step: int):
        """
        Whether the time is for update

        Parameters
        ----------
        step: int
            current training step
        """
        pass

    @abstractmethod
    def update(self, writer: SummaryWriter):
        """
        Update the algorithm

        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
        pass

    @abstractmethod
    def save_models(self, save_dir: str):
        """
        Save the model

        Parameters
        ----------
        save_dir: str
            path to save
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


class Expert:
    """
    Base class for all well-trained algorithms

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    """
    def __init__(self, state_shape: np.array, action_shape: np.array, device: torch.device):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.actor = None

    def exploit(self, state: np.array) -> np.array:
        """
        Act with deterministic policy

        Parameters
        ----------
        state: np.array
            current state

        Returns
        -------
        action: np.array
            action to take
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor(state.unsqueeze_(0))
        return action.cpu().numpy()[0]
