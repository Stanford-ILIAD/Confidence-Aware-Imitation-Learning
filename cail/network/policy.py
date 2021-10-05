import torch
import numpy as np

from torch import nn
from typing import Tuple
from .utils import build_mlp, reparameterize, evaluate_log_pi


class StateIndependentPolicy(nn.Module):
    """
    Stochastic policy \pi(a|s)

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    hidden_units: tuple
        hidden units of the policy
    hidden_activation: nn.Module
        hidden activation of the policy
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            hidden_units: tuple = (64, 64),
            hidden_activation: nn.Module = nn.Tanh()
    ):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            init=True
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get the mean of the stochastic policy

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            mean of the stochastic policy
        """
        return torch.tanh(self.net(states))

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions given states

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            actions to take
        log_pi: torch.Tensor
            log_pi of the actions
        """
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log(\pi(a|s)) of the given action

        Parameters
        ----------
        states: torch.Tensor
            states that the actions act in
        actions: torch.Tensor
            actions taken

        Returns
        -------
        log_pi: : torch.Tensor
            log(\pi(a|s))
        """
        return evaluate_log_pi(self.net(states), self.log_stds, actions)


class StateDependentPolicy(nn.Module):
    """
    State dependent policy defined in SAC

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    hidden_units: tuple
        hidden units of the policy
    hidden_activation: nn.Module
        hidden activation of the policy
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            hidden_units: tuple = (256, 256),
            hidden_activation: nn.Module = nn.ReLU(inplace=True)
    ):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get the mean of the stochastic policy

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            mean of the stochastic policy
        """
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions given states

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            actions to take
        log_pi: torch.Tensor
            log_pi of the actions
        """
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp(-20, 2))


class DeterministicPolicy(nn.Module):
    """
    Deterministic policy

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    hidden_units: tuple
        hidden units of the policy
    hidden_activation: nn.Module
        hidden activation of the policy
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            hidden_units: tuple = (256, 256),
            hidden_activation: nn.Module = nn.ReLU(inplace=True)
    ):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            init=True
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get actions given states

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            actions to take
        """
        return torch.tanh(self.net(states))
