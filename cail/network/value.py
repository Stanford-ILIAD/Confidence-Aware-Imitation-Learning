import torch
import numpy as np

from torch import nn
from typing import Tuple
from .utils import build_mlp


class StateFunction(nn.Module):
    """
    Value function that takes states as input

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    hidden_units: tuple
        hidden units of the value function
    hidden_activation: nn.Module
        hidden activation of the value function
    """
    def __init__(
            self,
            state_shape: np.array,
            hidden_units: tuple = (64, 64),
            hidden_activation: nn.Module = nn.Tanh()
    ):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            init=True
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Return values of the states

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        values: torch.Tensor
            values of the states
        """
        return self.net(states)


class StateActionFunction(nn.Module):
    """
    Value function that takes s-a pairs as input

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    hidden_units: tuple
        hidden units of the value function
    hidden_activation: nn.Module
        hidden activation of the value function
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            hidden_units: tuple = (100, 100),
            hidden_activation=nn.Tanh()
    ):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Return values of the s-a pairs

        Parameters
        ----------
        states: torch.Tensor
            input states
        actions: torch.Tensor
            actions corresponding to the states

        Returns
        -------
        values: torch.Tensor
            values of the s-a pairs
        """
        return self.net(torch.cat([states, actions], dim=-1))


class TwinnedStateActionFunction(nn.Module):
    """
    Twinned value functions that takes s-a pairs as input, used in SAC

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    hidden_units: tuple
        hidden units of the value function
    hidden_activation: nn.Module
        hidden activation of the value function
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            hidden_units: tuple = (256, 256),
            hidden_activation: nn.Module = nn.ReLU(inplace=True)
    ):
        super().__init__()

        self.net1 = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.net2 = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return twinned values of the s-a pairs

        Parameters
        ----------
        states: torch.Tensor
            input states
        actions: torch.Tensor
            actions corresponding to the states

        Returns
        -------
        values_1: torch.Tensor
            values of the s-a pairs
        values_2: torch.Tensor
            values of the s-a pairs
        """
        xs = torch.cat([states, actions], dim=-1)
        return self.net1(xs), self.net2(xs)

    def q1(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Return values of the s-a pairs

        Parameters
        ----------
        states: torch.Tensor
            input states
        actions: torch.Tensor
            actions corresponding to the states

        Returns
        -------
        values_1: torch.Tensor
            values of the s-a pairs
        """
        return self.net1(torch.cat([states, actions], dim=-1))
