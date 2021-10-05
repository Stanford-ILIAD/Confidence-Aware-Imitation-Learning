import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from .utils import build_mlp, build_param_list


class GAILDiscrim(nn.Module):
    """
    Discriminator used by GAIL, which takes s-a pair as input and output
    the probability that the s-a pair is sampled from demonstrations

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    hidden_units: tuple
        hidden units of the discriminator
    hidden_activation: nn.Module
        hidden activation of the discriminator
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            hidden_units: tuple = (100, 100),
            hidden_activation: nn.Module = nn.Tanh()
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
        Run discriminator

        Parameters
        ----------
        states: torch.Tensor
            input states
        actions: torch.Tensor
            actions corresponding to the states

        Returns
        -------
        result: torch.Tensor
            probability that this s-a pair belongs to demonstrations
        """
        return self.net(torch.cat([states, actions], dim=-1))

    def calculate_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Calculate reward using GAIL's learned reward signal log(D)

        Parameters
        ----------
        states: torch.Tensor
            input states
        actions: torch.Tensor
            actions corresponding to the states

        Returns
        -------
        rewards: torch.Tensor
            reward signal
        """
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(states, actions))


class AIRLDiscrim(nn.Module):
    """
    Discriminator used by AIRL, which takes s-a pair as input and output
    the probability that the s-a pair is sampled from demonstrations

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    gamma: float
        discount factor
    hidden_units_r: tuple
        hidden units of the discriminator r
    hidden_units_v: tuple
        hidden units of the discriminator v
    hidden_activation_r: nn.Module
        hidden activation of the discriminator r
    hidden_activation_v: nn.Module
        hidden activation of the discriminator v
    """
    def __init__(
            self,
            state_shape: np.array,
            gamma: float,
            hidden_units_r: tuple = (64, 64),
            hidden_units_v: tuple = (64, 64),
            hidden_activation_r: nn.Module = nn.ReLU(inplace=True),
            hidden_activation_v: nn.Module = nn.ReLU(inplace=True)
    ):
        super().__init__()

        self.g = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states: torch.Tensor, dones: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """
        Calculate the f(s, s') function

        Parameters
        ----------
        states: torch.Tensor
            input states
        dones: torch.Tensor
            whether the state is the end of an episode
        next_states: torch.Tensor
            next state corresponding to the current state

        Returns
        -------
        f: value of the f(s, s') function
        """
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(
            self,
            states: torch.Tensor,
            dones: torch.Tensor,
            log_pis: torch.Tensor,
            next_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Output the discriminator's result sigmoid(f - log_pi) without sigmoid

        Parameters
        ----------
        states: torch.Tensor
            input states
        dones: torch.Tensor
            whether the state is the end of an episode
        log_pis: torch.Tensor
            log(\pi(a|s))
        next_states: torch.Tensor
            next state corresponding to the current state

        Returns
        -------
        result: f - log_pi
        """
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(
            self,
            states: torch.Tensor,
            dones: torch.Tensor,
            log_pis: torch.Tensor,
            next_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate reward using AIRL's learned reward signal f

        Parameters
        ----------
        states: torch.Tensor
            input states
        dones: torch.Tensor
            whether the state is the end of an episode
        log_pis: torch.Tensor
            log(\pi(a|s))
        next_states: torch.Tensor
            next state corresponding to the current state

        Returns
        -------
        rewards: torch.Tensor
            reward signal
        """
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)


class AIRLDetachedDiscrim:
    """
    Detached discriminator used by AIRL, which takes s-a pair as input and output
    the probability that the s-a pair is sampled from demonstrations. This
    discriminator can be set parameters manually

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    gamma: float
        discount factor
    hidden_units_r: tuple
        hidden units of the discriminator r
    hidden_units_v: tuple
        hidden units of the discriminator v
    hidden_activation_r: nn.Module
        hidden activation of the discriminator r
    hidden_activation_v: nn.Module
        hidden activation of the discriminator v
    """
    def __init__(
            self,
            state_shape: np.array,
            gamma: float,
            device: torch.device,
            hidden_units_r: tuple = (64, 64),
            hidden_units_v: tuple = (64, 64),
            hidden_activation_r: nn.Module = nn.ReLU(inplace=True),
            hidden_activation_v: nn.Module = nn.ReLU(inplace=True)
    ):
        self.g_weights, self.g_biases = build_param_list(
            input_dim=state_shape[0],
            output_dim=1,
            device=device,
            hidden_units=hidden_units_r,
            requires_grad=True,
        )
        self.h_weights, self.h_biases = build_param_list(
            input_dim=state_shape[0],
            output_dim=1,
            device=device,
            hidden_units=hidden_units_v,
            requires_grad=True
        )

        self.state_shape = state_shape
        self.hidden_units_r = hidden_units_r
        self.hidden_units_v = hidden_units_v
        self.hidden_activation_r = hidden_activation_r
        self.hidden_activation_v = hidden_activation_v
        self.gamma = gamma

    def g(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calculate g(s) function

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        x: torch.Tensor
            g(s)
        """
        x = states
        for weight, bias in zip(self.g_weights, self.g_biases):
            x = self.hidden_activation_r(x.mm(weight.transpose(0, 1)) + bias)
        return x

    def h(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calculate h(s) function

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        x: torch.Tensor
            h(s)
        """
        x = states
        for weight, bias in zip(self.h_weights, self.h_biases):
            x = self.hidden_activation_v(x.mm(weight.transpose(0, 1)) + bias)
        return x

    def f(self, states: torch.Tensor, dones: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """
        Calculate the f(s, s') function

        Parameters
        ----------
        states: torch.Tensor
            input states
        dones: torch.Tensor
            whether the state is the end of an episode
        next_states: torch.Tensor
            next state corresponding to the current state

        Returns
        -------
        f: value of the f(s, s') function
        """
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(
            self,
            states: torch.Tensor,
            dones: torch.Tensor,
            log_pis: torch.Tensor,
            next_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Output the discriminator's result sigmoid(f - log_pi) without sigmoid

        Parameters
        ----------
        states: torch.Tensor
            input states
        dones: torch.Tensor
            whether the state is the end of an episode
        log_pis: torch.Tensor
            log(\pi(a|s))
        next_states: torch.Tensor
            next state corresponding to the current state

        Returns
        -------
        result: f - log_pi
        """
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(
            self,
            states: torch.Tensor,
            dones: torch.Tensor,
            log_pis: torch.Tensor,
            next_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate reward using AIRL's learned reward signal f

        Parameters
        ----------
        states: torch.Tensor
            input states
        dones: torch.Tensor
            whether the state is the end of an episode
        log_pis: torch.Tensor
            log(\pi(a|s))
        next_states: torch.Tensor
            next state corresponding to the current state

        Returns
        -------
        rewards: torch.Tensor
            reward signal
        """
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)

    def set_parameters(self, vector: torch.Tensor):
        """
        Set parameters of the discriminator

        Parameters
        ----------
        vector: torch.Tensor
            vector of parameters
        """
        pointer = 0
        for layer in range(len(self.g_weights)):
            n_param = int(self.g_weights[layer].shape[0] * self.g_weights[layer].shape[1])
            self.g_weights[layer] = vector[pointer: pointer + n_param].view(self.g_weights[layer].shape)
            pointer += n_param
            n_param = self.g_biases[layer].shape[0]
            self.g_biases[layer] = vector[pointer: pointer + n_param].view(self.g_biases[layer].shape)
            pointer += n_param

        for layer in range(len(self.h_weights)):
            n_param = int(self.h_weights[layer].shape[0] * self.h_weights[layer].shape[1])
            self.h_weights[layer] = vector[pointer: pointer + n_param].view(self.h_weights[layer].shape)
            pointer += n_param
            n_param = self.h_biases[layer].shape[0]
            self.h_biases[layer] = vector[pointer: pointer + n_param].view(self.h_biases[layer].shape)
            pointer += n_param

        assert pointer == vector.shape[0]

    def get_parameters(self) -> torch.Tensor:
        """
        Return all parameters in a vector

        Returns
        -------
        vector: torch.Tensor
            vector of parameters
        """
        vector = []
        for layer in range(len(self.g_weights)):
            vector.append(self.g_weights[layer].view(-1))
            vector.append(self.g_biases[layer].view(-1))
        for layer in range(len(self.h_weights)):
            vector.append(self.h_weights[layer].view(-1))
            vector.append(self.h_biases[layer].view(-1))
        return torch.cat(vector)

    def num_param_g(self) -> int:
        """
        Get number of parameters in g(s)

        Returns
        -------
        n: int
            number of parameters in g(s)
        """
        n = 0
        units = self.state_shape[0]
        for next_units in self.hidden_units_r:
            n += next_units * units
            n += next_units
            units = next_units
        n += 1 * units
        n += 1
        return n

    def num_param_h(self) -> int:
        """
        Get number of parameters in h(s)

        Returns
        -------
        n: int
            number of parameters in h(s)
        """
        n = 0
        units = self.state_shape[0]
        for next_units in self.hidden_units_v:
            n += next_units * units
            n += next_units
            units = next_units
        n += 1 * units
        n += 1
        return n
