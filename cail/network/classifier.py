import torch
import torch.nn as nn
import numpy as np


class Classifier(nn.Module):
    """
    Neural Network classifier

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    hidden_units: tuple
        hidden units of the classifier
    hidden_activation: nn.Module
        hidden activation of the classifier
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            hidden_units: tuple = (64, 64),
            hidden_activation: nn.Module = nn.Tanh()
    ):
        super(Classifier, self).__init__()

        layers = []
        units = state_shape[0] + action_shape[0]
        for next_units in hidden_units:
            layers.append(nn.Linear(units, next_units))
            layers.append(hidden_activation)
            layers.append(nn.Dropout(0.5))
            units = next_units
        layers.append(nn.Linear(units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify

        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
        result: torch.Tensor
            classify result
        """
        return self.net(x)
