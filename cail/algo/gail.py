import torch
import torch.nn.functional as F
import os
import numpy as np

from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from .ppo import PPO, PPOExpert
from cail.network import GAILDiscrim
from cail.buffer import SerializedBuffer


class GAIL(PPO):
    """
    Implementation of GAIL, using PPO as the backbone RL algorithm

    Reference:
    ----------
    [1] Ho, J. and Ermon, S.
    Generative adversarial imitation learning.
    In Advances in neural information processing systems, pp. 4565–4573, 2016.

    Parameters
    ----------
    buffer_exp: SerializedBuffer
        buffer of demonstrations
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
    rollout_length: int
        rollout length of the buffer
    mix_buffer: int
        times for rollout buffer to mix
    batch_size: int
        batch size for sampling from current policy and demonstrations
    lr_actor: float
        learning rate of the actor
    lr_critic: float
        learning rate of the critic
    lr_disc: float
        learning rate of the discriminator
    units_actor: tuple
        hidden units of the actor
    units_critic: tuple
        hidden units of the critic
    units_disc: tuple
        hidden units of the discriminator
    epoch_ppo: int
        at each update period, update ppo for these times
    epoch_disc: int
        at each update period, update the discriminator for these times
    clip_eps: float
        clip coefficient in PPO's objective
    lambd: float
        lambd factor
    coef_ent: float
        entropy coefficient
    max_grad_norm: float
        maximum gradient norm
    """
    def __init__(
            self,
            buffer_exp: SerializedBuffer,
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device,
            seed: int,
            gamma: float = 0.995,
            rollout_length: int = 50000,
            mix_buffer: int = 1,
            batch_size: int = 64,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            lr_disc: float = 3e-4,
            units_actor: tuple = (64, 64),
            units_critic: tuple = (64, 64),
            units_disc: tuple = (100, 100),
            epoch_ppo: int = 50,
            epoch_disc: int = 10,
            clip_eps: float = 0.2,
            lambd: float = 0.97,
            coef_ent: float = 0.0,
            max_grad_norm: float = 10.0
    ):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # expert's buffer
        self.buffer_exp = buffer_exp

        # discriminator
        self.disc = GAILDiscrim(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_disc,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, writer: SummaryWriter):
        """
        Update the algorithm

        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # samples from current policy's trajectories
            states, actions = self.buffer.sample(self.batch_size)[:2]

            # samples from expert's demonstrations
            states_exp, actions_exp = self.buffer_exp.sample(self.batch_size)[:2]

            # update discriminator
            self.update_disc(states, actions, states_exp, actions_exp, writer)

        # we don't use reward signals here
        states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # calculate rewards
        rewards = self.disc.calculate_reward(states, actions)

        # update PPO using estimated rewards
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states, writer)

    def update_disc(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            states_exp: torch.Tensor,
            actions_exp: torch.Tensor,
            writer: SummaryWriter
    ):
        """
        Train the discriminator to distinguish the expert's behavior
        and the imitation learning policy's behavior

        Parameters
        ----------
        states: torch.Tensor
            states sampled from current IL policy
        actions: torch.Tensor
            actions sampled from current IL policy
        states_exp: torch.Tensor
            states sampled from demonstrations
        actions_exp: torch.Tensor
            actions sampled from demonstrations
        writer: SummaryWriter
            writer for logs
        """
        # output of discriminator is (-inf, inf), not [0, 1]
        logits_pi = self.disc(states, actions)
        logits_exp = self.disc(states_exp, actions_exp)

        # discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)]
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar('loss/disc', loss_disc.item(), self.learning_steps)

            # discriminator's accuracies
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)

    def save_models(self, save_dir: str):
        """
        Save the model

        Parameters
        ----------
        save_dir: str
            path to save
        """
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        torch.save(self.disc.state_dict(), f'{save_dir}/disc.pkl')
        torch.save(self.actor.state_dict(), f'{save_dir}/actor.pkl')


class GAILExpert(PPOExpert):
    """
    Well-trained GAIL agent

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    path: str
        path to the well-trained weights
    units_actor: tuple
        hidden units of the actor
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device,
            path: str,
            units_actor: tuple = (64, 64)
    ):
        super(GAILExpert, self).__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            path=path,
            units_actor=units_actor
        )
