import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.convert_parameters import parameters_to_vector
from tqdm import tqdm
from .ppo import PPO, PPOExpert
from cail.network import StateFunction
from cail.buffer import SerializedBuffer


class TREX(PPO):
    """
    Implementation of T-REX

    Reference:
    ----------
    [1] Brown, D., Goo, W., Nagarajan, P., and Niekum, S.
    Extrapolating beyond suboptimal demonstrations via inverse reinforcement learning from observations.
    In International Conference on Machine Learning, pp. 783–792. PMLR, 2019.

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
    lr_reward: float
        learning rate of the T-REX's reward function
    units_actor: tuple
        hidden units of the actor
    units_critic: tuple
        hidden units of the critic
    units_reward: tuple
        hidden units of the T-REX's reward function
    epoch_ppo: int
        at each update period, update ppo for these times
    epoch_reward: int
        training epoch for the T-REX's reward function
    l2_ratio: float
        ratio of the l2 loss in T-REX's loss
    steps_exp: int
        length of demonstrations used while training T-REX's reward function
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
            rollout_length: int = 4096,
            mix_buffer: int = 1,
            batch_size: int = 64,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            lr_reward: float = 3e-4,
            units_actor: tuple = (64, 64),
            units_critic: tuple = (64, 64),
            units_reward: tuple = (256, 256),
            epoch_ppo: int = 10,
            epoch_reward: int = 10000,
            l2_ratio: float = 0.01,
            steps_exp: int = 40,
            clip_eps: float = 0.2,
            lambd: float = 0.95,
            coef_ent: float = 0.0,
            max_grad_norm: float = 0.5
    ):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # expert's buffer
        self.buffer_exp = buffer_exp

        # reward function
        self.reward_func = StateFunction(
            state_shape,
            hidden_units=units_reward,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)

        self.l2_ratio = l2_ratio
        self.learning_steps_reward = 0
        self.optim_reward = Adam(self.reward_func.parameters(), lr=lr_reward)
        self.batch_size = batch_size
        self.epoch_reward = epoch_reward
        self.steps_exp = steps_exp
        self.save_reward_func = False

        # train reward function
        self.train_reward()

    def train_reward(self):
        """# train reward function"""
        print("Training reward function")
        for t in tqdm(range(self.epoch_reward)):
            states_x_exp = torch.Tensor([]).to(self.device)
            states_y_exp = torch.Tensor([]).to(self.device)
            base_label = torch.Tensor([]).to(self.device)
            for i in range(self.batch_size):
                traj_x_states, _, traj_x_rewards, _ = self.buffer_exp.sample_traj(1)
                while traj_x_states[0].shape[0] < self.steps_exp:
                    traj_x_states, _, traj_x_rewards, _ = self.buffer_exp.sample_traj(1)
                traj_y_states, _, traj_y_rewards, _ = self.buffer_exp.sample_traj(1)
                while traj_y_states[0].shape[0] < self.steps_exp:
                    traj_y_states, _, traj_y_rewards, _ = self.buffer_exp.sample_traj(1)
                traj_x_states = traj_x_states[0]
                traj_y_states = traj_y_states[0]
                traj_x_rewards = traj_x_rewards[0]
                traj_y_rewards = traj_y_rewards[0]
                x_ptr = np.random.randint(traj_x_states.shape[0] - self.steps_exp)
                y_ptr = np.random.randint(traj_y_states.shape[0] - self.steps_exp)
                states_x_exp = torch.cat((states_x_exp,
                                          traj_x_states[x_ptr:x_ptr + self.steps_exp].unsqueeze(0)), dim=0)
                states_y_exp = torch.cat((states_y_exp,
                                          traj_y_states[y_ptr:y_ptr + self.steps_exp].unsqueeze(0)), dim=0)

                if torch.sum(traj_x_rewards) > torch.sum(traj_y_rewards):
                    base_label = torch.cat((base_label, torch.zeros(1).to(self.device)), dim=0)
                else:
                    base_label = torch.cat((base_label, torch.ones(1).to(self.device)), dim=0)

            # train reward function
            logits_x_exp = self.reward_func(states_x_exp).sum(1)
            logits_y_exp = self.reward_func(states_y_exp).sum(1)

            # reward function is to tell which trajectory is better by giving rewards based on states
            base_label = base_label.long()
            logits_xy_exp = torch.cat((logits_x_exp, logits_y_exp), dim=1)
            loss_cal = torch.nn.CrossEntropyLoss()
            loss_cmp = loss_cal(logits_xy_exp, base_label)
            loss_l2 = self.l2_ratio * parameters_to_vector(self.reward_func.parameters()).norm() ** 2
            loss_reward = loss_cmp + loss_l2

            self.optim_reward.zero_grad()
            loss_reward.backward()
            self.optim_reward.step()
            self.learning_steps_reward += 1

            if self.learning_steps_reward % 1000 == 0:
                tqdm.write(f'step: {self.learning_steps_reward}, loss: {loss_reward.item():.3f}')
        print("Reward function finished training")

    def eval_reward(self, save_dir: str):
        """
        Plot the true reward and the learned reward

        Parameters
        ----------
        save_dir: str
            path to save
        """
        states, _, rewards, _, _ = self.buffer_exp.get()
        with torch.no_grad():
            rewards_pred = self.reward_func(states)
        rewards_pred = rewards_pred.cpu().detach().numpy().squeeze(1)
        rewards = rewards.cpu().detach().numpy().squeeze(1)
        idxes = np.argsort(rewards)
        plt.figure(0)
        plt.scatter(np.arange(rewards.shape[0]),
                    (rewards_pred[idxes] - rewards_pred.min()) / (rewards_pred.max() - rewards_pred.min()),
                    s=0.1)
        plt.scatter(np.arange(rewards.shape[0]),
                    (rewards[idxes] - rewards.min()) / (rewards.max() - rewards.min()),
                    s=0.1)
        plt.savefig(os.path.join(save_dir, '..', 'reward.png'))
        plt.close()

    def update(self, writer: SummaryWriter):
        """
        Update the algorithm

        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
        self.learning_steps += 1
        states, actions, _, dones, log_pis, next_states = \
            self.buffer.get()
        with torch.no_grad():
            rewards = self.reward_func(states)
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

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
        if not self.save_reward_func:
            torch.save(self.reward_func.state_dict(), f'{save_dir}/../reward_func.pkl')
            self.save_reward_func = True
            self.eval_reward(save_dir)
        torch.save(self.actor.state_dict(), f'{save_dir}/actor.pkl')


class TREXExpert(PPOExpert):
    """
    Well-trained T-REX agent

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
        super(TREXExpert, self).__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            path=path,
            units_actor=units_actor
        )
