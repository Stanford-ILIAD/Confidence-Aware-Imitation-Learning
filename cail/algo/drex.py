import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Adam
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .base import Expert
from .ppo import PPO, PPOExpert
from .utils import NoisePreferenceDataset
from cail.network import StateFunction, DeterministicPolicy
from cail.utils import disable_gradient
from cail.buffer import SerializedBuffer
from cail.env import NormalizedEnv


class DREX(PPO):
    """
    Implementation of D-REX

    Reference:
    ----------
    [1] Brown,  D.  S.,  Goo,  W.,  and  Niekum,  S.
    Better-than-demonstrator imitation learning via automatically-ranked demonstrations.
    In Conference on Robot Learning, pp.330–359. PMLR, 2020.

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
    env: NormalizedEnv
        environment
    gamma: float
        discount factor
    rollout_length: int
        rollout length of the buffer
    mix_buffer: int
        times for rollout buffer to mix
    batch_size_bc: int
        batch size for training Behavior Cloning
    batch_size_reward: int
        batch size for training D-REX's reward function
    size_reward_dataset: int
        size of the dataset for training D-REX's reward function
    n_reward_model: int
        number of reward functions to train
    lr_actor: float
        learning rate of the actor
    lr_critic: float
        learning rate of the critic
    lr_bc: float
        learning rate of the Behavior Cloning
    lr_reward: float
        learning rate of the D-REX's reward function
    units_actor: tuple
        hidden units of the actor
    units_critic: tuple
        hidden units of the critic
    units_bc: tuple
        hidden units of the Behavior Cloning
    units_reward: tuple
        hidden units of the D-REX's reward function
    epoch_ppo: int
        at each update period, update ppo for these times
    epoch_bc: int
        training epoch for Behavior Cloning
    epoch_reward: int
        training epoch for the D-REX's reward function
    l2_ratio_bc: float
        ratio of the l2 loss in Behavior Cloning's loss
    l2_ratio_reward: float
        ratio of the l2 loss in D-REX's loss
    noise_range: np.array
        range of noise injected to the BC policy
    n_demo_traj: int
        number of demonstration trajectories in the noise dataset
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
            env: NormalizedEnv,
            gamma: float = 0.995,
            rollout_length: int = 4096,
            mix_buffer: int = 1,
            batch_size_bc: int = 128,
            batch_size_reward: int = 64,
            size_reward_dataset: int = 5000,
            n_reward_model: int = 3,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            lr_bc: float = 1e-3,
            lr_reward: float = 1e-4,
            units_actor: tuple = (64, 64),
            units_critic: tuple = (64, 64),
            units_bc: tuple = (256, 256, 256),
            units_reward: tuple = (256, 256),
            epoch_ppo: int = 10,
            epoch_bc: int = 50000,
            epoch_reward: int = 10000,
            l2_ratio_bc: float = 1e-3,
            l2_ratio_reward: float = 0.01,
            noise_range: np.array = np.arange(0., 1.0, 0.05),
            n_demo_traj: int = 5,
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

        # synthetic noisy dataset
        self.noisy_dataset = NoisePreferenceDataset(
            env=env,
            device=device,
            max_steps=50,
            min_margin=0.3,
        )
        self.env = env

        # expert's buffer
        self.buffer_exp = buffer_exp

        # BC policy
        self.bc = DeterministicPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_bc,
            hidden_activation=nn.ReLU(inplace=True),
        ).to(device)
        self.optim_bc = Adam(self.bc.parameters(), lr=lr_bc)
        self.loss_bc = nn.MSELoss()
        self.batch_size_bc = batch_size_bc
        self.epoch_bc = epoch_bc
        self.l2_ratio_bc = l2_ratio_bc
        self.learning_steps_bc = 0
        self.save_bc = False

        # reward function
        self.reward_funcs = []
        self.reward_optims = []
        for i in range(n_reward_model):
            self.reward_funcs.append(StateFunction(
                state_shape=state_shape,
                hidden_units=units_reward,
                hidden_activation=nn.ReLU(inplace=True),
            ).to(device))
            self.reward_optims.append(Adam(self.reward_funcs[i].parameters(), lr=lr_reward))
        self.batch_size_reward = batch_size_reward
        self.epoch_reward = epoch_reward
        self.l2_ratio_reward = l2_ratio_reward
        self.learning_steps_reward = 0
        self.save_reward_func = False
        self.steps_exp = steps_exp

        # train bc
        self.train_bc()
        self.eval_bc(episodes=10)

        # build synthetic noisy dataset
        self.noisy_dataset.build(
            actor=self.bc,
            noise_range=noise_range,
            n_trajs=n_demo_traj,
        )
        self.save_noisy_dataset = False

        # train reward function
        self.train_reward(size_reward_dataset)

    def train_bc(self):
        """Train Behavior Cloning using the demonstrations"""
        print("Training behavior cloning")
        for t_bc in tqdm(range(self.epoch_bc)):
            states, actions, _, _, _ = self.buffer_exp.sample(self.batch_size_bc)
            loss = self.loss_bc(self.bc(states), actions) + \
                   self.l2_ratio_bc * parameters_to_vector(self.bc.parameters()).norm() ** 2
            self.optim_bc.zero_grad()
            loss.backward()
            self.optim_bc.step()
            self.learning_steps_bc += 1

            if self.learning_steps_bc % 5000 == 0:
                tqdm.write(f'step: {self.learning_steps_bc}, loss: {loss.item():.3f}')
        print("BC finished training")

    def eval_bc(self, episodes: int = 10):
        """
        Evaluate the BC policy

        Parameters
        ----------
        episodes: int
            number of episodes to evaluate
        """
        state = self.env.reset()
        rewards = []
        for _ in range(episodes):
            t = 0
            reward_traj = 0
            while True:
                t += 1
                action = self.bc(torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0))
                next_state, reward, done, _ = self.env.step(action.cpu().detach().numpy()[0])
                reward_traj += reward
                if done or t >= self.env.max_episode_steps:
                    state = self.env.reset()
                    rewards.append(reward_traj)
                    break
                state = next_state
        print(f'BC reward: {np.mean(rewards)}+/-{np.std(rewards)}')

    def train_reward(self, size_reward_dataset: int):
        """
        Train D-REX's reward function

        Parameters
        ----------
        size_reward_dataset: int
            size of the noise dataset
        """
        print("Training reward function")

        # train each reward function
        for i in range(len(self.reward_funcs)):
            print(f'Reward function: {i}')
            self.learning_steps_reward = 0

            # load data
            data = self.noisy_dataset.sample(size_reward_dataset)

            idxes = np.random.permutation(len(data))
            train_idxes = idxes[:int(len(data) * 0.8)]
            valid_idxes = idxes[int(len(data) * 0.8):]

            def _load(idx_list, add_noise=True):
                if len(idx_list) > self.batch_size_reward:
                    idx = np.random.choice(idx_list, self.batch_size_reward, replace=False)
                else:
                    idx = idx_list

                batch = []
                for j in idx:
                    batch.append(data[j])

                b_x, b_y, b_l = zip(*batch)
                x_split = np.array([len(x) for x in b_x], dtype=int)
                y_split = np.array([len(y) for y in b_y], dtype=int)
                b_x, b_y, b_l = np.concatenate(b_x, axis=0), np.concatenate(b_y, axis=0), np.array(b_l)

                if add_noise:
                    b_l = (b_l + np.random.binomial(1, 0.1, self.batch_size_reward)) % 2  # flip with probability 0.1

                return (
                    torch.tensor(b_x, dtype=torch.float, device=self.device),
                    torch.tensor(b_y, dtype=torch.float, device=self.device),
                    x_split,
                    y_split,
                    torch.tensor(b_l, dtype=torch.float, device=self.device),
                )

            for it in tqdm(range(self.epoch_reward)):
                states_x, states_y, states_x_split, states_y_split, labels = _load(train_idxes, add_noise=True)
                logits_x = self.reward_funcs[i](states_x)
                logits_y = self.reward_funcs[i](states_y)

                ptr_x = 0
                logits_x_split = torch.tensor([], dtype=torch.float, device=self.device, requires_grad=True)
                for i_split in range(states_x_split.shape[0]):
                    logits_x_split = \
                        torch.cat((logits_x_split, logits_x[ptr_x: ptr_x + states_x_split[i_split]].sum().unsqueeze(0)))
                    ptr_x += states_x_split[i_split]
                ptr_y = 0
                logits_y_split = torch.tensor([], dtype=torch.float, device=self.device, requires_grad=True)
                for i_split in range(states_y_split.shape[0]):
                    logits_y_split = \
                        torch.cat((logits_y_split, logits_y[ptr_y: ptr_y + states_y_split[i_split]].sum().unsqueeze(0)))
                    ptr_y += states_y_split[i_split]

                labels = labels.long()
                logits_xy = torch.cat((logits_x_split.unsqueeze(1), logits_y_split.unsqueeze(1)), dim=1)
                loss_cal = torch.nn.CrossEntropyLoss()
                loss_cmp = loss_cal(logits_xy, labels)
                loss_l2 = self.l2_ratio_reward * parameters_to_vector(self.reward_funcs[i].parameters()).norm() ** 2
                loss_reward = loss_cmp + loss_l2
                self.reward_optims[i].zero_grad()
                loss_reward.backward()
                self.reward_optims[i].step()
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
        rewards_pred = 0
        with torch.no_grad():
            for i in range(len(self.reward_funcs)):
                rewards_pred += self.reward_funcs[i](states) / len(self.reward_funcs)
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
        rewards = torch.zeros((states.shape[0], 1)).type_as(states)
        with torch.no_grad():
            for i in range(len(self.reward_funcs)):
                rewards += self.reward_funcs[i](states) / len(self.reward_funcs)
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
            for i in range(len(self.reward_funcs)):
                torch.save(self.reward_funcs[i].state_dict(), f'{save_dir}/../reward_func_{i}.pkl')
            self.save_reward_func = True
            self.eval_reward(save_dir)
        if not self.save_bc:
            torch.save(self.bc.state_dict(), f'{save_dir}/../bc.pkl')
            self.save_bc = True
        if not self.save_noisy_dataset:
            self.noisy_dataset.save(f'{save_dir}/..')
            self.save_noisy_dataset = True
        torch.save(self.actor.state_dict(), f'{save_dir}/actor.pkl')


class DREXExpert(PPOExpert):
    """
    Well-trained D-REX agent

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
        super(DREXExpert, self).__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            path=path,
            units_actor=units_actor
        )


class DREXBCExpert(Expert):
    """
    Well-trained D-REX's BC policy

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
            units_actor: tuple = (256, 256, 256)
    ):
        super().__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
        )
        self.actor = DeterministicPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True),
        ).to(device)
        self.actor.load_state_dict(torch.load(path, map_location=device))
        disable_gradient(self.actor)
