import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

from torch import nn
from torch.optim import Adam
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import curve_fit
from tqdm import tqdm
from .ppo import PPO, PPOExpert
from .airl import AIRLExpert, AIRLReward
from cail.buffer import NoisePreferenceBuffer, SerializedBuffer
from cail.network import StateFunction
from cail.env import NormalizedEnv


class SSRR(PPO):
    """
    Implementation of SSRR

    Reference:
    [1] Chen, L., Paleja, R., and Gombolay, M.
    Learning from suboptimal demonstration via self-supervised reward regression.
    In Conference on Robot Learning. PMLR, 2020.

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
    airl_actor_path: str
        path to the trained AIRL actor
    airl_discriminator_path: str
        path to the trained AIRL discriminator
    gamma: float
        discount factor
    rollout_length: int
        rollout length of the buffer
    mix_buffer: int
        times for rollout buffer to mix
    batch_size_reward: int
        batch size for training SSRR's reward function
    lr_actor: float
        learning rate of the actor
    lr_critic: float
        learning rate of the critic
    lr_reward: float
        learning rate of the D-REX's reward function
    units_actor: tuple
        hidden units of the actor
    units_critic: tuple
        hidden units of the critic
    units_reward: tuple
        hidden units of the D-REX's reward function
    epoch_ppo: int
        at each update period, update ppo for these times
    epoch_reward: int
        training epoch for the SSRR's reward function
    l2_ratio_reward: float
        ratio of the l2 loss in D-REX's loss
    n_reward_model: int
        number of reward functions to train
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
            airl_actor_path: str,
            airl_discriminator_path: str,
            gamma: float = 0.995,
            rollout_length: int = 10000,
            mix_buffer: int = 1,
            batch_size_reward: int = 64,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            lr_reward: float = 1e-4,
            units_actor: tuple = (64, 64),
            units_critic: tuple = (64, 64),
            units_reward: tuple = (256, 256),
            epoch_ppo: int = 10,
            epoch_reward: int = 20000,
            l2_ratio_reward: float = 0.01,
            n_reward_model: int = 3,
            noise_range: np.array = np.arange(0., 1., 0.05),
            n_demo_traj: int = 5,
            steps_exp: int = 50,
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

        # use the given AIRL model
        self.airl_actor = AIRLExpert(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            path=airl_actor_path,
        )
        self.airl_reward = AIRLReward(
            state_shape=state_shape,
            device=device,
            path=airl_discriminator_path,
        )
        print('AIRL loaded')
        self.save_airl = False

        # noisy buffer
        self.noisy_buffer = NoisePreferenceBuffer(
            env=env,
            actor=self.airl_actor,
            device=self.device,
            reward_func=self.airl_reward.get_reward,
            max_steps=steps_exp,
        )
        self.env = env
        self.noisy_buffer.build(noise_range=noise_range, n_trajs=n_demo_traj)
        print('Noisy buffer built')
        self.save_noisy_buffer = False

        # fit noise-performance relationship using sigmoid function
        self.sigma = None
        self.fit_noise_performance()

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
        self.loss_reward = nn.MSELoss()
        self.train_reward()

    def eval_airl(self, save_dir: str):
        """
        Plot the true reward and the AIRL's learned reward

        Parameters
        ----------
        save_dir: str
            path to save
        """
        states, actions, rewards, _, _ = self.buffer_exp.get()
        with torch.no_grad():
            rewards_pred = self.airl_reward.get_reward(states)
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
        plt.savefig(os.path.join(save_dir, '..', 'reward_airl.png'))
        plt.close()

    def fit_noise_performance(self):
        """Fit the noise-performance curve"""
        print('Fitting noise-performance')

        # sigmoid curve
        def sigmoid(eff, x):
            x0, y0, c, k = eff
            y = c / (1 + np.exp(-k * (x - x0))) + y0
            return y

        def residuals(eff, x, y):
            return y - sigmoid(eff, x)

        # fit the curve
        noise_reward_data = self.noisy_buffer.get_noise_reward()
        label = noise_reward_data[:, 1]
        label_scale = label.max() - label.min()
        label_intercept = label.min()
        label = (label - label_intercept) / label_scale
        noises = noise_reward_data[:, 0]

        eff_guess = np.array([np.median(noises), np.median(label), 1.0, -1.0])
        eff_fit, cov, infodict, mesg, ier = scipy.optimize.leastsq(
            residuals, eff_guess, args=(noises, label), full_output=True)

        def fitted_sigma(x):
            return sigmoid(eff_fit, x) * label_scale + label_intercept

        self.sigma = fitted_sigma

    def eval_noise_performance(self, save_dir: str):
        """
        Plot the figure of noise-performance curve

        Parameters
        ----------
        save_dir: str
            path to save
        """
        noise_reward_data = self.noisy_buffer.get_noise_reward()

        x = np.linspace(0, 1, 100)
        y = self.sigma(x)
        plt.figure()
        plt.plot(x, y)
        plt.scatter(noise_reward_data[:, 0], noise_reward_data[:, 1], s=0.5)
        plt.savefig(os.path.join(save_dir, '..', 'sigmoid.png'))
        plt.close()

    def train_reward(self):
        """Train SSRR's reward function"""
        print('Training reward function')
        for i in range(len(self.reward_funcs)):
            print(f'Reward function: {i}')
            self.learning_steps_reward = 0
            for it in tqdm(range(self.epoch_reward)):
                self.learning_steps_reward += 1

                # load data
                data = self.noisy_buffer.sample(self.batch_size_reward)
                noises, trajs = zip(*data)
                noises = np.asarray(noises)
                states, actions, _ = zip(*trajs)
                lengths = [x.shape[0] for x in states]
                states = torch.cat([x for x in states], dim=0)
                rewards = self.reward_funcs[i](states).squeeze(0)

                # calculate traj rewards, using AIRL's reward function
                traj_rewards = torch.tensor([], dtype=torch.float, device=self.device, requires_grad=True)
                ptr = 0
                for length in lengths:
                    traj_rewards = torch.cat((traj_rewards, rewards[ptr: ptr + length].sum().unsqueeze(0)))
                    ptr += length

                # target reward is sigma(eta)
                targets = torch.tensor(self.sigma(noises) * np.array(lengths) / self.env.max_episode_steps,
                                       dtype=torch.float, device=self.device)

                # update
                loss_cmp = self.loss_reward(traj_rewards, targets)
                loss_l2 = self.l2_ratio_reward * parameters_to_vector(self.reward_funcs[i].parameters()).norm() ** 2
                loss = loss_cmp + loss_l2
                self.reward_optims[i].zero_grad()
                loss.backward()
                self.reward_optims[i].step()

                if self.learning_steps_reward % 1000 == 0:
                    tqdm.write(f'step: {self.learning_steps_reward}, loss: {loss.item():.3f}')
        print("Reward function finished training")

    def eval_reward(self, save_dir: str):
        """
        Plot the true reward and the learned reward

        Parameters
        ----------
        save_dir: str
            path to save
        """
        states, actions, rewards, _, _ = self.buffer_exp.get()
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
        plt.savefig(os.path.join(save_dir, '..', 'reward_exp.png'))
        plt.close()

        data = self.noisy_buffer.sample(len(self.noisy_buffer.trajs))
        noises, trajs = zip(*data)
        noises = np.asarray(noises)
        states, actions, rewards_airl = zip(*trajs)
        lengths = [x.shape[0] for x in states]
        states = torch.cat([x for x in states], dim=0)
        # actions = torch.cat([x for x in actions], dim=0)
        rewards_airl = torch.cat([x for x in rewards_airl], dim=0)
        rewards_pred = 0
        with torch.no_grad():
            for i in range(len(self.reward_funcs)):
                rewards_pred += self.reward_funcs[i](states) / len(self.reward_funcs)

        traj_rewards_pred = torch.tensor([], dtype=torch.float, device=self.device)
        traj_rewards_airl = torch.tensor([], dtype=torch.float, device=self.device)
        traj_rewards = torch.tensor([], dtype=torch.float, device=self.device)
        ptr = 0
        for length in lengths:
            traj_rewards_pred = torch.cat((traj_rewards_pred, rewards_pred[ptr: ptr + length].sum().unsqueeze(0)))
            traj_rewards_airl = torch.cat((traj_rewards_airl, rewards_airl[ptr: ptr + length].sum().unsqueeze(0)))
            traj_rewards = torch.cat(
                (traj_rewards,
                 torch.tensor(rewards[ptr: ptr + length], dtype=torch.float, device=self.device).sum().unsqueeze(0)))
            ptr += length
        traj_rewards_airl = traj_rewards_airl.cpu().detach().numpy()
        traj_rewards_pred = traj_rewards_pred.cpu().detach().numpy()
        traj_rewards = traj_rewards.cpu().detach().numpy()
        idxes = np.argsort(traj_rewards)
        plt.figure(1)
        plt.scatter(np.arange(traj_rewards.shape[0]),
                    (traj_rewards[idxes] - traj_rewards.min()) / (traj_rewards.max() - traj_rewards.min()),
                    s=1, label='true')
        plt.scatter(np.arange(traj_rewards.shape[0]),
                    (traj_rewards_airl[idxes] - traj_rewards_airl.min()) / (
                            traj_rewards_airl.max() - traj_rewards_airl.min()),
                    s=1, label='airl')
        plt.scatter(np.arange(traj_rewards.shape[0]),
                    (traj_rewards_pred[idxes] - traj_rewards_pred.min()) / (
                                traj_rewards_pred.max() - traj_rewards_pred.min()),
                    s=1, label='learned')
        plt.legend()
        plt.savefig(os.path.join(save_dir, '..', 'reward_noise.png'))
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
            self.eval_airl(save_dir)
            self.eval_noise_performance(save_dir)
        if not self.save_airl:
            torch.save(self.airl_actor.actor.state_dict(), f'{save_dir}/../airl_actor.pkl')
            torch.save(self.airl_reward.disc.state_dict(), f'{save_dir}/../airl_disc.pkl')
            self.save_airl = True
        if not self.save_noisy_buffer:
            self.noisy_buffer.save(f'{save_dir}/..')
            self.save_noisy_buffer = True
        torch.save(self.actor.state_dict(), f'{save_dir}/actor.pkl')


class SSRRExpert(PPOExpert):
    """
    Well-trained SSRR agent

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
        super(SSRRExpert, self).__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            path=path,
            units_actor=units_actor
        )
