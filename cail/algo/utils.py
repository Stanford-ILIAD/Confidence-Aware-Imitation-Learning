import torch
import torch.nn as nn
import pickle
import numpy as np

from typing import Tuple, List
from cail.env import NormalizedEnv


def calculate_gae(
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor,
        gamma: float,
        lambd: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate generalized advantage estimator

    Parameters
    ----------
    values: torch.Tensor
        values of the states
    rewards: torch.Tensor
        rewards given by the reward function
    dones: torch.Tensor
        if this state is the end of the episode
    next_values: torch.Tensor
        values of the next states
    gamma: float
        discount factor
    lambd: float
        lambd factor

    Returns
    -------
    advantages: torch.Tensor
        advantages
    gaes: torch.Tensor
        normalized gae
    """
    # calculate TD errors
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # initialize gae
    gaes = torch.empty_like(rewards)

    # calculate gae recursively from behind
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.shape[0] - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class CULoss(nn.Module):
    """
    CU-Loss defined in 2IWIL

    Reference:
    ----------
    [1] Wu, Y.-H., Charoenphakdee, N., Bao, H., Tangkaratt, V.,and Sugiyama, M.
    Imitation learning from imperfect demonstration.
    In International Conference on MachineLearning, pp. 6818–6827, 2019.

    Parameters
    ----------
    conf: torch.Tensor
        confidence
    beta: float
        ratio of non-labeled data
    device: torch.device
        cpu or cuda
    non: bool
        clip the loss or not
    """
    def __init__(
            self,
            conf: torch.Tensor,
            beta: float,
            device: torch.device,
            non: bool = False
    ):
        super(CULoss, self).__init__()
        self.loss = nn.SoftMarginLoss()
        self.beta = beta
        self.non = non
        self.device = device
        if conf.mean() > 0.5:
            self.UP = True
        else:
            self.UP = False

    def forward(
            self,
            conf: torch.Tensor,
            labeled: torch.Tensor,
            unlabeled: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the CU-Loss

        Parameters
        ----------
        conf: torch.Tensor
            confidence
        labeled: torch.Tensor
            classifier's output of the labeled data
        unlabeled: torch.Tensor
        classifier's output of the unlabeled data

        Returns
        -------
        objective: torch.Tensor
            CU-Loss
        """
        y_conf_pos = self.loss(labeled, torch.ones(labeled.shape).to(self.device))
        y_conf_neg = self.loss(labeled, -torch.ones(labeled.shape).to(self.device))

        if self.UP:
            # conf_risk = torch.mean((1-conf) * (y_conf_neg - y_conf_pos) + (1 - self.beta) * y_conf_pos)
            unlabeled_risk = torch.mean(self.beta * self.loss(unlabeled, torch.ones(unlabeled.shape).to(self.device)))
            neg_risk = torch.mean((1 - conf) * y_conf_neg)
            pos_risk = torch.mean((conf - self.beta) * y_conf_pos) + unlabeled_risk
        else:
            # conf_risk = torch.mean(conf * (y_conf_pos - y_conf_neg) + (1 - self.beta) * y_conf_neg)
            unlabeled_risk = torch.mean(self.beta * self.loss(unlabeled, -torch.ones(unlabeled.shape).to(self.device)))
            pos_risk = torch.mean(conf * y_conf_pos)
            neg_risk = torch.mean((1 - self.beta - conf) * y_conf_neg) + unlabeled_risk
        if self.non:
            objective = torch.clamp(neg_risk, min=0) + torch.clamp(pos_risk, min=0)
        else:
            objective = neg_risk + pos_risk
        return objective


class NoisePreferenceDataset:
    """
    Synthetic dataset by injecting noise in the bc policy

    Parameters
    ----------
    env: NormalizedEnv
        environment to collect data
    device: torch.device
        cpu or cuda
    max_steps: int
        maximum steps in a slice
    min_margin: float
        minimum margin between two samples
    """
    def __init__(
            self,
            env: NormalizedEnv,
            device: torch.device,
            max_steps: int = None,
            min_margin: float = None
    ):
        self.env = env
        self.device = device
        self.max_steps = max_steps
        self.min_margin = min_margin
        self.trajs = []

    def get_noisy_traj(self, actor: nn.Module, noise_level: float) -> Tuple[np.array, np.array, np.array]:
        """
        Get one noisy trajectory

        Parameters
        ----------
        actor: nn.Module
            policy network
        noise_level: float
            noise to inject

        Returns
        -------
        states: np.array
            states in the noisy trajectory
        action: np.array
            actions in the noisy trajectory
        rewards: np.array
            rewards of the s-a pairs
        """
        states, actions, rewards = [], [], []

        state = self.env.reset()
        t = 0
        while True:
            t += 1
            if np.random.rand() < noise_level:
                action = self.env.action_space.sample()
            else:
                action = actor(torch.from_numpy(state).unsqueeze(0).float().to(self.device)).cpu().detach().numpy()[0]
            next_state, reward, done, _ = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            if done or t >= self.env.max_episode_steps:
                break
            state = next_state

        return np.array(states), np.array(actions), np.array(rewards)

    def build(self, actor: nn.Module, noise_range: np.array, n_trajs: int):
        """
        Build noisy dataset

        Parameters
        ----------
        actor: nn.Module
            policy network
        noise_range: np.array
            range of noise
        n_trajs: int
             number of trajectories
        """
        print('Collecting noisy demonstrations')
        for noise_level in noise_range:
            agent_trajs = []
            reward_traj = 0
            for i_traj in range(n_trajs):
                states, actions, rewards = self.get_noisy_traj(actor, noise_level)
                agent_trajs.append((states, actions, rewards))
                reward_traj += rewards.sum()
            self.trajs.append((noise_level, agent_trajs))
            reward_traj /= n_trajs
            print(f'Noise level: {noise_level:.3f}, traj reward: {reward_traj:.3f}')
        print('Collecting finished')

    def sample(self, n_sample: int) -> List:
        """
        Sample from the data set

        Parameters
        ----------
        n_sample: int
            number of samples

        Returns
        -------
        data: List
            list of trajectories, each element contains:
                1.) trajectory 1
                2.) trajectory 2
                3.) whether trajectory 1's reward is larger than trajectory 2
        """
        data = []

        for _ in range(n_sample):
            # pick two noise level set
            x_idx, y_idx = np.random.choice(len(self.trajs), 2, replace=False)
            while abs(self.trajs[x_idx][0] - self.trajs[y_idx][0]) < self.min_margin:
                x_idx, y_idx = np.random.choice(len(self.trajs), 2, replace=False)

            # pick trajectory from each set
            x_traj = self.trajs[x_idx][1][np.random.choice(len(self.trajs[x_idx][1]))]
            y_traj = self.trajs[y_idx][1][np.random.choice(len(self.trajs[y_idx][1]))]

            # sub-sampling from a trajectory
            if len(x_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(x_traj[0]) - self.max_steps)
                x_slice = slice(ptr, ptr + self.max_steps)
            else:
                x_slice = slice(len(x_traj[0]))

            if len(y_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(y_traj[0]) - self.max_steps)
                y_slice = slice(ptr, ptr + self.max_steps)
            else:
                y_slice = slice(len(y_traj[0]))

            # done
            data.append(
                (x_traj[0][x_slice],
                 y_traj[0][y_slice],
                 0 if self.trajs[x_idx][0] < self.trajs[y_idx][0] else 1)
            )

        return data

    def save(self, save_dir: str):
        """
        Save the dataset

        Parameters
        ----------
        save_dir: str
            path to save
        """
        with open(f'{save_dir}/noisy_trajs.pkl', 'wb') as f:
            pickle.dump(self.trajs, f)
