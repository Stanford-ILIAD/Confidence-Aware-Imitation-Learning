import os
import numpy as np
import torch
import random
import pickle

from typing import Tuple
from .algo.base import Expert
from .env import NormalizedEnv


class SerializedBuffer:
    """
    Serialized buffer, containing [states, actions, rewards, done, next_states]
     and trajectories, often used as demonstrations

    Parameters
    ----------
    path: str
        path to the saved buffer
    device: torch.device
        cpu or cuda
    label_ratio: float
        ratio of labeled data
    sparse_sample: bool
        if true, sample the buffer with the largest gap.
        Often used when the buffer is not shuffled
    use_mean: bool
        if true, use the mean reward of the trajectory s-a pairs to sort trajectories
    """
    def __init__(
            self,
            path: str,
            device: torch.device,
            label_ratio: float = 1,
            sparse_sample: bool = True,
            use_mean: bool = False
    ):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

        self.traj_states = []
        self.traj_actions = []
        self.traj_rewards = []
        self.traj_next_states = []

        all_traj_states = []
        all_traj_actions = []
        all_traj_rewards = []
        all_traj_next_states = []

        self.n_traj = 0
        traj_states = torch.Tensor([]).to(self.device)
        traj_actions = torch.Tensor([]).to(self.device)
        traj_rewards = 0
        traj_next_states = torch.Tensor([]).to(self.device)
        traj_length = 0
        for i, done in enumerate(self.dones):
            traj_states = torch.cat((traj_states, self.states[i].unsqueeze(0)), dim=0)
            traj_actions = torch.cat((traj_actions, self.actions[i].unsqueeze(0)), dim=0)
            traj_rewards += self.rewards[i]
            traj_next_states = torch.cat((traj_next_states, self.next_states[i].unsqueeze(0)), dim=0)
            traj_length += 1
            if done == 1:
                all_traj_states.append(traj_states)
                all_traj_actions.append(traj_actions)
                if use_mean:
                    all_traj_rewards.append(traj_rewards / traj_length)
                else:
                    all_traj_rewards.append(traj_rewards)
                all_traj_next_states.append(traj_next_states)
                traj_states = torch.Tensor([]).to(self.device)
                traj_actions = torch.Tensor([]).to(self.device)
                traj_rewards = 0
                traj_next_states = torch.Tensor([]).to(self.device)
                self.n_traj += 1
                traj_length = 0

        i_traj = random.sample(range(self.n_traj), int(label_ratio * self.n_traj))
        n_labeled_traj = int(label_ratio * self.n_traj)
        if sparse_sample:
            i_traj = [i * int(self.n_traj / n_labeled_traj) for i in range(n_labeled_traj)]
        self.n_traj = n_labeled_traj
        self.label_ratio = label_ratio

        for i in i_traj:
            self.traj_states.append(all_traj_states[i])
            self.traj_actions.append(all_traj_actions[i])
            self.traj_rewards.append(all_traj_rewards[i])
            self.traj_next_states.append(all_traj_next_states[i])

    def sample(
            self,
            batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample data from the buffer

        Parameters
        ----------
        batch_size: int
            batch size

        Returns
        -------
        states: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        next_states: torch.Tensor
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all data in the buffer

        Returns
        -------
        states: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        next_states: torch.Tensor
        """
        return (
            self.states,
            self.actions,
            self.rewards,
            self.dones,
            self.next_states
        )

    def sample_traj(self, batch_size: int) -> Tuple[list, list, list, list]:
        """
        Sample trajectories from the buffer

        Parameters
        ----------
        batch_size: int
            number of trajectories in a batch

        Returns
        -------
        sample_states: a list of torch.Tensor
            each tensor is the states in one trajectory
        sample_actions: a list of torch.Tensor
            each tensor is the actions in one trajectory
        sample_rewards: a list of torch.Tensor
            each tensor is the rewards in one trajectory
        sample_next_states: a list of torch.Tensor
            each tensor is the next_states in one trajectory
        """
        idxes = np.random.randint(low=0, high=self.n_traj, size=batch_size)
        sample_states = []
        sample_actions = []
        sample_rewards = []
        sample_next_states = []

        for i in idxes:
            sample_states.append(self.traj_states[i])
            sample_actions.append(self.traj_actions[i])
            sample_rewards.append(self.traj_rewards[i])
            sample_next_states.append(self.traj_next_states[i])

        return (
            sample_states,
            sample_actions,
            sample_rewards,
            sample_next_states
        )


class Buffer(SerializedBuffer):
    """
    Buffer used while collecting demonstrations

    Parameters
    ----------
    buffer_size: int
        size of the buffer
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    """
    def __init__(
            self,
            buffer_size: int,
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device
    ):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state: np.array, action: np.array, reward: float, done: bool, next_state: np.array):
        """
        Save a transition in the buffer

        Parameters
        ----------
        state: np.array
            current state
        action: np.array
            action taken in the state
        reward: float
            reward of the s-a pair
        done: bool
            whether the state is the end of the episode
        next_state: np.array
            next states that the s-a pair transferred to
        """
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path: str):
        """
        Save the buffer

        Parameters
        ----------
        path: str
            path to save
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)


class RolloutBuffer:
    """
    Rollout buffer that often used in training RL agents

    Parameters
    ----------
    buffer_size: int
        size of the buffer
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    mix: int
        the buffer will be mixed using these time of data
    """
    def __init__(
            self,
            buffer_size: int,
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device,
            mix: int = 1
    ):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.device = device
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(
            self,
            state: np.array,
            action: np.array,
            reward: float,
            done: bool,
            log_pi: float,
            next_state: np.array
    ):
        """
        Save a transition in the buffer

        Parameters
        ----------
        state: np.array
            current state
        action: np.array
            action taken in the state
        reward: float
            reward of the s-a pair
        done: bool
            whether the state is the end of the episode
        log_pi: float
            log(\pi(a|s))
        next_state: np.array
            next states that the s-a pair transferred to
        """
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all data in the buffer

        Returns
        -------
        states: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        log_pis: torch.Tensor
        next_states: torch.Tensor
        """
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(
            self,
            batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample data from the buffer

        Parameters
        ----------
        batch_size: int
            batch size

        Returns
        -------
        states: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        log_pis: torch.Tensor
        next_states: torch.Tensor
        """
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample_traj(self, batch_size: int) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Sample trajectories from the buffer

        Parameters
        ----------
        batch_size: int
            number of trajectories in a batch

        Returns
        -------
        sample_states: an array of torch.Tensor
            each tensor is the states in one trajectory
        sample_actions: an array of torch.Tensor
            each tensor is the actions in one trajectory
        sample_rewards: an array of torch.Tensor
            each tensor is the rewards in one trajectory
        sample_next_states: an array of torch.Tensor
            each tensor is the next_states in one trajectory
        """
        assert self._p % self.buffer_size == 0

        n_traj = 0
        all_traj_states = []
        all_traj_actions = []
        all_traj_next_states = []
        all_traj_rewards = []
        traj_states = torch.Tensor([]).to(self.device)
        traj_actions = torch.Tensor([]).to(self.device)
        traj_next_states = torch.Tensor([]).to(self.device)
        traj_rewards = 0
        for i, done in enumerate(self.dones):
            traj_states = torch.cat((traj_states, self.states[i].unsqueeze(0)), dim=0)
            traj_actions = torch.cat((traj_actions, self.actions[i].unsqueeze(0)), dim=0)
            traj_next_states = torch.cat((traj_next_states, self.next_states[i].unsqueeze(0)), dim=0)
            traj_rewards += self.rewards[i]
            if done == 1:
                all_traj_states.append(traj_states)
                all_traj_actions.append(traj_actions)
                all_traj_next_states.append(traj_next_states)
                all_traj_rewards.append(traj_rewards)
                traj_states = torch.Tensor([]).to(self.device)
                traj_actions = torch.Tensor([]).to(self.device)
                traj_next_states = torch.Tensor([]).to(self.device)
                traj_rewards = 0
                n_traj += 1

        idxes = np.random.randint(low=0, high=n_traj, size=batch_size)
        return (
            np.array(all_traj_states)[idxes],
            np.array(all_traj_actions)[idxes],
            np.array(all_traj_rewards)[idxes],
            np.array(all_traj_next_states)[idxes]
        )


class NoisePreferenceBuffer:
    """
    synthetic dataset by injecting noise in the actor, used in SSRR

    Parameters
    ----------
    env: NormalizedEnv
        environment to collect data
    actor: Expert
        one of the algorithms in algo package
    device: torch.device
        cpu or cuda
    reward_func: callable
        learned reward function
    max_steps: int
        maximum steps in a slice
    min_margin: float
        minimum margin between two samples
    """
    def __init__(
            self,
            env: NormalizedEnv,
            actor: Expert,
            device: torch.device,
            reward_func: callable = None,
            max_steps: int = 100,
            min_margin: float = 0
    ):
        self.env = env
        self.actor = actor
        self.device = device
        self.reward_func = reward_func
        self.max_steps = max_steps
        self.min_margin = min_margin
        self.trajs = []
        self.noise_reward = None

    def get_noisy_traj(self, noise_level: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get one noisy trajectory

        Parameters
        ----------
        noise_level: float
            noise to inject

        Returns
        -------
        states: torch.Tensor
            states in the noisy trajectory
        action: torch.Tensor
            actions in the noisy trajectory
        rewards: torch.Tensor
            rewards of the s-a pairs
        next_states: torch.Tensor
            next states the agent transferred to
        """
        states, actions, rewards, next_states = [], [], [], []

        state = self.env.reset()
        t = 0
        while True:
            t += 1
            if np.random.rand() < noise_level:
                action = self.env.action_space.sample()
            else:
                action = self.actor.exploit(state)
            next_state, reward, done, _ = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            if done or t >= self.env.max_episode_steps:
                break
            state = next_state

        return (
            torch.tensor(states, dtype=torch.float, device=self.device),
            torch.tensor(actions, dtype=torch.float, device=self.device),
            torch.tensor(rewards, dtype=torch.float, device=self.device),
            torch.tensor(next_states, dtype=torch.float, device=self.device),
        )

    def build(self, noise_range: np.array, n_trajs: int):
        """
        Build noisy buffer

        Parameters
        ----------
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
                states, actions, rewards, next_states = self.get_noisy_traj(noise_level)
                reward_traj += rewards.sum()

                # if given reward function, use that instead of the ground truth
                if self.reward_func is not None:
                    rewards = self.reward_func(states)

                agent_trajs.append((states, actions, rewards, next_states))
            self.trajs.append((noise_level, agent_trajs))
            reward_traj /= n_trajs
            print(f'Noise level: {noise_level:.3f}, traj reward: {reward_traj:.3f}')
        print('Collecting finished')

    def get(self) -> list:
        """
        Get all the trajectories

        Returns
        -------
        trajs: list, all trajectories
        """
        return self.trajs

    def get_noise_reward(self):
        """
        Get rewards and the corresponding noise level

        Returns
        -------
        noise_reward: list
            each element is an array with (noise_level, reward)
        """
        if self.noise_reward is None:
            self.noise_reward = []

            prev_noise = 0.0
            noise_reward = 0
            n_traj = 0
            for traj in self.trajs:
                for agent_traj in traj[1]:
                    if prev_noise == traj[0]:  # noise level has not changed
                        noise_reward += agent_traj[2].mean()
                        n_traj += 1
                        prev_noise = traj[0]
                    else:  # noise level changed
                        self.noise_reward.append([prev_noise, noise_reward / n_traj])
                        prev_noise = traj[0]
                        noise_reward = agent_traj[2].mean()
                        n_traj = 1
            self.noise_reward = np.array(self.noise_reward, dtype=np.float)
        return self.noise_reward

    def sample(self, n_sample: int):
        """
        Sample data from the buffer

        Parameters
        ----------
        n_sample: int
            number of samples

        Returns
        -------
        data: list
            each element contains (noise level, (states, actions, rewards) in the trajectory)
        """
        data = []
        for _ in range(n_sample):
            noise_idx = np.random.choice(len(self.trajs))
            traj = self.trajs[noise_idx][1][np.random.choice(len(self.trajs[noise_idx][1]))]
            if len(traj[0]) > self.max_steps:
                ptr = np.random.randint(len(traj[0]) - self.max_steps)
                x_slice = slice(ptr, ptr + self.max_steps)
            else:
                x_slice = slice(len(traj[0]))
            states, actions, rewards, _ = traj
            data.append((self.trajs[noise_idx][0], (states[x_slice], actions[x_slice], rewards[x_slice])))
        return data

    def save(self, save_dir: str):
        """
        Save the buffer

        Parameters
        ----------
        save_dir: str
            path to save

        Returns
        -------

        """
        with open(f'{save_dir}/noisy_trajs.pkl', 'wb') as f:
            pickle.dump(self.trajs, f)
