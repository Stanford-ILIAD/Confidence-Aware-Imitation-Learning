import numpy as np
import torch
import torch.nn as nn
import time

from tqdm import tqdm
from .buffer import Buffer
from .algo.base import Expert
from .env import NormalizedEnv


def soft_update(target, source, tau):
    """Soft update for SAC"""
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network: nn.Module):
    """Disable the gradients of parameters in the network"""
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    """Add random noise to the action"""
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def collect_demo(
        env: NormalizedEnv,
        algo: Expert,
        buffer_size: int,
        device: torch.device,
        std: float,
        p_rand: float,
        seed: int = 0
):
    """
    Collect demonstrations using the well-trained policy

    Parameters
    ----------
    env: NormalizedEnv
        environment to collect demonstrations
    algo: Expert
        well-trained algorithm used to collect demonstrations
    buffer_size: int
        size of the buffer, also the number of s-a pairs in the demonstrations
    device: torch.device
        cpu or cuda
    std: float
        standard deviation add to the policy
    p_rand: float
        with probability of p_rand, the policy will act randomly
    seed: int
        random seed

    Returns
    -------
    buffer: Buffer
        buffer of demonstrations
    mean_return: float
        average episode reward
    """
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_steps = []
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0
    episode_steps = 0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.exploit(state)
            action = add_random_noise(action, std)

        next_state, reward, done, _ = env.step(action)
        mask = True if t == env.max_episode_steps else done
        buffer.append(state, action, reward, mask, next_state)
        episode_return += reward
        episode_steps += 1

        if done or t == env.max_episode_steps:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0
            num_steps.append(episode_steps)
            episode_steps = 0

        state = next_state

    mean_return = total_return / num_episodes
    print(f'Mean return of the expert is {mean_return}')
    print(f'Max episode steps is {np.max(num_steps)}')
    print(f'Min episode steps is {np.min(num_steps)}')

    return buffer, mean_return


def evaluation(
        env: NormalizedEnv,
        algo: Expert,
        episodes: int,
        render: bool,
        seed: int = 0,
        delay: float = 0.03
):
    """
    Evaluate the well-trained policy

    Parameters
    ----------
    env: NormalizedEnv
        environment to evaluate the policy
    algo: Expert
        well-trained policy to be evaluated
    episodes: int
        number of episodes used in evaluation
    render: bool
        render the environment or not
    seed: int
        random seed
    delay: float
        number of seconds to delay while rendering, in case the agent moves too fast

    Returns
    -------
    mean_return: float
        average episode reward
    """
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    total_return = 0.0
    num_episodes = 0
    num_steps = []

    state = env.reset()
    t = 0
    episode_return = 0.0
    episode_steps = 0

    while num_episodes < episodes:
        t += 1

        action = algo.exploit(state)
        next_state, reward, done, _ = env.step(action)
        episode_return += reward
        episode_steps += 1
        state = next_state

        if render:
            env.render()
            time.sleep(delay)

        if done or t == env.max_episode_steps:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0
            num_steps.append(episode_steps)
            episode_steps = 0

    mean_return = total_return / num_episodes
    print(f'Mean return of the policy is {mean_return}')
    print(f'Max episode steps is {np.max(num_steps)}')
    print(f'Min episode steps is {np.min(num_steps)}')

    return mean_return
