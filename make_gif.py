import argparse
import os
import torch
import numpy as np
import imageio

from mujoco_py import GlfwContext
from cail.env import make_env
from cail.algo.algo import EXP_ALGOS


def make_gif(args):
    env = make_env(args.env_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo = EXP_ALGOS[args.algo](
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        path=args.weight
    )

    # setup seeds
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # mkdir
    if not os.path.exists(args.gif_path):
        os.mkdir(args.gif_path)
    if not os.path.exists(os.path.join(args.gif_path, args.env_id)):
        os.mkdir(os.path.join(args.gif_path, args.env_id))

    # set up engine
    GlfwContext(offscreen=True)
    frames = []

    total_return = 0.0
    num_episodes = 0
    num_steps = []

    state = env.reset()
    t = 0
    episode_return = 0.0
    episode_steps = 0

    while num_episodes < args.episodes:
        t += 1

        action = algo.exploit(state)
        next_state, reward, done, _ = env.step(action)
        episode_return += reward
        episode_steps += 1
        state = next_state

        frames.append(env.render(mode='rgb_array', width=250, height=200))

        if done or t == env.max_episode_steps:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0
            num_steps.append(episode_steps)
            episode_steps = 0

    training_steps = args.weight.split("/")[-1].split('.')[0]
    with imageio.get_writer(os.path.join(args.gif_path, args.env_id, f'{args.algo}_{training_steps}.gif'),
                            mode='I', fps=32) as writer:
        for frame in frames:
            writer.append_data(frame)

    mean_return = total_return / num_episodes
    print(f'Mean return of the policy is {mean_return}')
    print(f'Max episode steps is {np.max(num_steps)}')
    print(f'Min episode steps is {np.min(num_steps)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # required
    p.add_argument('--weight', type=str, required=True,
                   help='path to the well-trained weights of the agent')
    p.add_argument('--env-id', type=str, required=True,
                   help='name of the environment')
    p.add_argument('--algo', type=str, required=True,
                   help='name of the well-trained agent')

    # custom
    p.add_argument('--render', action='store_true', default=False,
                   help='render the environment or not')
    p.add_argument('--gif-path', type=str, default='./gifs',
                   help='path to save gifs')

    # default
    p.add_argument('--seed', type=int, default=0,
                   help='random seed')
    p.add_argument('--episodes', type=int, default=5,
                   help='number of episodes used in making gifs')

    args = p.parse_args()
    make_gif(args)
