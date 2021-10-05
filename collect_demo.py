import os
import argparse
import torch

from cail.env import make_env
from cail.algo.algo import EXP_ALGOS
from cail.utils import collect_demo


def run(args):
    """Collect demonstrations"""
    env = make_env(args.env_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo = EXP_ALGOS[args.algo](
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        path=args.weight
    )

    buffer, mean_return = collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=device,
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed
    )

    if os.path.exists(os.path.join(
        'buffers/Raw',
        args.env_id,
        f'size{args.buffer_size}_reward{round(mean_return, 2)}.pth'
    )):
        print('Error: demonstrations with the same reward exists')
    else:
        buffer.save(os.path.join(
            'buffers/Raw',
            args.env_id,
            f'size{args.buffer_size}_reward{round(mean_return, 2)}.pth'
        ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # required
    p.add_argument('--weight', type=str, required=True,
                   help='path to the well-trained weights of the agent')
    p.add_argument('--env-id', type=str, required=True,
                   help='name of the environment')
    p.add_argument('--algo', type=str, required=True,
                   help='name of the well-trained agent')

    # default
    p.add_argument('--buffer-size', type=int, default=40000,
                   help='size of the buffer')
    p.add_argument('--std', type=float, default=0.01,
                   help='standard deviation add to the policy')
    p.add_argument('--p-rand', type=float, default=0.0,
                   help='with probability of p_rand, the policy will act randomly')
    p.add_argument('--seed', type=int, default=0,
                   help='random seed')

    args = p.parse_args()
    run(args)
