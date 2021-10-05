import argparse
import torch

from cail.env import make_env
from cail.algo.algo import EXP_ALGOS
from cail.utils import evaluation


def run(args):
    env = make_env(args.env_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo = EXP_ALGOS[args.algo](
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        path=args.weight
    )

    mean_return = evaluation(
        env=env,
        algo=algo,
        episodes=args.episodes,
        render=args.render,
        seed=args.seed,
        delay=args.delay
    )


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

    # default
    p.add_argument('--episodes', type=int, default=10,
                   help='number of episodes used in evaluation')
    p.add_argument('--seed', type=int, default=0,
                   help='random seed')
    p.add_argument('--delay', type=float, default=0,
                   help='number of seconds to delay while rendering, in case the agent moves too fast')

    args = p.parse_args()
    run(args)
