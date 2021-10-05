import os
import argparse
import torch

from datetime import datetime
from cail.env import make_env
from cail.algo.algo import ALGOS
from cail.trainer import Trainer


def run(args):
    """Train experts using PPO or SAC"""
    env = make_env(args.env_id)
    env_test = env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo = ALGOS[args.algo](
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        seed=args.seed,
        rollout_length=args.rollout
    )

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # required
    p.add_argument('--env-id', type=str, required=True,
                   help='name of the environment')

    # custom
    p.add_argument('--algo', type=str, default='ppo',
                   help='algorithm used, currently support ppo | sac')
    p.add_argument('-n', '--num-steps', type=int, default=5*10**6,
                   help='number of steps to train')
    p.add_argument('--eval-interval', type=int, default=10**4,
                   help='time interval between evaluations')

    # default
    p.add_argument('--seed', type=int, default=0,
                   help='random seed')
    p.add_argument('--rollout', type=int, default=50000,
                   help='rollout length of the buffer')

    args = p.parse_args()
    run(args)
