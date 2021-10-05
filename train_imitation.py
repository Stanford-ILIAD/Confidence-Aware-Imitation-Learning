import os
import argparse
import torch
import numpy as np

from datetime import datetime
from cail.env import make_env
from cail.buffer import SerializedBuffer
from cail.algo.algo import ALGOS
from cail.trainer import Trainer


def run(args):
    """Train Imitation Learning algorithms"""
    env = make_env(args.env_id)
    env_test = env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=device,
        label_ratio=args.label,
        use_mean=args.use_transition
    )

    if args.algo == 'cail':
        algo = ALGOS[args.algo](
            buffer_exp=buffer_exp,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=device,
            seed=args.seed,
            rollout_length=args.rollout_length,
            lr_conf=args.lr_conf,
            pretrain_steps=args.pre_train,
            use_transition=args.use_transition
        )
    elif args.algo == 'drex':
        algo = ALGOS[args.algo](
            buffer_exp=buffer_exp,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=device,
            seed=args.seed,
            rollout_length=args.rollout_length,
            env=env
        )
    elif args.algo == 'ssrr':
        algo = ALGOS[args.algo](
            buffer_exp=buffer_exp,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=device,
            seed=args.seed,
            rollout_length=args.rollout_length,
            env=env,
            airl_actor_path=args.airl_actor,
            airl_discriminator_path=args.airl_disc,
        )
    else:
        algo = ALGOS[args.algo](
            buffer_exp=buffer_exp,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=device,
            seed=args.seed,
            rollout_length=args.rollout_length,
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
        num_eval_episodes=args.num_eval_epi,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # required
    p.add_argument('--buffer', type=str, required=True,
                   help='path to the demonstration buffer')
    p.add_argument('--env-id', type=str, required=True,
                   help='name of the environment')
    p.add_argument('--algo', type=str, required=True,
                   help='Imitation Learning algorithm to be trained')

    # custom
    p.add_argument('--rollout-length', type=int, default=10000,
                   help='rollout length of the buffer')
    p.add_argument('--num-steps', type=int, default=10**6,
                   help='number of steps to train')
    p.add_argument('--eval-interval', type=int, default=10**4,
                   help='time interval between evaluations')

    # for CAIL
    p.add_argument('--lr-conf', type=float, default=0.1,
                   help='learning rate of confidence for CAIL')
    p.add_argument('--pre-train', type=int, default=20000000,
                   help='pre-train steps for CAIL')
    p.add_argument('--use-transition', action='store_true', default=False,
                   help='use state transition reward for cail')

    # for SSRR
    p.add_argument('--airl-actor', type=str,
                   help='path to pre-trained AIRL actor for SSRR')
    p.add_argument('--airl-disc', type=str,
                   help='path to pre-trained AIRL discriminator for SSRR')

    # default
    p.add_argument('--num-eval-epi', type=int, default=5,
                   help='number of episodes for evaluation')
    p.add_argument('--seed', type=int, default=0,
                   help='random seed')
    p.add_argument('--label', type=float, default=0.05,
                   help='ratio of labeled data')

    args = p.parse_args()
    run(args)
