import os
import torch
import argparse

from cail.buffer import Buffer, SerializedBuffer
from cail.env import make_env


def mix_demo(folder: str, env_id: str):
    """
    Create a mixture of demonstrations based on demonstrations in the folder

    Parameters
    ----------
    folder: str
        folder containing demos to be mixed
    env_id: str
        name of the environment
    """
    size = []
    buffer_name = []
    files = os.listdir(folder)
    for file in sorted(files):
        buffer_name.append(os.path.join(folder, file))
        size.append(int(file.split('_')[0].split('size')[1]))

    device = torch.device("cpu")
    env = make_env(env_id)

    output_buffer = Buffer(
        buffer_size=sum(size),
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    buffers = []
    for i_buffer, name in enumerate(buffer_name):
        buffers.append(
            SerializedBuffer(
                path=name,
                device=device
            )
        )
        states, actions, rewards, dones, next_states = buffers[i_buffer].get()
        for i_demo in range(size[i_buffer]):
            output_buffer.append(
                states[i_demo].numpy(),
                actions[i_demo].numpy(),
                rewards[i_demo].numpy(),
                dones[i_demo].numpy(),
                next_states[i_demo].numpy()
            )

    rewards_name = ''
    for name in buffer_name:
        mean_reward = name.split('reward')[1].split('.pth')[0]
        rewards_name = rewards_name + '_' + mean_reward

    if os.path.exists(os.path.join(
        'buffers',
        env_id,
        f'size{sum(size)}_reward{rewards_name}.pth'
    )):
        print('Error: demonstrations with the same reward exists')
    else:
        output_buffer.save(os.path.join(
            'buffers',
            env_id,
            f'size{sum(size)}_reward{rewards_name}.pth'
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument('--folder', type=str, required=True,
                        help='folder containing demos to be mixed')
    parser.add_argument('--env-id', type=str, required=True,
                        help='name of the environment')

    args = parser.parse_args()
    mix_demo(folder=args.folder, env_id=args.env_id)
