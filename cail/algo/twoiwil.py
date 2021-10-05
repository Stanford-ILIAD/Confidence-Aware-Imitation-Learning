import torch
import os
import torch.nn.functional as F
import numpy as np
import copy

from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple
from .ppo import PPO, PPOExpert
from .utils import CULoss
from cail.network import AIRLDiscrim, Classifier
from cail.buffer import SerializedBuffer


class TwoIWIL(PPO):
    """
    Implementation of 2IWIL, using PPO-based AIRL as the backbone IL algorithm

    Reference:
    ----------
    [1] Wu, Y.-H., Charoenphakdee, N., Bao, H., Tangkaratt, V.,and Sugiyama, M.
    Imitation learning from imperfect demonstration.
    In International Conference on MachineLearning, pp. 6818–6827, 2019.

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
    gamma: float
        discount factor
    rollout_length: int
        rollout length of the buffer
    mix_buffer: int
        times for rollout buffer to mix
    batch_size: int
        batch size for sampling from current policy and demonstrations
    lr_actor: float
        learning rate of the actor
    lr_critic: float
        learning rate of the critic
    lr_disc: float
        learning rate of the discriminator
    units_actor: tuple
        hidden units of the actor
    units_critic: tuple
        hidden units of the critic
    units_disc_r: tuple
        hidden units of the discriminator r
    units_disc_v: tuple
        hidden units of the discriminator v
    epoch_ppo: int
        at each update period, update ppo for these times
    epoch_disc: int
        at each update period, update the discriminator for these times
    clip_eps: float
        clip coefficient in PPO's objective
    lambd: float
        lambd factor
    coef_ent: float
        entropy coefficient
    max_grad_norm: float
        maximum gradient norm
    classifier_iter: int
        iteration of training the classifier
    lr_classifier: float
        learning rate of the classifier
    """
    def __init__(
            self,
            buffer_exp: SerializedBuffer,
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device,
            seed: int,
            gamma: float = 0.995,
            rollout_length: int = 10000,
            mix_buffer: int = 1,
            batch_size: int = 64,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            lr_disc: float = 3e-4,
            units_actor: tuple = (64, 64),
            units_critic: tuple = (64, 64),
            units_disc_r: tuple = (100, 100),
            units_disc_v: tuple = (100, 100),
            epoch_ppo: int = 50,
            epoch_disc: int = 10,
            clip_eps: float = 0.2,
            lambd: float = 0.97,
            coef_ent: float = 0.0,
            max_grad_norm: float = 10.0,
            classifier_iter: int = 25000,
            lr_classifier: float = 3e-4
    ):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # expert's buffer
        self.buffer_exp = buffer_exp

        # discriminator
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

        # classifier
        self.classifier = Classifier(state_shape, action_shape).to(device)
        self.n_label_traj = self.buffer_exp.n_traj
        self.classifier_iter = classifier_iter
        self.optim_classifier = Adam(self.classifier.parameters(), lr=lr_classifier)
        self.train_classifier()
        self.save_classifier = False

        # label conf
        states_exp, action_exp, _, _, _ = self.buffer_exp.get()
        self.conf = torch.sigmoid(self.classifier(torch.cat((states_exp, action_exp), dim=-1)))

    def train_classifier(self):
        """Train a classifier"""
        print('Training classifier')
        label_traj_states = copy.deepcopy(self.buffer_exp.traj_states)
        label_traj_actions = copy.deepcopy(self.buffer_exp.traj_actions)
        label_traj_rewards = copy.deepcopy(self.buffer_exp.traj_rewards)

        # use ranking to label confidence
        conf_gap = 1.0 / float(self.n_label_traj - 1)
        ranking = np.argsort(label_traj_rewards)
        traj_lengths = np.asarray([i.shape[0] for i in label_traj_states])
        n_label_demos = traj_lengths.sum()
        label = np.zeros(n_label_demos)
        ptr = 0
        for i in range(traj_lengths.shape[0]):
            label[ptr: ptr + traj_lengths[i]] = ranking[i] * conf_gap
            ptr += traj_lengths[i]
        label = torch.from_numpy(label).to(self.device)

        label_traj = torch.cat((torch.cat(label_traj_states), torch.cat(label_traj_actions)), dim=-1)
        batch = min(128, label_traj.shape[0])
        ubatch = int(batch / label_traj.shape[0] * self.buffer_exp.buffer_size)
        loss_fun = CULoss(label, beta=1 - self.buffer_exp.label_ratio, device=self.device, non=True)

        # start training
        for i_iter in tqdm(range(self.classifier_iter)):
            idx = np.random.choice(label_traj.shape[0], batch)
            labeled = self.classifier(Variable(label_traj[idx, :]))
            smp_conf = label[idx]

            states_exp, actions_exp, _, _, _ = self.buffer_exp.sample(ubatch)
            unlabeled = self.classifier(torch.cat((states_exp, actions_exp), dim=-1))

            self.optim_classifier.zero_grad()
            risk = loss_fun(smp_conf, labeled, unlabeled)

            risk.backward()
            self.optim_classifier.step()

            if i_iter % 2000 == 0:
                tqdm.write(f'iteration: {i_iter}\tcu loss: {risk.data.item():.3f}')

        self.classifier = self.classifier.eval()
        print("Classifier finished training")

    def sample_exp(
            self,
            batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample from expert's demonstrations

        Parameters
        ----------
        batch_size: int
            number of samples

        Returns
        -------
        states: torch.Tensor
            expert's states
        actions: torch.Tensor
            expert's actions
        dones: torch.Tensor
            expert's dones
        next_states: torch.Tensor
            expert's next states
        conf: torch.Tensor
            confidence of expert's demonstrations
        """
        # Samples from expert's demonstrations.
        all_states_exp, all_actions_exp, _, all_dones_exp, all_next_states_exp = \
            self.buffer_exp.get()
        all_conf = Variable(self.conf)
        all_conf_mean = Variable(all_conf.mean())
        conf = all_conf / all_conf_mean
        with torch.no_grad():
            self.conf = conf
        idxes = np.random.randint(low=0, high=all_states_exp.shape[0], size=batch_size)
        return (
            all_states_exp[idxes],
            all_actions_exp[idxes],
            all_dones_exp[idxes],
            all_next_states_exp[idxes],
            self.conf[idxes]
        )

    def update(self, writer: SummaryWriter):
        """
        Update the algorithm

        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # samples from current policy's trajectories
            states, _, _, dones, log_pis, next_states = self.buffer.sample(self.batch_size)

            # samples from expert's demonstrations
            states_exp, actions_exp, dones_exp, next_states_exp, conf = self.sample_exp(self.batch_size)

            # calculate log probabilities of expert actions
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(states_exp, actions_exp)

            # update discriminator
            self.update_disc(
                states, dones, log_pis, next_states, states_exp,
                dones_exp, log_pis_exp, next_states_exp, conf, writer
            )

        # we don't use reward signals here
        states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # calculate rewards
        rewards = self.disc.calculate_reward(
            states, dones, log_pis, next_states)

        # update PPO using estimated rewards
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_disc(
            self,
            states: torch.Tensor,
            dones: torch.Tensor,
            log_pis: torch.Tensor,
            next_states: torch.Tensor,
            states_exp: torch.Tensor,
            dones_exp: torch.Tensor,
            log_pis_exp: torch.Tensor,
            next_states_exp: torch.Tensor,
            conf: torch.Tensor,
            writer: SummaryWriter
    ):
        """
        Update the discriminator

        Parameters
        ----------
        states: torch.Tensor
            states sampled from current IL policy
        dones: torch.Tensor
            dones sampled from current IL policy
        log_pis: torch.Tensor
            log(\pi(s|a)) sampled from current IL policy
        next_states: torch.Tensor
            next states sampled from current IL policy
        states_exp: torch.Tensor
            states sampled from demonstrations
        dones_exp: torch.Tensor
            dones sampled from demonstrations
        log_pis_exp: torch.Tensor
            log(\pi(s|a)) sampled from demonstrations
        next_states_exp: torch.Tensor
            next states sampled from demonstrations
        conf: torch.Tensor
            learned confidence of the demonstration samples
        writer: SummaryWriter
            writer for logs
        """
        # output of discriminator is (-inf, inf), not [0, 1]
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(states_exp, dones_exp, log_pis_exp, next_states_exp)

        # discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)]
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -(F.logsigmoid(logits_exp).mul(conf)).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # discriminator's accuracies
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)

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
        torch.save(self.disc.state_dict(), f'{save_dir}/disc.pkl')
        torch.save(self.actor.state_dict(), f'{save_dir}/actor.pkl')
        if not self.save_classifier:
            torch.save(self.classifier.state_dict(), f'{save_dir}/../classifier.pkl')
            self.save_classifier = True


class TwoIWILExpert(PPOExpert):
    """
    Well-trained 2IWIL agent

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
        super(TwoIWILExpert, self).__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            path=path,
            units_actor=units_actor
        )
