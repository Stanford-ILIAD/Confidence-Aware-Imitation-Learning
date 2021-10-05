import torch
import torch.nn.functional as F
import numpy as np
import itertools
import os

from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from .ppo import PPO, PPOExpert
from cail.network import AIRLDiscrim, AIRLDetachedDiscrim
from cail.buffer import SerializedBuffer


class CAIL(PPO):
    """
    Implementation of CAIL, using PPO-based AIRL as the backbone IL
    algorithm and ranking loss as the outer loss

    Reference:
    ----------
    [1] Zhang, S., Cao, Z., Sadigh, D., Sui, Y.
    Confidence-Aware Imitation Learning from Demonstrations with Varying Optimality.
    In Advances in neural information processing systems, 2021.

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
    traj_batch_size: int
        batch size for sampling trajectories to calculate the outer loss
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
    lr_conf: float
        learning rate of confidence
    pretrain_steps: int
        steps for pre-training
    use_transition: bool
        if true, CAIL will use AIRL's f(s,s') as reward function,
        else, CAIL will use AIRL's g(s)
    save_all_conf: bool
        if true, all the confidence will be saved (space consuming),
        else, only the convergent confidence will be saved
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
            batch_size: int = 100,
            traj_batch_size: int = 20,
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
            lr_conf: float = 1e-1,
            pretrain_steps: int = 2000000,
            use_transition: bool = False,
            save_all_conf: bool = False
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
        self.detached_disc = AIRLDetachedDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            device=device,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        )

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.lr_disc = lr_disc
        self.epoch_disc = epoch_disc

        # init confidence
        self.conf = torch.ones(self.buffer_exp.buffer_size, 1).to(device)

        self.learning_steps_conf = 0
        self.lr_conf = lr_conf
        self.epoch_conf = self.epoch_disc

        self.batch_size = batch_size
        self.traj_batch_size = traj_batch_size
        self.pretrain_steps = pretrain_steps
        self.use_transition = use_transition
        self.save_all_conf = save_all_conf

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
        # samples from expert's demonstrations
        all_states_exp, all_actions_exp, _, all_dones_exp, all_next_states_exp = \
            self.buffer_exp.get()
        all_conf = Variable(self.conf)
        all_conf_mean = Variable(all_conf.mean())
        conf = all_conf / all_conf_mean
        conf.clamp_(0, 2)
        with torch.no_grad():
            self.conf = conf
        self.conf.requires_grad = True
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

            # ---update the discriminator for step 1
            # samples from current policy's trajectories
            states, _, _, dones, log_pis, next_states = self.buffer.sample(self.batch_size)

            # samples from expert's demonstrations
            states_exp, actions_exp, dones_exp, next_states_exp, conf = self.sample_exp(self.batch_size)

            # calculate log probabilities of expert actions
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(states_exp, actions_exp)

            # update discriminator (retain grad)
            self.update_disc_retain_grad(
                states, dones, log_pis, next_states, states_exp,
                dones_exp, log_pis_exp, next_states_exp, conf
            )

            # ---update confidence
            self.learning_steps_conf += 1

            # sample trajectories from demonstrations
            states_traj, actions_traj, rewards_traj, next_states_traj \
                = self.buffer_exp.sample_traj(self.traj_batch_size)

            # update conf
            conf_grad = self.update_conf(states_traj, next_states_traj, rewards_traj, writer)

            # ---update the discriminator for step 2
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
                dones_exp, log_pis_exp, next_states_exp, Variable(conf), conf_grad, writer
            )

        # we don't use reward signals here
        states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # calculate rewards
        rewards = self.disc.calculate_reward(states, dones, log_pis, next_states)

        # update PPO using estimated rewards
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states, writer)

    def update_disc_retain_grad(
            self,
            states: torch.Tensor,
            dones: torch.Tensor,
            log_pis: torch.Tensor,
            next_states: torch.Tensor,
            states_exp: torch.Tensor,
            dones_exp: torch.Tensor,
            log_pis_exp: torch.Tensor,
            next_states_exp: torch.Tensor,
            conf: torch.Tensor
    ):
        """
        Pseudo-update the discriminator while retaining the gradients

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
        """
        # output of discriminator is (-inf, inf), not [0, 1]
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(states_exp, dones_exp, log_pis_exp, next_states_exp)

        # discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [\frac{r}{\alpha}log(D)]
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -(F.logsigmoid(logits_exp).mul(conf)).mean()
        loss_disc = loss_pi + loss_exp

        loss_grad = torch.autograd.grad(loss_disc,
                                        self.disc.parameters(),
                                        create_graph=True,
                                        retain_graph=True)
        discLoss_wrt_omega = parameters_to_vector(loss_grad)
        disc_param_vector = parameters_to_vector(self.disc.parameters()).clone().detach()
        disc_param_vector -= self.lr_disc * discLoss_wrt_omega
        self.detached_disc.set_parameters(disc_param_vector)

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
            conf_grad: torch.Tensor,
            writer: SummaryWriter
    ):
        """
        Real update of the discriminator

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
        conf_grad: torch.Tensor
            gradient of the confidence
        writer: SummaryWriter
            writer for logs
        """
        # output of discriminator is (-inf, inf), not [0, 1]
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(states_exp, dones_exp, log_pis_exp, next_states_exp)

        # discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [conf * log(D)]
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -(F.logsigmoid(logits_exp).mul(conf)).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward(create_graph=True)
        disc_grad = torch.autograd.grad(loss_disc, self.disc.parameters())

        disc_grad = parameters_to_vector(disc_grad)
        grad_product = torch.dot(disc_grad, conf_grad)

        # only update discriminator when the angle between two gradients are less than \pi/2
        if grad_product >= 0 or self.learning_steps < self.pretrain_steps:
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

    def update_conf(
            self,
            states_traj: list,
            next_states_traj: list,
            rewards_traj: list,
            writer: SummaryWriter
    ):
        """
        Update the confidence according to the outer loss

        Parameters
        ----------
        states_traj: list
            be list of tensors. Trajectories states sampled from demonstrations.
            Each tensor is a trajectory
        next_states_traj: list
            be list of tensors. Trajectories next_states sampled from demonstrations.
            Each tensor is a trajectory
        rewards_traj: list
            be list of tensors. Trajectories rewards sampled from demonstrations.
            Each tensor is a trajectory
        writer: SummaryWriter
            writer for logs

        Returns
        -------
        conf_grad: torch.Tensor
            gradient of confidence
        """
        learned_rewards_traj = []
        for i in range(len(states_traj)):
            if self.use_transition:
                learned_rewards_traj.append(
                    self.detached_disc.f(
                        states_traj[i],
                        torch.cat((torch.zeros(states_traj[i].shape[0] - 1, 1),
                                   torch.ones(1, 1)), dim=0).to(self.device),
                        next_states_traj[i]
                    ).mean().unsqueeze(0)
                )
            else:
                learned_rewards_traj.append(self.detached_disc.g(states_traj[i]).sum().unsqueeze(0))
        outer_loss = self.ranking_loss(rewards_traj, torch.cat(learned_rewards_traj, dim=0))

        outer_loss.backward()
        with torch.no_grad():
            self.conf -= self.lr_conf * self.conf.grad
        self.conf.requires_grad = True
        self.conf.grad.zero_()

        # check outer loss's gradient w.r.t. \alpha_t
        learned_rewards_traj_t = []
        for i in range(len(states_traj)):
            if self.use_transition:
                learned_rewards_traj_t.append(
                    self.disc.f(
                        states_traj[i],
                        torch.cat((torch.zeros(states_traj[i].shape[0] - 1, 1),
                                   torch.ones(1, 1)), dim=0).to(self.device),
                        next_states_traj[i]
                    ).mean().unsqueeze(0)
                )
            else:
                learned_rewards_traj_t.append(self.disc.g(states_traj[i]).sum().unsqueeze(0))
        outer_loss_t = self.ranking_loss(rewards_traj, torch.cat(learned_rewards_traj_t, dim=0))
        if self.use_transition:
            conf_grad = list(torch.autograd.grad(outer_loss_t, self.disc.parameters()))
        else:
            conf_grad = list(torch.autograd.grad(outer_loss_t, self.disc.g.parameters()))
        conf_grad = parameters_to_vector(conf_grad)
        if not self.use_transition:
            conf_grad = torch.cat((conf_grad, torch.zeros(self.detached_disc.num_param_h()).to(self.device)), dim=-1)

        if self.learning_steps_conf % self.epoch_conf == 0:
            writer.add_scalar(
                'loss/outer', outer_loss.item(), self.learning_steps
            )
        return conf_grad

    def ranking_loss(self, truth: list, approx: torch.Tensor) -> torch.Tensor:
        """
        Calculate the total ranking loss of two list of rewards

        Parameters
        ----------
        truth: list
            ground truth rewards of trajectories
        approx: torch.Tensor
            learned rewards of trajectories

        Returns
        -------
        loss: torch.Tensor
            ranking loss
        """
        margin = 1e-5  # factor to make the loss Lipschitz-smooth

        loss_func = nn.MarginRankingLoss().to(self.device)
        loss = torch.Tensor([0]).to(self.device)

        # loop over all the combinations of the rewards
        for c in itertools.combinations(range(approx.shape[0]), 2):
            if truth[c[0]] > truth[c[1]]:
                if torch.abs(abs(approx[c[0]] - approx[c[1]])) < margin:
                    loss += (1 / (4 * margin)) * (torch.abs(approx[c[0]] - approx[c[1]]) - margin) ** 2
                else:
                    y = torch.Tensor([1]).to(self.device)
                    loss += loss_func(approx[c[0]].unsqueeze(0), approx[c[1]].unsqueeze(0), y)
            else:
                if torch.abs(abs(approx[c[0]] - approx[c[1]])) < margin:
                    loss += (1 / (4 * margin)) * (torch.abs(approx[c[0]] - approx[c[1]]) - margin) ** 2
                else:
                    y = torch.Tensor([-1]).to(self.device)
                    loss += loss_func(approx[c[0]].unsqueeze(0), approx[c[1]].unsqueeze(0), y)
        return loss

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
        # all_states_exp, all_actions_exp, _, _, _ = self.buffer_exp.get()
        all_conf = self.conf
        if self.save_all_conf:
            file = f'{save_dir}/conf.csv'
        else:
            file = f'{save_dir}/../conf.csv'
        with open(file, 'w') as f:
            for i in range(all_conf.shape[0]):
                f.write(f'{all_conf[i].item()}\n')


class CAILExpert(PPOExpert):
    """
    Well-trained CAIL agent

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
        super(CAILExpert, self).__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            path=path,
            units_actor=units_actor
        )
