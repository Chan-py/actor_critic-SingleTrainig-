from dataclasses import dataclass
from typing import Any, Optional, Union, Dict
import sys
from time import time

from einops import rearrange
import numpy as np
import torch
from torch.multiprocessing import Pool
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import cv2

from utils import compute_lambda_returns, LossWithIntermediateLosses
from parallel_env import ParallelEnv

Batch = Dict[str, torch.Tensor]

@dataclass
class ActorCriticOutput:
    logits_actions: torch.FloatTensor
    means_values: torch.FloatTensor


@dataclass
class RolloutOutput:
    observations: torch.ByteTensor
    actions: torch.LongTensor
    logits_actions: torch.FloatTensor
    values: torch.FloatTensor
    rewards: torch.FloatTensor
    ends: torch.BoolTensor


class ActorCritic(nn.Module):
    def __init__(self, act_vocab_size=4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm_dim = 512
        self.lstm = nn.LSTMCell(1024, self.lstm_dim)
        self.hx, self.cx = None, None

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, act_vocab_size)

    def __repr__(self) -> str:
        return "actor_critic"

    def clear(self) -> None:
        self.hx, self.cx = None, None

    def reset(self, n: int) -> None:
        device = self.conv1.weight.device
        self.hx = torch.zeros(n, self.lstm_dim, device=device)
        self.cx = torch.zeros(n, self.lstm_dim, device=device)

    def forward(self, inputs: torch.FloatTensor) -> ActorCriticOutput:
        assert inputs.ndim == 4 and inputs.shape[1:] == (3, 64, 64)
        assert 0 <= inputs.min() <= 1 and 0 <= inputs.max() <= 1

        x = inputs

        x = x.mul(2).sub(1)
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)

        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))

        logits_actions = rearrange(self.actor_linear(self.hx), 'b a -> b 1 a')
        means_values = rearrange(self.critic_linear(self.hx), 'b 1 -> b 1 1')

        return ActorCriticOutput(logits_actions, means_values)

    def compute_loss(self, envs, gamma: float = 0.99, lambda_: float = 0.95, entropy_weight: float = 0.001, **kwargs: Any) -> LossWithIntermediateLosses:
        outputs = self.batch_rollout(envs)
        rewards_total = np.mean(torch.sum(outputs.rewards, dim=1).detach().cpu().numpy())

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.values,
                ends=outputs.ends,
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]

        values = outputs.values[:, :-1]

        d = Categorical(logits=outputs.logits_actions[:, :-1])
        log_probs = d.log_prob(outputs.actions[:, :-1])
        loss_actions = -1 * (log_probs * (lambda_returns - values.detach())).mean()
        loss_entropy = - entropy_weight * d.entropy().mean()
        loss_values = F.mse_loss(values, lambda_returns)

        return (LossWithIntermediateLosses(loss_actions=loss_actions, loss_values=loss_values, loss_entropy=loss_entropy), rewards_total)

    def batch_rollout(self, envs) -> RolloutOutput:
        device = self.conv1.weight.device
        n = envs.n

        self.reset(n=n)

        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []

        envs.reset()
        obss, _, dones = envs.step(np.zeros(n))

        for _ in range(119):
            if np.any(dones):
                envs.reset()
            obss = torch.FloatTensor(obss).to(device)
            all_observations.append(obss)
            outputs_ac = self(obss)
            action_token = Categorical(logits=outputs_ac.logits_actions).sample()
            all_actions.append(action_token)
            all_logits_actions.append(outputs_ac.logits_actions)
            all_values.append(outputs_ac.means_values)
            obss, step_rewards, dones = envs.step(action_token.squeeze(1))
            all_rewards.append(torch.tensor(step_rewards).reshape(-1, 1))
            all_ends.append(torch.tensor(dones).reshape(-1, 1))

        if np.any(dones):
            envs.reset()

        self.clear()

        return RolloutOutput(
            observations=torch.stack(all_observations, dim=1).mul(255).byte(),      # (B, T, C, H, W) in [0, 255]
            actions=torch.cat(all_actions, dim=1),                                  # (B, T)
            logits_actions=torch.cat(all_logits_actions, dim=1),                    # (B, T, #actions)
            values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),         # (B, T)
            rewards=torch.cat(all_rewards, dim=1).to(device),                       # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),                             # (B, T)
        )

if __name__ == '__main__':
    import pickle
    import os
    from matplotlib import pyplot as plt

    alg = ActorCritic()
    if os.path.exists('alg.pt'):
        alg.load_state_dict(torch.load('alg.pt'))
    
    optimizer = torch.optim.Adam(alg.parameters(), lr=0.0001)
    if os.path.exists('optimizer.pt'):
        optimizer.load_state_dict(torch.load('optimizer.pt'))

    num_envs = 16
    envs = ParallelEnv(num_envs)

    n_epoch = 10000
    max_grad_norm = 10

    # loss_actions_all = []
    # loss_values_all = []
    # loss_entropies = []
    loss_totals = []
    rewards_all = []

    if os.path.exists('losses.pkl'):
        loss_totals, rewards_all = pickle.load(open('losses.pkl', 'rb'))


    for epoch in tqdm(range(n_epoch)):
        optimizer.zero_grad()

        losses, rewards = alg.compute_loss(envs)
        loss_total_step = losses.loss_total
        loss_total_step.backward()

        loss_totals.append(loss_total_step.item())
        # loss_values_all.append(losses["loss_values"].item())
        # loss_entropies.append(losses["loss_entropy"].item())
        # loss_actions_all.append(losses["loss_actions"].item())
        rewards_all.append(rewards)

        # pickle.dump([loss_totals, loss_values_all, loss_entropies, loss_actions_all, rewards_all], open('losses.pkl', 'wb'))
        pickle.dump([loss_totals, rewards_all], open('losses.pkl', 'wb'))
        print('###########', epoch, rewards)

        torch.nn.utils.clip_grad_norm_(alg.parameters(), max_grad_norm)

        optimizer.step()

        if epoch != 0 and epoch % 10 == 0:
            torch.save(alg.state_dict(), 'alg.pt')
            torch.save(optimizer.state_dict(), 'optimizer.pt')

            # plt.plot(loss_totals)
            # plt.savefig('loss_totals.png')
            # plt.close()

            plt.plot(rewards_all)
            plt.savefig('rewards_all.png')
            plt.close()
