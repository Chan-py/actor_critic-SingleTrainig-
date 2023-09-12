import numpy as np
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.multiprocessing import Process, Pipe
import cv2
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs):
        self.num_envs = num_envs

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        #logger.warn('Render not defined for %s' % self)
        pass
        
    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

action_dict = {0:0, 1:1, 2:3, 3:4}

def worker(remote, parent_remote, worker_id):
    parent_remote.close()

    env_name="envs/v15/Window_build"

    seed = np.random.randint(100000000)

    env = UnityEnvironment(env_name, seed=seed, worker_id=worker_id)
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            data = data.item()
            assert data in np.arange(4)

            # data = action_dict[data]
            a = np.random.randint(0, 4)

            action_tuple0 = ActionTuple(np.array([[]], dtype=np.float32), np.array([[data + 1]], dtype=np.int32))
            action_tuple1 = ActionTuple(np.array([[]], dtype=np.float32), np.array([[a+1]], dtype=np.int32))

            env.set_action_for_agent(behavior_name, 0, action_tuple0)
            env.set_action_for_agent(behavior_name, 1, action_tuple1)

            env.step()

            decision_steps = env.get_steps(behavior_name)
            if len(decision_steps[1].reward):
                done = True
                decision_steps = decision_steps[1]
            else:
                done = False
                decision_steps = decision_steps[0]

            obs = decision_steps.obs[0][0]
            obs = cv2.resize(obs, dsize=(64, 64))
            obs = torch.FloatTensor(obs)
            obs = obs.permute(2, 0, 1)
            reward = decision_steps.reward[0]

            remote.send((obs, reward, done))
        elif cmd == 'reset':
            env.reset()
            remote.send(None)

        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError


class ParallelEnv(VecEnv):
    def __init__(self, n=4):
        """
        envs: list of mlagents_envs environments to run in subprocesses
        adopted from openai baseline
        """
        self.n = n
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n)])
        self.ps = [Process(target=worker, args=(work_remote, remote, worker_id)) for worker_id, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes))]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        VecEnv.__init__(self, n)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

def rollout(num_envs, num_steps):
    envs = ParallelEnv(num_envs)
    rewards_total = []

    envs.reset()

    for epoch in range(3):

        epoch_rewards = []
        for _ in range(num_steps):
            actions = torch.randint(0, 7, size=(num_envs,))
            next_states, step_rewards, dones = envs.step(actions)
            epoch_rewards.append(step_rewards)
            if np.any(dones):
                assert np.all(dones)
                envs.reset()
                break
        rewards_total.append(epoch_rewards)

    # envs.close()

    return rewards_total


if __name__ == '__main__':
    num_envs = 4
    num_steps = 100

    rewards = rollout(num_envs, num_steps)
