# -*- coding: utf-8 -*-
import gym
from tqdm import tqdm
import sys
import os
from utils_tools.common import log2json, visualize_result, dirs_creat, seed_torch
from utils_tools.TD3 import TD3
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore")

MAX_TIMESTEP = 2000
MAX_EPISODE = 300
TRAINABLE = True
LOGGER_KEYS = ['mode', 'epochs', 'timestep', 'ep_reward', 'reward', 'action']


torch.set_printoptions(sci_mode=False)

if __name__ == '__main__':

    env = gym.make('Pendulum-v1')
    env = env.unwrapped
    # env.seed(1)
    test_train_flag = TRAINABLE

    action_shape = np.array((env.action_space.low.item(), env.action_space.high.item()), dtype='float')
    action_dim = env.action_space.shape[0]
    state_shape = env.observation_space.shape[0]          # [1., 1., 1.]  ~  [-1.,  0.,  0.]

    # 初始化td3算法
    td3 = TD3(action_space=action_shape, state_shape=state_shape,
              action_dim=action_dim)

    # 初始化logger
    logger_json = log2json()
    logger_csv = log2json(type_json=False, logger_keys=LOGGER_KEYS)

    count = 0
    ep_history = []

    ep_tqdm = tqdm(range(MAX_EPISODE), position=0, leave=False, colour='green')
    for epoch in ep_tqdm:
        obs = env.reset()
        obs = obs.reshape(1, 3)
        ep_rh = 0
        td3.ep += 1

        ep_step_tqdm = tqdm(range(MAX_TIMESTEP), position=1, leave=False, colour='red')
        for t in ep_step_tqdm:
            env.render()
            action = td3.get_action(obs)
            obs_t1, reward, done, _ = env.step(action.detach().numpy().reshape(1, 1))
            obs_t1 = obs_t1.reshape(1, 3)
            reward = (reward + 16) / 16
            td3.state_store_memory(obs, action.detach().numpy().reshape(1, 1), reward, obs_t1, float(done))

            if td3.trainable and td3.start_train < len(td3.memory):
                td3.update()

            td3.t += 1
            obs = obs_t1
            ep_rh += reward

            ep_step_tqdm.set_description(f'epochs: {td3.ep:.4f}, ep_reward: {ep_rh.item():.4f}, '
                                         f'action: {action.detach().numpy().reshape(1, 1).item():.4f}, '
                                         f'reward: {reward.item():.4f}')

            # save to json log file
            log_tag = {'mode': 'train',
                       'epochs': td3.ep,
                       'timestep': td3.t,
                       'ep_reward': ep_rh.item(),
                       'reward': reward.item(),
                       'action': action.detach().numpy().item()}

            logger_json.write2json(log_tag)
            logger_csv.write2csv(log_tag)

        ep_history.append(ep_rh)

    ep_history = np.array(ep_history)
    plt.plot(np.arange(ep_history.shape[0]), ep_history)
    plt.show()
    env.close()
