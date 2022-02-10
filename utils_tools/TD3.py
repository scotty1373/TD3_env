# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from model import Model
from collections import deque
from copy import deepcopy

MAX_TIMESTEP = 500
MAX_EPOCH = 300
BATCH_SIZE = 32


class TD3:
    def __init__(self, action_space, state_shape, action_dim):
        self.action_space = action_space
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.t = 0
        # 初始化actor + 双q网络
        self._init(self.state_shape, self.action_dim, [400, 300])
        self.batch_size = BATCH_SIZE
        self.lr_actor = 1e-3
        self.lr_critic = 1e-4
        self.discount_index = 0.95

    def _init(self, state_dim, action_dim, hidden_unit):
        self.actor_model = Model(state_dim, hidden_unit, action_dim)
        self.critic_model1 = Model(state_dim, hidden_unit, action_dim)
        self.critic_model2 = Model(state_dim, hidden_unit, action_dim)
        self.memory = deque(maxlen=24000)

    def state_store_memory(self, state, action, reward, state_t1):
        self.memory.append((state, action, reward, state_t1, self.t))

    def get_action(self, state):
        pass

    def update(self):
        pass

    def action_update(self):
        pass

    def critic_update(self):
        pass

    def load_model(self, file_name):
        checkpoint = torch.load(file_name)
        self.actor_model.load_state_dict(checkpoint['actor'])
        self.critic_model1.load_state_dict(checkpoint['critic1'])
        self.critic_model2.load_state_dict(checkpoint['critic2'])

    def model_hard_update(self, current, target):
        pass

    def model_soft_update(self):
        pass

