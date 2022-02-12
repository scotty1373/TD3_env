# -*- coding: utf-8 -*-
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
import os
import torch


class log2json:
    def __init__(self):
        pass

    def flush_write(self):
        pass

    def write2file(self):
        pass

    def write2csv(self):
        pass


class visualize_result:
    def __init__(self):
        pass

    def reward(self):
        pass

    def uni_loss(self):
        pass

    def average_value(self):
        pass


def dirs_creat():
    if platform.system() == 'windows':
        temp = os.getcwd()
        CURRENT_PATH = temp.replace('\\', '/')
    else:
        CURRENT_PATH = os.getcwd()
    CURRENT_PATH = os.path.join(CURRENT_PATH, 'save_Model')
    if not os.path.exists(CURRENT_PATH):
        os.makedirs(CURRENT_PATH)


# 设置相同训练种子
def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)         # 为当前CPU 设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        # 为当前的GPU 设置随机种子
        torch.cuda.manual_seed_all(seed)        # 当使用多块GPU 时，均设置随机种子
        torch.backends.cudnn.deterministic = True       # 设置每次返回的卷积算法是一致的
        torch.backends.cudnn.benchmark = True      # cuDNN使用的非确定性算法自动寻找最适合当前配置的高效算法，设置为False
        torch.backends.cudnn.enabled = True     # pytorch使用cuDNN加速