# -*- coding: utf-8 -*-
import csv
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
import os
import torch

TIMESTAMP = str(round(time.time()))


class log2json:
    def __init__(self, filename='train_log', type_json=True, log_path='log', logger_keys=None):
        self.root_path = os.getcwd()
        self.log_path = os.path.join(self.root_path, log_path, TIMESTAMP)

        assert os.path.exists(os.path.join(self.root_path, log_path))

        # 创建当前训练log保存目录
        try:
            os.makedirs(self.log_path)
        except FileExistsError as e:
            print(e)

        if type_json:
            filename = os.path.join(self.log_path, filename + '.json')
            self.fp = open(filename, 'w')
        else:
            filename = os.path.join(self.log_path, filename + '.csv')
            self.fp = open(filename, 'w', encoding='utf-8', newline='')
            self.csv_writer = csv.writer(self.fp)
            self.csv_keys = logger_keys
            self.csv_writer.writerow(self.csv_keys)

    def flush_write(self, string):
        self.fp.write(string + '\n')
        self.fp.flush()

    def write2json(self, log_dict):
        format_str = json.dumps(log_dict)
        self.flush_write(format_str)

    def write2csv(self, log_dict):
        format_str = list(log_dict.values())
        self.csv_writer.writerow(format_str)


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
    ckpt_path = os.path.join(CURRENT_PATH, 'save_Model')
    log_path = os.path.join(CURRENT_PATH, 'log')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)


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
        torch.backends.cudnn.enabled = True        # pytorch使用cuDNN加速
