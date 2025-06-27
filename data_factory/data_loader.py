import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# 添加你的自定义CSV数据加载器
class CustomCSVSegLoader(object):
    # 类变量用于缓存数据，避免重复加载
    _cached_data = None
    _cached_path = None
    
    def __init__(self, data_path, win_size, step, mode="train", train_ratio=0.8):
        """
        修正版的自定义CSV数据加载器
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        
        # 只有当路径改变时才重新加载数据
        if CustomCSVSegLoader._cached_path != data_path or CustomCSVSegLoader._cached_data is None:
            print(f"Loading CSV data from: {data_path}")
            self._load_and_process_data(data_path, train_ratio)
            CustomCSVSegLoader._cached_path = data_path
        else:
            print(f"Using cached data for: {data_path}")
            
        # 从缓存获取数据
        self._assign_data_from_cache()

    def _load_and_process_data(self, data_path, train_ratio):
        """加载并处理数据"""
        # 读取CSV数据
        data = pd.read_csv(data_path)
        
        # 确保列名正确
        if 'timestamp' in data.columns and 'value' in data.columns and 'label' in data.columns:
            # 按时间戳排序
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            # 提取值和标签
            values = data['value'].values.reshape(-1, 1)
            labels = data['label'].values
            
            # 处理NaN值
            values = np.nan_to_num(values)
            
            # 数据标准化
            scaler = StandardScaler()
            values = scaler.fit_transform(values)
            
            # 分割数据 - 确保有足够的数据进行训练
            total_len = len(values)
            split_idx = int(total_len * train_ratio)
            
            # 确保训练集和测试集都有足够的数据
            min_test_size = max(self.win_size * 10, 1000)  # 至少要有10个窗口
            if total_len - split_idx < min_test_size:
                split_idx = max(total_len - min_test_size, total_len // 2)
            
            train_values = values[:split_idx]
            test_values = values[split_idx:]
            test_labels = labels[split_idx:]
            
            # 缓存数据
            CustomCSVSegLoader._cached_data = {
                'train': train_values,
                'test': test_values,
                'test_labels': test_labels,
                'scaler': scaler
            }
            
            print("train shape:", train_values.shape)
            print("test shape:", test_values.shape) 
            print("test_labels shape:", test_labels.shape)
            
        else:
            raise ValueError("CSV文件必须包含 'timestamp', 'value', 'label' 列")

    def _assign_data_from_cache(self):
        """从缓存分配数据"""
        cached = CustomCSVSegLoader._cached_data
        self.train = cached['train']
        self.test = cached['test']
        self.test_labels = cached['test_labels']
        self.scaler = cached['scaler']
        
        # 根据模式设置对应的数据
        if self.mode == 'train':
            self.data = self.train
            # 训练模式使用全0标签（无监督学习）
            self.labels = np.zeros((len(self.train), 1))
        elif self.mode in ['val', 'test', 'thre']:
            self.data = self.test
            self.labels = self.test_labels.reshape(-1, 1)
        else:
            self.data = self.test
            self.labels = self.test_labels.reshape(-1, 1)

    def __len__(self):
        """返回数据集长度"""
        if self.mode == "train":
            # 确保有足够的训练步骤
            return max(10, (self.data.shape[0] - self.win_size) // self.step + 1)
        else:
            return max(1, (self.data.shape[0] - self.win_size) // self.step + 1)

    def __getitem__(self, index):
        """获取数据项"""
        # 确保索引不会超出范围
        max_start_idx = self.data.shape[0] - self.win_size
        start_idx = min(index * self.step, max_start_idx)
        end_idx = start_idx + self.win_size
        
        # 获取数据窗口
        data_window = self.data[start_idx:end_idx]
        
        # 获取标签窗口
        if self.mode == 'train':
            # 训练模式使用全0标签
            label_window = np.zeros((self.win_size, 1), dtype=np.float32)
        else:
            # 确保标签索引不超出范围
            label_start_idx = min(start_idx, len(self.labels) - self.win_size)
            label_end_idx = label_start_idx + self.win_size
            label_window = self.labels[label_start_idx:label_end_idx]
            
            # 确保标签是正确的形状
            if len(label_window.shape) == 1:
                label_window = label_window.reshape(-1, 1)
        
        return np.float32(data_window), np.float32(label_window)


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if dataset == 'SMD':
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif dataset == 'MSL':
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'SMAP':
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'PSM':
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'credit':  # 添加对credit数据集的支持
        dataset = CustomCSVSegLoader(data_path, win_size, step, mode)
    else:
        raise ValueError(f"未知的数据集类型: {dataset}")

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0)
    return data_loader
