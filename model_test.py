import torch
from model.AnomalyTransformer import AnomalyTransformer  # 根据项目中的模型类名调整
from data_factory.data_loader import CustomCSVSegLoader
from torch.utils.data import DataLoader

# 实例化模型
model = AnomalyTransformer()  # 参数需与训练时一致
# 加载权重
model.load_state_dict(torch.load('checkpoints/credit_checkpoint.pth'))
# 设置为评估模式
model.eval()

# 选择设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import pandas as pd
import numpy as np

# 读取 CSV 文件
df = pd.read_csv('data/YT.11PI_45201.PV.csv')
def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='credit'):
    if dataset == 'credit':
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

from collections import deque

# 设置窗口大小（与训练时一致）
window_size = 100  # 根据项目配置调整
history_window = deque(maxlen=window_size)

# 示例：添加新数据点
def update_window(new_data_point):
    history_window.append(new_data_point)

# 将窗口数据转换为 PyTorch tensor
input_data = torch.tensor(np.array(history_window), dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():  # 推理时关闭梯度计算
    output = model(input_data)
    anomaly_score = output.item()  # 假设模型输出异常分数

threshold = 0.9  # 根据训练数据分布或实验调整阈值
if anomaly_score > threshold:
    print("检测到异常！分数:", anomaly_score)
else:
    print("正常数据，分数:", anomaly_score)

import time

while True:
    df = pd.read_csv('data/YT.11PI_45201.PV.csv')
    new_data_point = df['value'].iloc[-1]  # 获取最新数据点
    update_window(new_data_point)
    if len(history_window) == window_size:  # 窗口填满后开始检测
        input_data = torch.tensor(np.array(history_window), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_data)
            anomaly_score = output.item()
            if anomaly_score > threshold:
                print("检测到异常！分数:", anomaly_score)
            else:
                print("正常数据，分数:", anomaly_score)
    time.sleep(60)  # 控制检测频率，例如每秒检测一次
