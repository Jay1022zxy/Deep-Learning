import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6): # features: 输入特征的维度, eps: 防止除以0的一个小数
        super().__init__()
        self.eps = eps  # 防止除以0的一个小数
        self.gamma = nn.Parameter(torch.ones(features))  # 可学习的权重，初始化为1
        self.beta = nn.Parameter(torch.zeros(features))  # 可学习的偏移参数，初始化为0

    def forward(self, x):
        # x的形状为(batch_size, seq_len, features)，对最后一个维度进行归一化
        # 计算输入x的均值和标准差，保持维度不变以便后续计算
        mean = x.mean(dim=-1, keepdim=True)  # 计算输入x的均值，保持维度不变   (batch_size, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)    # 计算输入x的标准差，保持维度不变 (batch_size, seq_len, 1)
        # eps是为了防止除以0的情况，进行数值稳定性的处理
        return self.gamma * (x - mean) / (std + self.eps) + self.beta  # 对输入x进行归一化，并应用可学习的权重和偏移参数，得到最终的输出
        
        