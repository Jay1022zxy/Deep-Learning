import torch
import torch.nn as nn
import math

class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float): # d_model: 模型维度, seq_len: 序列长度, dropout: dropout概率
        super().__init__()
        self.d_model = d_model  # 模型维度(embedding维度)
        self.seq_len = seq_len  # 序列长度
        self.dropout = nn.Dropout(dropout)  # dropout层
        # 创建一个位置编码矩阵，形状为(seq_len, d_model)，用于存储每个位置的编码
        pe = torch.zeros(seq_len, d_model)  # 初始化位置编码矩阵为全零
        # 创建一个位置向量，表示每个位置的索引，形状为(seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # 创建一个位置向量，表示每个位置的索引，形状为(seq_len, 1)
        # 计算分母
        div_term = torch.pow(10000.0, -torch.arange(0, d_model, 2).float() / d_model)  # 计算分母，形状为(d_model/2)
        # 偶数位使用sin函数，奇数位使用cos函数，计算位置编码矩阵的值
        pe[:, 0::2] = torch.sin(position * div_term)   # sin(position / (10000^(2i/d_model))) 
        pe[:, 1::2] = torch.cos(position * div_term)   # cos(position / (10000^(2i/d_model)))
        # 增加batch维度，形状变为(1, seq_len, d_model)，并注册为模型的一个buffer，这样在保存和加载模型时会自动处理
        pe = pe.unsqueeze(0)  # 增加batch维度，形状变为(1, seq_len, d_model)
        self.register_buffer('pe', pe)  # 注册位置编码矩阵为模型的一个buffer，这样在保存和加载模型时会自动处理

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False) # 将输入x与位置编码矩阵相加，位置编码矩阵根据输入的序列长度进行切片，并设置requires_grad为False，表示位置编码不参与梯度计算
        return self.dropout(x)  # 应用dropout并返回结果
