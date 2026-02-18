import torch
import torch.nn as nn   
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        # nn.Linear也继承自nn.Module, 输入为input_dim, 输出一个值
        self.linear = nn.Linear(input_dim, 1)  # 定义一个线性层，输入维度为input_dim，输出维度为1

    def forward(self, x):  # logistic regression 输出一个概率值, 因此需要经过sigmoid函数
        return torch.sigmoid(self.linear(x))  # 前向传播，输出经过sigmoid函数的结果