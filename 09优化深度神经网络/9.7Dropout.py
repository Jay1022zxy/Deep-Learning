# 模型定义 （加入 Dropout 层）
# Dropout 是一种正则化技术，在训练过程中随机丢弃神经网络中的一部分神经元，以防止过拟合。
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),        
            nn.Dropout(0.2),        # 添加 Dropout 层，丢弃率为 0.2
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),        # 添加 Dropout 层，丢弃率为 0.2
            nn.Linear(128, 64),
            nn.ReLU(),   
            nn.Dropout(0.2),        # 添加 Dropout 层，丢弃率为 0.2 
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)
    
# model.train()    # 训练模式，启用 Dropout
## 训练模型

# model.eval()     # 评估模式，禁用 Dropout
## 评估模型
