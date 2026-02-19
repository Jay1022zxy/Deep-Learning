import torch.nn as nn
import torch

# 批量归一化目的是通过对每一层的输入进行归一化来加速训练和提高模型的稳定性

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128, bias=False),     # 不用 bias
            nn.BatchNorm1d(128),                     # 添加 BatchNorm1d 层,实现批量归一化
            nn.ReLU(),

            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)
    
# model.train()    # 训练模式，启用 BatchNorm
# 训练模型

# model.eval()     # 评估模式，启用 BatchNorm
# 评估模型    

# 切换训练和评估模式是必要的，因为BatchNorm 在训练模式下会计算当前批次的均值和方差来进行归一化
# 而在评估模式下会使用在训练过程中累积的均值和方差来进行归一化。因此，在训练和评估时需要切换模式以确保 BatchNorm 的正确行为。