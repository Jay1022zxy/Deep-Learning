import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init___(self):
        super(LeNet, self).__init__()                   # 调用父类的初始化方法
        # C1: 卷积层，输入通道数为1，输出通道数为6，卷积核大小为5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # S2:平均池化层，使用2x2的平均池化
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: 卷积层，输入通道数为6，输出通道数为16，卷积核大小为5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # S4:平均池化层，使用2x2的平均池化
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C5:全连接等价层(输入16×5×5的特征图，输出120个特征)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        # F6:全连接层，输入120个特征，输出84个特征
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        # 输出层，输入84个特征，输出10个类别的概率
        self.fc2 = nn.Linear(in_features=84, out_features=10)
    
    def forward(self, x):                            # 定义前向传播方法，接受输入数据并返回输出
        x = F.tanh(self.conv1(x))                    # C1 + tanh激活函数
        x = self.pool1(x)                            # S2层进行平均池化
        x = F.tanh(self.conv2(x))                    # C3 + tanh激活函数
        x = self.pool2(x)                            # S4层进行平均池化
        x = F.tanh(self.conv3(x))                    # C5 + tanh激活函数
        x = x.view(-1, 120)                          # 将特征图展平为一维向量，准备输入全连接层
        x = F.tanh(self.fc1(x))                      # F6 + tanh激活函数
        x = self.fc2(x)                              # 输出层全连接，得到最终的类别概率输出
        return x

