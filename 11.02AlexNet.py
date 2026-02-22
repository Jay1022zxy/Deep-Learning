import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):      # 初始化方法，定义网络层，接受一个参数num_classes，表示输出类别的数量，默认为1000
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),      # 卷积层,输出: 96x55x55
            nn.ReLU(inplace=True),                                      # 激活函数，使用ReLU，inplace=True表示直接在输入上进行修改以节省内存
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0), # 局部响应归一化层，size=5表示在局部区域内进行归一化，alpha、beta和k是归一化的超参数
            nn.MaxPool2d(kernel_size=3, stride=2),                      # 最大池化层,输出: 96x27x27

            nn.Conv2d(96, 256, kernel_size=5, padding=2),               # 卷积层,输出: 256x27x27
            nn.ReLU(inplace=True),                                      # 激活函数，使用ReLU，inplace=True表示直接在输入上进行修改以节省内存
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0), # 局部响应归一化层，size=5表示在局部区域内进行归一化，alpha、beta和k是归一化的超参数
            nn.MaxPool2d(kernel_size=3, stride=2),                      # 最大池化层, 输出: 256x13x13
 
            nn.Conv2d(256, 384, kernel_size=3, padding=1),              # 卷积层,输出: 384x13x13
            nn.ReLU(inplace=True),                                      # 激活函数，使用ReLU，inplace=True表示直接在输入上进行修改以节省内存

            nn.Conv2d(384, 384, kernel_size=3, padding=1),              # 卷积层,输出: 384x13x13
            nn.ReLU(inplace=True),                                      # 激活函数，使用ReLU，inplace=True表示直接在输入上进行修改以节省内存

            nn.Conv2d(384, 256, kernel_size=3, padding=1),              # 卷积层,输出: 256x13x13
            nn.ReLU(inplace=True),                                      # 激活函数，使用ReLU，inplace=True表示直接在输入上进行修改以节省内存
            nn.MaxPool2d(kernel_size=3, stride=2)                       # 最大池化层,输出: 256x6x6
        )
        self.classifier = nn.Sequential(                                # 分类器部分，使用全连接层和激活函数组成的序列
            nn.Dropout(p=0.5),                                          # Dropout层，p=0.5表示在训练过程中以50%的概率随机丢弃神经元，以防止过拟合
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),                                      # 激活函数，使用ReLU，inplace=True表示直接在输入上进行修改以节省内存

            nn.Dropout(p=0.5),                                          # Dropout层，p=0.5表示在训练过程中以50%的概率随机丢弃神经元，以防止过拟合
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),                                      # 激活函数，使用ReLU，inplace=True表示直接在输入上进行修改以节省内存

            nn.Linear(4096, num_classes)                                # 输出层，输入4096个特征，输出num_classes个类别的概率
        )

    def forward(self, x):                               # 定义前向传播方法，接受输入数据并返回输出
        x = self.features(x)                            # 通过特征提取部分处理输入数据，得到卷积层的输出特征图
        x = x.view(x.size(0), 256 * 6 * 6)              # 将卷积层的输出特征图展平为一维向量，准备输入全连接层，x.size(0)表示批次大小 
        x = self.classifier(x)                          # 通过分类器部分处理展平后的特征向量，得到最终的类别概率输出
        return x