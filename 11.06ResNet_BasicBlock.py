import torch
import torch.nn as nn

# 整体思想：
# 两个 3x3 卷积，每个卷积后跟 BN 和 ReLU。若输入输出维度不一致，则在捷径路径使用 1x1 卷积。

class BasicBlock(nn.Module):
    expansion = 1            # 扩展系数，表示输出通道数相对于输入通道数的倍数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock,self).__init__()
        # 第一个卷积层可能输入和输出通道数不同
        # 除了Conv2,对于Conv3-5的第一个Residual Block，输入和输出通道数不一致，则步长设置为2，以实现空间尺寸的减半
        # 其他情况下，输入和输出通道数相同，步长设置为1，以保持空间尺寸不变
        self.cov1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)  # 第一个卷积层，使用3x3卷积核，步长和填充根据参数设置
        self.bn1 = nn.BatchNorm2d(out_channels)                      # 第一个卷积层后的批归一化层,用于加速训练和提高模型的稳定性
        self.relu = nn.ReLU(inplace=True)                            # ReLU激活函数，inplace=True表示直接在输入上进行修改以节省内存
        # 第二个卷积层，输入和输出通道数相同，步长为1，保持空间尺寸不变
        self.cov2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)                      # 第二个卷积层后的批归一化层
        self.downsample = downsample                                 # 下采样层，实际为1×1，步长为2的卷积层，用于调整输入的通道数和空间尺寸以匹配残差连接的输出

    def forward(self, x):                            # 定义前向传播方法，接受输入数据并返回输出
        identity = x                                 # 保存输入数据作为残差连接的输入

        out = self.cov1(x)                           # 通过第一个卷积层处理输入数据
        out = self.bn1(out)                          # 通过第一个批归一化层处理卷积层的输出
        out = self.relu(out)                         # 通过ReLU激活函数处理批归一化层的输出

        out = self.cov2(out)                         # 通过第二个卷积层处理数据
        out = self.bn2(out)                          # 通过第二个批归一化层处理卷积层的输出

        if self.downsample is not None:              # 如果存在下采样层，说明输入和输出通道数或空间尺寸不匹配，需要进行调整
            identity = self.downsample(x)            # 通过下采样层调整输入数据以匹配残差连接的输出

        out += identity                              # 将调整后的输入数据与卷积层的输出进行相加，形成残差连接
        out = self.relu(out)                         # 通过ReLU激活函数处理相加后的结果，得到最终的输出

        return out