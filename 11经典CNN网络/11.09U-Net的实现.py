import torch 
import torch.nn as nn

class DoubleConv(nn.Module):                        # 生成两个连续的3x3卷积层（padding为1）+ BatchNorm + ReLU
    def __init__(self, in_channels, out_channels): 
        super(DoubleConv, self).__init__() 
        self.double_conv = nn.Sequential( 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),   # 第一个卷积层
            nn.BatchNorm2d(out_channels),                                                 # 批归一化层，有助于加速训练和提高模型的稳定性
            nn.ReLU(inplace=True),                                                        # ReLU激活函数

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) 
        ) 
    def forward(self, x):                    # 前向传播函数
        return self.double_conv(x)
    
class UNet(nn.Module):                           # 定义UNet模型类，继承自nn.Module
    def __init__(self, in_channels=3, out_channels=1): 
        super(UNet, self).__init__() 
        # 编码路径
        self.conv1 = DoubleConv(in_channels, 64)       
        self.pool1 = nn.MaxPool2d(kernel_size=2)       
        self.conv2 = DoubleConv(64, 128)                
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = DoubleConv(512, 1024)             # 最低层

        # 解码路径 （转置卷积增加特征图宽和高，但是降低通道数）
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 上采样层 ×2
        self.conv6 = DoubleConv(1024, 512)             # 拼接后通道数为1024（512+512）
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)              # 拼接后通道数为512（256+256）
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)              # 拼接后通道数为256（128+128）
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)               # 拼接后通道数为128（64+64）
        self.conv10 = nn.Conv2d(64, out_channels, kernel_size=1)  # 最后输出层，使用1x1卷积将输出通道数转换为要分的类别数

    def forward(self, x):
        # 编码路径
        c1 = self.conv1(x)        # c1：[B, 64, H, W]
        p1 = self.pool1(c1)       # p1：[B, 64, H/2, W/2]
        c2 = self.conv2(p1)       # c2：[B, 128, H/2, W/2]
        p2 = self.pool2(c2)       # p2：[B, 128, H/4, W/4]
        c3 = self.conv3(p2)       # c3：[B, 256, H/4, W/4]
        p3 = self.pool3(c3)       # p3：[B, 256, H/8, W/8]
        c4 = self.conv4(p3)       # c4：[B, 512, H/8, W/8]
        p4 = self.pool4(c4)       # p4：[B, 512, H/16, W/16]
        c5 = self.conv5(p4)       # c5：[B, 1024, H/16, W/16] 最低层的特征图，包含了输入图像的全局上下文信息，但空间分辨率较低

        # 解码,第一级上采样
        up6 = self.up6(c5)                     # up6：[B, 512, H/8, W/8] 上采样后特征图的尺寸增加到H/8 x W/8，通道数减少到512
        # 直接拼接 c4([B, 512, H/8, W/8]) 和 up6([B, 512, H/8, W/8])，得到 merge6([B, 1024, H/8, W/8])
        merge6 = torch.cat([up6, c4], dim=1)   # 拼接后：[B, 1024, H/8, W/8]
        c6 = self.conv6(merge6)                # c6：[B, 512, H/8, W/8] 经过卷积后，通道数减少到512，空间分辨率保持不变
        # 解码，第二级
        up7 = self.up7(c6)                     # up7：[B, 256, H/4, W/4] 上采样后特征图的尺寸增加到H/4 x W/4，通道数减少到256
        # 直接拼接 c3([B, 256, H/4, W/4]) 和 up7([B, 256, H/4, W/4])，得到 merge7([B, 512, H/4, W/4])
        merge7 = torch.cat([up7, c3], dim=1)   # 拼接后：[B, 512, H/4, W/4]
        c7 = self.conv7(merge7)                # c7：[B, 256, H/4, W/4] 经过卷积后，通道数减少到256，空间分辨率保持不变
        # 解码，第三级
        up8 = self.up8(c7)                     # up8：[B, 128, H/2, W/2] 上采样后特征图的尺寸增加到H/2 x W/2，通道数减少到128
        # 直接拼接 c2([B, 128, H/2, W/2]) 和 up8([B, 128, H/2, W/2])，得到 merge8([B, 256, H/2, W/2])
        merge8 = torch.cat([up8, c2], dim=1)   # 拼接后：[B, 256, H/2, W/2]
        c8 = self.conv8(merge8)                # c8：[B, 128, H/2, W/2] 经过卷积后，通道数减少到128，空间分辨率保持不变
        # 解码，第四级                                
        up9 = self.up9(c8)                     # up9：[B, 64, H, W] 上采样后特征图的尺寸增加到H x W，通道数减少到64
        # 直接拼接 c1([B, 64, H, W]) 和 up9([B, 64, H, W])，得到 merge9([B, 128, H, W])
        merge9 = torch.cat([up9, c1], dim=1)   # 拼接后：[B, 128, H, W]
        c9 = self.conv9(merge9)                # c9：[B, 64, H, W] 经过卷积后，通道数减少到64，空间分辨率保持不变
        # 最后一层1x1卷积，输出分割结果
        output = self.conv10(c9)  # output：[B, out_channels, H, W]，其中out_channels是要分的类别数（例如二分类时为1，多分类时为类别数）
        return output

        