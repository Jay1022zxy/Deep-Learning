# 更换分类头
from torchvision import models     # 导入预训练模型
import torch.nn as nn

model = models.resnet18(pretrained=True)     # 加载预训练的ResNet-18模型,pretrained=True表示加载预训练权重
for param in model.parameters():             
    param.requires_grad = False              # 冻结所有层,使其在训练过程中不更新

in_features = model.fc.in_features           # 获取ResNet-18模型中最后一个全连接层的输入特征数
model.fc = nn.Linear(in_features, 1)         # 新生成一个二分类头

# 更换分类头的同时解冻最后一个残差块的参数
model = models.resnet18(pretrained=True)     
for param in model.parameters():             
    param.requires_grad = False              

# 解冻最后一个残差块的参数
for param in model.layer4.parameters():
    param.requires_grad = True

in_features = model.fc.in_features           
model.fc = nn.Linear(in_features, 1)       
