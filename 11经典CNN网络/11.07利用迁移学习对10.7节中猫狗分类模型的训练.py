from torch.utils.data import Dataset, DataLoader      
from PIL import Image                            # 图像处理库
import random                                    # 用于随机选择图像
import torch                                                
from torchvision import transforms               # 图像预处理库
import torch.nn as nn                            # 神经网络模块
import os                                        # 用于文件和目录操作
# 导入预训练模型进行迁移学习
import torchvision.models                        
from torchvision import models                  

torchvision.models.resnet18()                    # 加载ResNet-18模型的结构，但不加载预训练权重

def verify_images(image_folder):                 # 这里可以添加代码来验证图像文件是否存在，并且可以被正确加载
    classes = ["Cat","Dog"]
    class_to_idx = {"Cat": 0, "Dog": 1}
    samples = []                                   # 存储图像路径和标签的列表
    for cls_name in classes:
        cls_dir = os.path.join(image_folder, cls_name)
        for fname in os.listdir(cls_dir):          # 遍历类别目录中的每个文件
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(cls_dir, fname)
            try:
                with Image.open(path) as img:
                    img.verify()                       # 验证图像是否可以被正确加载
                samples.append((path, class_to_idx[cls_name]))
            except Exception:
                print(f"Warning: {path} is not a valid image file and will be skipped.")
    return samples

class ImageDataset(Dataset):                           # 定义一个自定义数据集类，继承自torch.utils.data.Dataset
    def __init__(self, samples, transform=None):       # 初始化方法，接受图像文件夹路径和可选的图像预处理变换
        self.samples = samples                         # 验证图像并获取样本列表
        self.transform = transform                     # 存储图像预处理变换

    def __len__(self):                                 # 返回数据集的大小
        return len(self.samples)

    def __getitem__(self, idx):                        # 根据索引获取图像和标签
        path, label = self.samples[idx]                # 获取图像路径和标签
        with Image.open(path) as img:                  # 打开图像文件
            image = img.convert('RGB')                 # 打开图像并转换为RGB格式
            if self.transform:                         # 如果有预处理变换，则应用它们
               image = self.transform(image)           # 应用图像预处理变换
        return image, label                            # 返回图像和标签
    
def evaluate(model,test_dataloader):                  # 定义一个评估函数，接受模型和测试数据加载器作为输入，并返回验证准确率
    model.eval()                                      # 将模型设置为评估模式
    val_correct = 0                                   # 初始化正确预测的计数器
    val_total = 0                                     # 初始化总样本数的计数器

    with torch.no_grad():                                    # 在评估过程中不计算梯度
        for inputs,labels in test_dataloader:                # 遍历测试数据加载器中的每个批次
            inputs = inputs.to(DEVICE)                       # 将输入数据和标签移动到设备上
            labels = labels.float().unsqueeze(1).to(DEVICE)  # 将标签转换为浮点数，并调整形状以匹配模型输出

            outputs = model(inputs)                             # 获取模型的输出
            predicted = (outputs > 0.5).float()                 # 将输出转换为二分类预测结果，使用0.5作为阈值
            val_correct += (predicted == labels).sum().item()   # 累加正确预测的数量
            val_total += labels.size(0)                         # 累加总样本数

    val_accuracy = val_correct / val_total                      # 计算验证准确率
    return val_accuracy                                         # 返回验证准确率

if __name__ == "__main__":                   
    DATA_DIR = r"E:\code\DeepLearning\10卷积神经网络\PetImages"  # 数据集目录路径
    BATCH_SIZE = 64 
    IMG_SIZE = 128
    EPOCHS = 15
    LR = 0.001
    PRINT_STEP = 10              # 每隔多少步打印一次训练状态

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测是否有可用的GPU，如果没有则使用CPU

    all_samples = verify_images(DATA_DIR)     # 验证图像并获取样本列表
    random.seed(42)                           # 设置随机种子以确保结果可复现
    random.shuffle(all_samples)               # 打乱样本列表以增加训练的随机性
    train_size = int(0.8 * len(all_samples))  # 计算训练集的大小
    train_samples = all_samples[:train_size]  # 获取训练集样本(到train_size之前的样本)
    valid_samples = all_samples[train_size:]  # 获取验证集样本（从train_size到结束的样本）

    train_transform = transforms.Compose([    # 对训练集进行一系列图像预处理和数据增强变换
    transforms.Resize((150, 150)),
    transforms.RandomCrop(size=(IMG_SIZE, IMG_SIZE)),   # 随机裁剪图像到指定大小
    transforms.RandomHorizontalFlip(p=0.5),             # 随机水平翻转图像，p=0.5表示有50%的概率进行翻转
    transforms.ColorJitter(                             # 定义一个颜色抖动变换，随机调整图像的亮度、对比度、饱和度和色调
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.1
    ), 
    transforms.RandomRotation(degrees=30),              # 随机旋转图像，旋转角度在-30到30度之间
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([    # 对验证集进行图像预处理变换，通常不包括数据增强，以保持验证数据的真实性
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(train_samples, transform=train_transform)  # 创建训练数据集对象，传入训练样本和预处理变换
    valid_dataset = ImageDataset(valid_samples, transform=valid_transform)  # 创建验证数据集对象，传入验证样本和预处理变换

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4)  # 创建训练数据加载器，设置批量大小和是否打乱数据,num_workers=4表示使用4个子进程加载数据
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)  # 创建验证数据加载器，设置批量大小和不打乱数据,num_workers=4表示使用4个子进程加载数据

    model = models.resnet18(pretrained=True)       # 加载预训练的ResNet-18模型,pretrained=True表示加载预训练权重
    for param in model.parameters():
        param.requires_grad = False                # 冻结所有层
 
    # 解冻第四阶段的参数（最后一个残差模块）
    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)           # 二分类
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()             # 定义损失函数，使用二分类交叉熵损失函数，适用于二分类问题，并且结合了Sigmoid激活函数
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 定义优化器，使用Adam优化器，并传入模型参数和学习率

    for epoch in range(EPOCHS):  # 训练循环，迭代指定的训练轮数
        print(f"Epoch {epoch+1}/{EPOCHS}")  # 打印当前的训练轮数
        model.train()  # 将模型设置为训练模式
        running_loss = 0.0  # 初始化运行损失

        for step, (inputs, labels) in enumerate(train_dataloader):  # 遍历训练数据加载器中的每个批次
            inputs = inputs.to(DEVICE)  # 将输入数据和标签移动到设备上
            labels = labels.float().unsqueeze(1).to(DEVICE)  # 将标签转换为浮点数，并调整形状以匹配模型输出

            optimizer.zero_grad()               # 清除之前的梯度
            outputs = model(inputs)             # 获取模型的输出
            loss = criterion(outputs, labels)   # 计算损失
            loss.backward()                     # 反向传播计算梯度
            optimizer.step()                    # 更新模型参数

            running_loss += loss.item()         # 累加当前批次的损失

            if (step + 1) % PRINT_STEP == 0:    # 每隔PRINT_STEP步打印一次训练状态
                print(f"Step {step+1}/{len(train_dataloader)}, Loss: {running_loss / PRINT_STEP:.4f}")  # 打印当前步骤和平均损失
                running_loss = 0.0              # 重置运行损失,以便计算下一个PRINT_STEP步的平均损失

        val_accuracy = evaluate(model, valid_dataloader)                         # 在每个训练轮结束后评估模型在验证集上的准确率
        print(f"Validation Accuracy after Epoch {epoch+1}: {val_accuracy:.4f}")  # 打印当前训练轮结束后的验证准确率

# 未进行图像增强  Validation Accuracy after Epoch 15: 0.8710
# 进行图像增强后  Validation Accuracy after Epoch 15: 0.8188
# 进行图像增强后，验证准确率有所下降，这可能是因为数据增强引入了更多的变异性，使得模型在训练过程中更难以收敛到最佳解。
# 数据增强虽然可以提高模型的泛化能力，但也可能增加训练的难度，特别是在训练轮数较少或者模型容量有限的情况下。
# 应该需要更多的EPOCHS来让模型适应增强后的数据，从而提高验证准确率。

# 利用ResNet-18模型进行迁移学习，解冻最后一个残差块的参数，并更换分类头为二分类(存在图像增强)
# Validation Accuracy after Epoch 15: 0.9684
# 进行迁移学习后，验证准确率显著提高，这表明利用预训练模型的特征提取能力，
# 并且允许最后一个残差块进行微调，可以更好地适应新的二分类任务，从而提升模型的性能。
