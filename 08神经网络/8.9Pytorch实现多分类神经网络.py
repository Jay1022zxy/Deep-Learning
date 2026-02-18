import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class MNISTDataset(Dataset):                                  # 定义一个MNIST数据集
    def __init__(self, file_path):
        self.images, self.labels = self._read_file(file_path) # 读取文件，获取图像数据和标签数据

    def _read_file(self, file_path):                          # 对文件的读取和处理方法
        images = []                                           # 存储图像数据的列表
        labels = []                                           # 存储标签数据的列表
        with open(file_path, "r") as f:                       # 打开文件进行读取
            next(f)                                           # 跳过第一行（标题行）
            for line in f:                                    # 逐行读取文件内容
                line = line.rstrip("\n")                      # 去除行末的换行符
                items = line.split(",")                       # 将每行数据按逗号分割成列表
                images.append([float(x) for x in items[1:]])  # 将图像数据转换为浮点数并添加到images列表中
                labels.append(int(items[0]))                  # 将标签数据转换为整数并添加到labels列表中
        return images, labels                                 # 返回图像数据和标签数据
    
    def __len__(self):                                        # 返回数据集的大小，方便DataLoader进行迭代
        return len(self.labels)
    
    def __getitem__(self, idx):                               # 根据索引获取数据，方便DataLoader进行批处理
        image, label = self.images[idx], self.labels[idx]     # 获取指定索引的图像数据和标签数据
        image = torch.tensor(image)                           # 将图像数据转换为PyTorch张量
        image = image / 255.0                                 # 将图像数据归一化到0-1之间
        image = (image - 0.1307) / 0.3081                     # 对图像数据进行标准化
        return image, label                                   # 返回图像数据和标签数据


class NeuralNetwork(nn.Module):          # 定义一个神经网络类，继承自nn.Module
    def __init__(self):                  # 初始化网络结构
        super().__init__()               # 调用父类的初始化方法
        self.model = nn.Sequential(      # 使用Sequential来构建网络结构，方便定义前向传播过程
            nn.Linear(28*28, 128),       # 第一层线性层，输入为28*28，输出为128
            nn.ReLU(),                   # 第一层ReLU激活函数
            nn.Linear(128, 128),         # 第二层线性层，输入为128，输出为128
            nn.ReLU(),                   # 第二层ReLU激活函数
            nn.Linear(128, 64),          # 第三层线性层，输入为128，输出为64
            nn.ReLU(),                   # 第三层ReLU激活函数
            nn.Linear(64, 10)            # 输出层线性层，输入为64，输出为10（类别数）
        )
    def forward(self, x):                # 定义前向传播过程
        return self.model(x)             # 将输入x通过定义的模型进行前向传播，得到输出
    
#  参数设置
lr = 0.1
epochs = 10
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 检测是否有可用的GPU，如果有则使用GPU，否则使用CPU

# 数据的加载
train_dataset = MNISTDataset(r"E:\code\DeepLearning\08神经网络\mnist\mnist_train.csv")    # 定义训练集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)             # 定义训练数据加载器，并打乱数据
test_dataset = MNISTDataset(r"E:\code\DeepLearning\08神经网络\mnist\mnist_test.csv")      # 定义测试集
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)               # 定义测试数据加载器，打乱数据

# 模型，损失函数和优化器的定义
model = NeuralNetwork().to(device)                         # 实例化神经网络模型，并将其移动到指定设备上
criterion = nn.CrossEntropyLoss()                          # 定义交叉熵损失函数，适用于多分类问题
optimizer = torch.optim.SGD(model.parameters(), lr=lr)     # 定义优化器，使用随机梯度下降（SGD），并设置学习率

# 训练过程
model.train()            # 设置模型为训练模式
for epoch in range(epochs): # 迭代训练
    correct = 0                           # 初始化正确预测的数量
    total_loss = 0.0                      # 初始化总损失
    total = 0                             # 初始化总样本数量
    for images, labels in train_loader:   # 遍历训练数据加载器，获取每个批次的特征和标签
        images, labels = images.to(device), labels.to(device)     # 将特征和标签移动到指定设备上

        outputs = model(images)               # 将特征输入模型，得到输出

        loss = criterion(outputs, labels)     # 计算损失值

        optimizer.zero_grad()                 # 清除之前的梯度信息
        loss.backward()                       # 反向传播，计算当前梯度
        optimizer.step()                      # 更新模型参数

        total_loss += loss.item()                 # 累加损失值

        # 计算训练准确率
        preds = torch.argmax(outputs, dim=1)      # 获取模型预测的类别
        correct += (preds == labels).sum().item() # 累加正确预测的数量
        total += labels.size(0)                   # 累加总样本数量
       
    avg_loss = total_loss / len(train_loader)     # 计算平均损失值
    train_accuracy = correct / total * 100        # 计算训练准确率
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%") # 打印当前迭代的损失值和准确率

# 测试过程
model.eval()             # 设置模型为评估模式
with torch.no_grad():    # 在评估模式下，不需要计算梯度，因此使用
    correct = 0                          
    total = 0                             
    for images, labels in test_loader:   
        images, labels = images.to(device), labels.to(device)     
        outputs = model(images)               
        preds = torch.argmax(outputs, dim=1)      
        correct += (preds == labels).sum().item() 
        total += labels.size(0)               

    test_accuracy = correct / total * 100          # 计算测试准确率
    print(f"Test Accuracy: {test_accuracy:.2f}%")  # 打印测试准确率


