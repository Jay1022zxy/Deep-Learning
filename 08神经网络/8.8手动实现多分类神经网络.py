import torch 
from torch.utils.data import Dataset, DataLoader

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
    
batch_size = 64
train_dataset = MNISTDataset(r"E:\code\DeepLearning\08神经网络\mnist\mnist_train.csv")    # 定义训练集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)             # 定义训练数据加载器，并打乱数据
test_dataset = MNISTDataset(r"E:\code\DeepLearning\08神经网络\mnist\mnist_test.csv")      # 定义测试集
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)               # 定义测试数据加载器，打乱数据

lr = 0.1 
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")              # 检测是否有可用的GPU，如果有则使用GPU，否则使用CPU
# 配置网络结构，包含输入层、隐藏层和输出层大小  
layers_sizes = [28*28, 128, 128, 64, 10]    # 定义网络层的大小，输入层为28*28（图像像素数量），隐藏层分别为128、128和64，输出层为10（类别数量）
# 手动初始化参数
weights = []                                # 存储权重的列表
biases = []                                 # 存储偏置的列表
for in_size, out_size in zip(layers_sizes[:-1], layers_sizes[1:]):  # 把每一层的输入大小和输出大小两两配对  
    weight = torch.randn(in_size, out_size, device=device) * (2.0 / in_size) ** 0.5  # 使用He初始化方法生成权重
    bias = torch.zeros(out_size, device=device)                                      # 初始化偏置为零
    weights.append(weight)              # 将权重添加到列表中
    biases.append(bias)                 # 将偏置添加到列表中

def relu(x):                            # 定义ReLU激活函数
    return torch.clamp(x, min=0.0)      # 将输入张量中的负值设置为0，正值保持不变

def relu_grad(x):                       # 定义ReLU函数对x的导数(梯度)
    return (x > 0).float()              # ReLU函数的梯度，输入大于0时返回1，否则返回0

def softmax(x):                                                       # 定义Softmax函数
    x_exp = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)   # 防止超过float32的范围,dim=1表示按行计算最大值，keepdim=True保持维度不变
    return x_exp / torch.sum(x_exp, dim=1, keepdim=True)              # 计算Softmax输出

def cross_entropy_loss(pred, label):      # 定义交叉熵损失函数
    N = pred.shape[0]                     # 获取批大小
    one_hot = torch.zeros_like(pred)      # 创建一个与pred形状相同的全零张量
    one_hot[torch.arange(N), label] = 1   # 将正确类别的位置设置为1，形成one-hot编码
    loss = -(one_hot * torch.log(pred + 1e-8)).sum() / N  # 计算交叉熵损失，添加一个小常数防止log(0)
    return loss, one_hot

# 定义训练循环
for epoch in range(epochs):           
    total_loss = 0.0                       # 初始化总损失
    for images, labels in train_loader:    # 迭代训练数据加载器中的批次
        x = images.to(device)              # 将图像数据移动到设备
        y = labels.to(device)              # 将标签数据移动到设备
        N = x.shape[0]                     # 获取批大小

        # 前向传播
        activations = [x]                   # 第一层输入为x
        pre_acts = []                       # 存储每层的线性变换结果
        for weight, bias in zip(weights[:-1], biases[:-1]):    # 迭代隐藏层的权重和偏置
            pre_act = activations[-1] @ weight + bias          # 计算线性变换
            pre_acts.append(pre_act)                           # 将线性变换结果添加到列表中
            activation = relu(pre_act)                         # 应用ReLU激活函数
            activations.append(activation)                     # 将激活结果添加到列表中
        # 输出层的线性变换
        pre_out = activations[-1] @ weights[-1] + biases[-1]   # 计算输出层的线性变换
        pre_acts.append(pre_out)                               # 将输出层的线性变换结果添加到列表中
        out = softmax(pre_out)                                 # 应用Softmax函数得到预测概率

        # 计算损失
        loss, one_hot = cross_entropy_loss(out, y)             # 计算交叉熵损失和one-hot编码
        total_loss += loss.item()                              # 累加损失

        # 反向传播
        grad_W = [None] * len(weights)                         # 初始化权重梯度列表
        grad_b = [None] * len(biases)                          # 初始化偏置梯度列表

        grad_out = (out - one_hot) / N                         # 输出层的梯度
        grad_W[-1] = activations[-1].t() @ grad_out            # 输出层权重的梯度
        grad_b[-1] = grad_out.sum(dim=0)                       # 输出层偏置的梯度
        # 反向传播到隐藏层
        for i in range(len(weights) - 2, -1, -1):             # 迭代隐藏层的索引，从最后一个隐藏层到第一个
            grad_out = grad_out @ weights[i + 1].t() * relu_grad(pre_acts[i])  # 计算当前层的梯度
            grad_W[i] = activations[i].t() @ grad_out         # 当前层权重的梯度
            grad_b[i] = grad_out.sum(dim=0)                   # 当前层偏置的梯度

        # 更新参数 
        with torch.no_grad():                                  # 在更新参数时不计算梯度
            for i in range(len(weights)):                      # 迭代所有层的索引
                weights[i] -= lr * grad_W[i]                   # 更新权重
                biases[i] -= lr * grad_b[i]                    # 更新偏置

        avg_loss = total_loss / len(train_loader)              # 计算平均损失
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")   # 打印当前迭代的平均损失


# 测试集评估
with torch.no_grad():  # 在评估模型时不计算梯度
    correct = 0
    total = 0
    for images, labels in test_loader:                        # 迭代测试数据加载器中的批次
        x = images.view(-1,layers_sizes[0]).to(device)        # 将图像数据展平并移动到设备
        y = labels.to(device)                                 # 将标签数据移动到设备

        # 前向传播 
        activations = x                                       # 第一层输入为x
        for weight, bias in zip(weights[:-1], biases[:-1]):   # 迭代隐藏层的权重和偏置
            activations = relu(activations @ weight + bias)   # 计算线性变换并应用ReLU激活函数
        logits = activations @ weights[-1] + biases[-1]       # 计算输出层的线性变换
        preds = logits.argmax(dim=1)                          # 获取预测类别

        correct += (preds == y).sum().item()                  # 累加正确预测的数量
        total += y.size(0)                                    # 累加总样本数

    print(f"Test Accuracy: {correct / total*100:.2f}%")       # 打印测试集的准确率