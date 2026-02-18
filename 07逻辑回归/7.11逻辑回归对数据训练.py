from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd 
import torch.nn as nn

class LogisticRegression(nn.Module):   # 定义自己的逻辑回归模型, 继承自nn.Module
    def __init__(self, input_dim):     # 实现init方法
        super(LogisticRegression, self).__init__()
        # nn.Linear也继承自nn.Module, 输入为input_dim, 输出一个值
        self.linear = nn.Linear(input_dim, 1)  # 定义一个线性层，输入维度为input_dim，输出维度为1

    def forward(self, x):  # logistic regression 输出一个概率值, 因此需要经过sigmoid函数
        return torch.sigmoid(self.linear(x))  # 前向传播，输出经过sigmoid函数的结果

class TitanicDataset(Dataset):     # 定义一个继承自Dataset的类, 用于加载和处理泰坦尼克号数据集
    def __init__(self, file_path):
        self.file_path = file_path # 保存文件路径
        self.data = self._load_data() # 加载数据
        self.feature_size = len(self.data.columns) - 1 # 特征数量（除去标签列）

    def _load_data(self):
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"]) ##删除不用的列
        df = df.dropna(subset=["Age"])                                   ##删除Age有缺失的行
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)  ##进行one-hot编码

        base_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        self.mean = df[base_features].mean()  # 计算基础特征的均值
        self.std = df[base_features].std()    # 计算基础特征的标准差
        # 对基础特征进行标准化
        for i in range(len(base_features)):   
            df[base_features[i]] = (df[base_features[i]] - self.mean[base_features[i]]) / self.std[base_features[i]] 
        
        # 因为精度不够，增加一些二次项
        for i in range(len(base_features)):
            for j in range(i, len(base_features)):    
                # 计算二次项的均值和标准差
                # 这里的二次项是指基础特征之间的乘积, 例如Pclass和Age的二次项就是Pclass_Age = Pclass * Age
                f1 = base_features[i]
                f2 = base_features[j]
                new_feature = f"{f1}_{f2}"
                df[new_feature] = df[f1] * df[f2]
                self.mean[new_feature] = df[new_feature].mean()
                self.std[new_feature] = df[new_feature].std()
                                                                   
        return df

    def __len__(self):                      # 返回数据集的大小, 方便确定迭代的次数
        return len(self.data)

    def __getitem__(self, idx):             # 根据索引获取数据, 方便DataLoader进行批处理
        features = self.data.drop(columns=["Survived"]).iloc[idx].values
        label = self.data["Survived"].iloc[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# 定义训练集
train_dataset = TitanicDataset(r"E:\code\DeepLearning\07逻辑回归\titanic\train.csv") 
# 定义验证集 
val_dataset = TitanicDataset(r"E:\code\DeepLearning\07逻辑回归\titanic\val.csv")

# 定义模型，并移动到GPU
model = LogisticRegression(train_dataset.feature_size)
model.to("cuda")
model.train()            # 设置模型为训练模式(部分模型在训练和评估模式下行为不同, 例如dropout和batchnorm)
                         # 但是对于逻辑回归模型来说, 训练和评估模式下的行为是一样的
optimizer = torch.optim.SGD(model.parameters(), lr=0.03) # 定义优化器, 这里使用随机梯度下降(SGD), 学习率为0.03
epochs = 120 # 定义迭代次数

for epoch in range(epochs): # 迭代训练
    correct = 0         # 初始化正确预测的数量
    step = 0            # 初始化步数
    total_loss = 0.0    # 初始化总损失
    for features, labels in DataLoader(train_dataset, batch_size=256, shuffle=True):# 使用DataLoader加载训练数据, 设置批大小为256, 并启用数据打乱
        step += 1 
        features = features.to("cuda") # 将输入数据移动到GPU
        labels = labels.to("cuda")     # 将标签数据移动到GPU
        optimizer.zero_grad()          # 清空旧梯度
        outputs = model(features).squeeze(1)  # 前向传播, 由于输出是一个二维张量, 需要使用squeeze(1)将其变为一维张量
        correct += torch.sum((outputs >= 0.5) == labels).item() # 计算正确预测的数量, 输出大于等于0.5的预测为1, 否则为0   
                                                 # 这里item()方法将张量转换为Python数值, 以便进行累加
        loss = nn.functional.binary_cross_entropy(outputs, labels) # 计算二元交叉熵损失(BCE)
        total_loss += loss.item()  # 累加损失, 这里item()方法将张量转换为Python数值
        loss.backward()            # 反向传播
        optimizer.step()           # 更新模型参数
    print(f"Epoch {epoch+1}, Loss: {total_loss/step:.4f}") # 打印每个epoch的平均损失, 这里total_loss/step计算平均损失
    print(f"Accuracy: {correct/len(train_dataset)}")   # 打印每个epoch的准确率, 这里correct/len(train_dataset)计算准确率

model.eval() # 设置模型为评估模式
with torch.no_grad(): # 在评估模式下, 不需要计算梯度, 因此使用torch.no_grad()上下文管理器来禁用梯度计算, 以节省内存和计算资源
    correct = 0       
    for features, labels in DataLoader(val_dataset, batch_size=256, shuffle=False): 
        features = features.to("cuda") 
        labels = labels.to("cuda")     
        outputs = model(features).squeeze(1)  
        correct += torch.sum((outputs >= 0.5) == labels).item() 
    print(f"Validation Accuracy: {correct/len(val_dataset)}") # 打印验证集的准确率


# lr = 0.1 , epoch = 100, Loss: 0.4148 ,Accuracy: 0.8163265306122449 ,Validation Accuracy: 0.8571428571428571
# lr = 0.1 , epoch = 120, Loss: 0.4108 ,Accuracy: 0.8181818181818182 ,Validation Accuracy: 0.8571428571428571
# lr = 0.03, epoch = 120, Loss: 0.4630 ,Accuracy: 0.8006279434850864 ,Validation Accuracy: 0.8831168831168831 

# 说明lr = 0.1时偏大 ，loss小说明可能出现过拟合，而lr = 0.03时虽然loss较大，但验证集准确率更高，说明减少了过拟合现象，模型在验证集上的表现更好。