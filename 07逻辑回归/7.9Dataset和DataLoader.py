import pandas as pd
from torch.utils.data import Dataset, DataLoader 
import torch

class TitanicDataset(Dataset):     # 定义一个继承自Dataset的类, 用于加载和处理泰坦尼克号数据集
    def __init__(self, file_path):
        self.file_path = file_path # 保存文件路径
        self.data = self._load_data() # 加载数据
        self.feature_size = len(self.data.columns) - 1 # 特征数量（除去标签列）

    def _load_data(self):
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"]) ##删除不用的列
        df = df.dropna(subset=["Age"])##删除Age有缺失的行
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)##进行one-hot编码

        base_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        self.mean = df[base_features].mean()  # 计算基础特征的均值
        self.std = df[base_features].std()    # 计算基础特征的标准差
        for i in range(len(base_features)):   # 对基础特征进行标准化
            df[base_features[i]] = (df[base_features[i]] - self.mean[base_features[i]]) / self.std[base_features[i]]
        return df

    def __len__(self):                      # 返回数据集的大小, 方便确定迭代的次数
        return len(self.data)

    def __getitem__(self, idx):             # 根据索引获取数据, 方便DataLoader进行批处理
        features = self.data.drop(columns=["Survived"]).iloc[idx].values
        label = self.data["Survived"].iloc[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

dataset = TitanicDataset(r"E:\code\DeepLearning\07逻辑回归\titanic\train.csv") # 创建数据集实例
dataloader = DataLoader(dataset, batch_size=256, shuffle=True) # 创建DataLoader实例, 设置批大小为256, 并启用数据打乱
for inputs, labels in dataloader: # 迭代DataLoader获取批次数据
    print(inputs.shape, labels.shape) # 打印输入和标签的形状
    break 