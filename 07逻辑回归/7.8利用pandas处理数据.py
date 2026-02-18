# 使用pandas库进行数据预处理
import pandas as pd
pd.set_option('display.max_columns', None)#显示所有列
# 从csv文件中读取数据
df = pd.read_csv(r"E:\code\DeepLearning\07逻辑回归\titanic\train.csv")
# 去除不必要的列
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
# 去除age列中的缺失值
df = df.dropna(subset=['Age'])
# 对Sex和Embarked列进行独热编码
df = pd.get_dummies(df, columns=['Sex', 'Embarked'],dtype=int)
print(df.head(10))
