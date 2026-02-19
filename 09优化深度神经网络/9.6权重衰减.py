# 通过在损失函数中添加权重衰减项来实现权重衰减
# 目的是通过惩罚模型的权重参数来防止过拟合
# pytorch中可以通过在优化器中设置weight_decay参数来实现权重衰减
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
# weight_decay = 0.001 就是权重衰减的系数 lambda