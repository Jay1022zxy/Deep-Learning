# 在Pytorch里添加L2正则化
# 目的是通过惩罚模型的权重参数来防止过拟合
# loss_l2 = loss + lamda * sum(p.pow(2.0).sum() for p in model.parameters())
# l2_norm = 0.0
# for param in model.parameters():
#    l2_norm += param.pow(2).sum()
# loss = criterion(outputs, labels) + 1e-4 * l2_norm