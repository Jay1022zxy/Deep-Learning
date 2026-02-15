X = [[10,3],[20,3],[25,3],[28,2.5],[30,2],[35,2.5],[40,2.5]] # 温度x1 价格x2 feature
y = [60,85,100,120,140,145,163] #销量y label
mid_y = sum(y[i] for i in range(len(y)))/len(y) # 销量的均值
w = [0.0,0.0,0.0] # 参数 w0,w1,w2
# y_hat =w0 + w1x1 + w2x2
lr = 0.0001 # 学习率
count = 1000 #迭代次数

# 梯度下降
for i in range(count):
    # 预测值
    y_pre = [w[0] + w[1] * x[0] + w[2] * x[1] for x in X]
    # 计算损失
    loss = sum((y_pre[j] - y[j]) ** 2 for j in range(len(y)) ) / len(y)
    # 计算梯度
    grad_w0 = 2 * (sum(y_pre[j] - y[j] for j in range(len(y))) / len(y))
    grad_w1 = 2 * (sum((y_pre[j] - y[j]) * X[j][0] for j in range(len(y))) / len(y))
    grad_w2 = 2 * (sum((y_pre[j] - y[j]) * X[j][1] for j in range(len(y))) / len(y))
    # 更新参数
    w[0] -= lr * grad_w0
    w[1] -= lr * grad_w1
    w[2] -= lr * grad_w2
    # 每100次输出一次损失
    if i % 100 == 0:
        print(f"现在是第{i}次,loss为:{loss}")
# 输出最终参数
print(f"最终参数为: w0 = {w[0]}, w1 = {w[1]}, w2 = {w[2]}")


