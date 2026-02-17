import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# lights distance
inputs = torch.tensor([[2,1000],[3,2000],[2,500],[1,800],[4,3000]], dtype=torch.float,device=device) 
# time     time = 2*lights + 0.01*distance + 5
labels = torch.tensor([[19],[31],[14],[15],[43]], dtype=torch.float,device=device) 

# 进行归一化(将每列除以对应的最大值进行归一化,但最大值可能是异常值)
#inputs = inputs / torch.tensor([4, 3000], device=device)  # 将每列除以对应的最大值进行归一化

#计算均值和标准差
mean = inputs.mean(dim=0)  # 计算每列的均值
std = inputs.std(dim=0)    # 计算每列的标准差
#进行标准化(会考虑所有数据的分布情况,避免受异常值的影响)
inputs = (inputs - mean) / std  # 对每列进行标准化

# 初始化权重和偏置
w = torch.ones(2, 1, device=device, requires_grad=True)  # 权重
b = torch.ones(1, device=device, requires_grad=True)  # 偏置

# 训练模型 
# 将数据移动到相同设备
inputs = inputs.to(device)
labels = labels.to(device)
# 定义学习率和迭代次数
lr = 0.1
epochs = 2000   

for i in range(epochs):
    # 前向传播
    predictions = inputs @ w + b  # 线性模型
    # 计算损失（均方误差）
    loss = torch.mean((predictions - labels) ** 2)
    print("loss:", loss.item()) # 打印当前的损失值(loss.item()将张量转换为Python数值)
    # 反向传播
    loss.backward()
    print("w.grad:", w.grad.tolist()) # 打印权重的梯度
    # 更新参数
    with torch.no_grad(): # 在更新参数时不需要计算梯度，否则会使计算图过大，导致内存占用过高
        w -= lr * w.grad
        b -= lr * b.grad

        # 清零梯度，因为PyTorch默认会累积梯度，如果不清零，下一次迭代的梯度会叠加在之前的基础上，导致参数更新不正确
        w.grad.zero_()
        b.grad.zero_()

# 对新采集的数据进行预测
new_inputs = torch.tensor([[3, 2500]], dtype=torch.float, device=device)  
# 进行相同的标准化处理
new_inputs = (new_inputs - mean) / std  # 对新输入数据进行标准化
# 进行预测
predicted_time = new_inputs @ w + b
# 输出预测结果
print("预测的时间:", predicted_time.item()) # 将预测结果转换为Python数值并打印