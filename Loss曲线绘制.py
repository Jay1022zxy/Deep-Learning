import torch

from torch.utils.tensorboard import SummaryWriter # 导入TensorBoard的SummaryWriter类
#用于记录训练过程中的数据，以便在TensorBoard中可视化

# 确定设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成数据
inputs = torch.rand(100,3) # 100个样本，每个样本3个特征,每个元素在0到1之间
weights = torch.tensor([[1.1],[4.5],[1.4]])  # 权重
bias = torch.tensor([2.2])  # 偏置
targets = inputs @ weights + bias + 0.1 * torch.rand(100,1) # 增加一些误差（噪声），模拟真实数据

# 创建SummaryWriter实例
writer = SummaryWriter(log_dir="E:/code/DeepLearning/logs") # 指定日志目录为"E:/code/Deep Learning/logs"

# 初始化参数直接放在CUDA上,并启用梯度追踪(使用随机值初始化权重和偏置)
w = torch.rand((3,1), device=device, requires_grad=True)  # 权重
b = torch.rand(1, device=device, requires_grad=True)  # 偏置

# 训练模型
# 将数据移动到相同设备
inputs = inputs.to(device)
targets = targets.to(device)
# 定义学习率和迭代次数
lr = 0.0035
epochs = 10000

for i in range(epochs):
    # 前向传播
    predictions = inputs @ w + b  # 线性模型
    # 计算损失（均方误差）
    loss = torch.mean((predictions - targets) ** 2)
    print("loss:", loss.item()) # 打印当前的损失值(loss.item()将张量转换为Python数值)
    
    # 将损失值记录到TensorBoard中,三个参数分别是标签、损失值和步数
    writer.add_scalar("Loss", loss.item(), i) # 将当前的损失值(loss.item())记录到TensorBoard中
    #标签为"Loss"，步数为i


    # 反向传播
    loss.backward()
    # 更新参数
    with torch.no_grad(): # 在更新参数时不需要计算梯度，否则会使计算图过大，导致内存占用过高
        w -= lr * w.grad
        b -= lr * b.grad

        # 清零梯度，因为PyTorch默认会累积梯度，如果不清零，下一次迭代的梯度会叠加在之前的基础上，导致参数更新不正确
        w.grad.zero_()
        b.grad.zero_()

print("训练后的权重:", w)
print("训练后的偏置:", b)