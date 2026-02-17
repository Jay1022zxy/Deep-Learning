import torch
x = torch.tensor(1.0, requires_grad=True) #创建一个标量张量，并设置requires_grad=True以便后续计算梯度
y = torch.tensor(1.0, requires_grad=True) #同样创建另一个标量张量
v = 3*x + 4*y
u = torch.square(v)
z = torch.log(u)

z.backward()#反向传播求梯度

print("x.grad:", x.grad) #输出x的梯度
print("y.grad:", y.grad) #输出y的梯度
