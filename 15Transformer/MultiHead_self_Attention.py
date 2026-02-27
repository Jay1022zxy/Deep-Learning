import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, h : int, dropout: float ): # d_model: 模型维度, h: 注意力头数, dropout: dropout概率
        super().__init__()
        self.d_model = d_model  # 模型维度(embedding维度)
        self.h = h              # 注意力头数
        # 确保模型维度可以被注意力头数整除
        assert d_model % h == 0, "d_model 不能被 h 整除"

        self.d_k = d_model // h  # 每个注意力头的维度

        self.w_q = nn.Linear(d_model, d_model,bias=False)  # 对每个输入向量进行线性变换以生成Q的权重矩阵
        self.w_k = nn.Linear(d_model, d_model,bias=False)  # 对每个输入向量进行线性变换以生成K的权重矩阵
        self.w_v = nn.Linear(d_model, d_model,bias=False)  # 对每个输入向量进行线性变换以生成V的权重矩阵

        self.w_o = nn.Linear(d_model, d_model,bias=False)  # 对最终attention拼接向量进行线性变换的权重矩阵
        self.dropout = nn.Dropout(dropout)                 # dropout层

    @staticmethod     # @staticmethod装饰器表示该方法是一个静态方法，不需要访问类的实例或类本身
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]  # 获取每个注意力头的维度
        # (batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, seq_len)
        # 多维矩阵乘法，只在最后两个维度上进行矩阵乘法，其他维度不变
        attention_scores = (query @ key.transpose(-2,-1))/ math.sqrt(d_k)  # 计算注意力分数，并进行缩放
        if mask is not None:   # 用于标记哪些位置是padding或者未来的token，这些位置在计算注意力分数时应该被忽略
            # 给mask中为0的位置设置一个非常大的负数，使得softmax后这些位置的权重接近于0
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        # (batch_size, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)  # 对注意力分数进行softmax，得到注意力权重,dim=-1表示在最后一个维度上进行softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)  # 应用dropout
        # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores  # 将注意力权重与value矩阵相乘，得到加权后的输出
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # 对输入的q进行线性变换，得到query矩阵
        key = self.w_k(k)    # 对输入的k进行线性变换，得到key矩阵
        value = self.w_v(v)  # 对输入的v进行线性变换，得到value矩阵

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        # 将query、key、value矩阵分成h个头，每个头的维度是d_k，并调整维度顺序以便后续计算(多头拆分)
        query = query.view(query.size(0), -1, self.h, self.d_k).transpose(1, 2)  # 将query矩阵分成h个头，并调整维度顺序
        key = key.view(key.size(0), -1, self.h, self.d_k).transpose(1, 2)        # 将key矩阵分成h个头，并调整维度顺序
        value = value.view(value.size(0), -1, self.h, self.d_k).transpose(1, 2)  # 将value矩阵分成h个头，并调整维度顺序

        # 计算多头注意力
        x, attention_scores = self.attention(query, key, value, mask, self.dropout)  # 计算多头注意力，得到加权后的输出和注意力权重

        # 多个注意力头的输出拼接在一起，并通过线性变换得到最终的输出
        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)  # 将多个注意力头的输出拼接在一起，并调整维度顺序

        # 对拼接后的输出进行线性变换，得到最终的输出
        return self.w_o(x)  
