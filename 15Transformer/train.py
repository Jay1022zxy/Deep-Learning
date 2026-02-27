import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from Transformer import build_transformer
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sp_en = spm.SentencePieceProcessor()
sp_en.load('14RNN_lstm/en2cn/en_bpe.model')   # 加载英文的SentencePiece模型，用于将英文文本转换为token id序列
sp_cn = spm.SentencePieceProcessor()
sp_cn.load('14RNN_lstm/en2cn/zh_bpe.model')   # 加载中文的SentencePiece模型，用于将中文文本转换为token id序列


def tokenize_en(text):
    return sp_en.encode(text, out_type=int)


def tokenize_cn(text):
    return sp_cn.encode(text, out_type=int)

# 中文和英文一致,取英文。
PAD_ID = sp_en.pad_id()  # 1
UNK_ID = sp_en.unk_id()  # 0
BOS_ID = sp_en.bos_id()  # 2
EOS_ID = sp_en.eos_id()  # 3

# 2. Dataset & DataLoader
class TranslationDataset(Dataset):
    ## 初始化方法，读取英文和中文训练文本。然后给每个句子前后增加<bos>和<eos>。 为了防止训练时显存不足，对于长度超过限制的
    ## 句子进行过滤。
    def __init__(self, src_file, trg_file, src_tokenizer, trg_tokenizer, max_len=64):  
        # 设置max_len为64，表示只保留输入和输出序列token数同时小于64的训练样本。这样可以防止训练时显存不足，同时也符合Transformer模型的输入要求。
        with open(src_file, encoding='utf-8') as f:
            src_lines = f.read().splitlines()
        with open(trg_file, encoding='utf-8') as f:
            trg_lines = f.read().splitlines()

        src_lines = src_lines[:100000]  # 取前10万行，防止训练时显存不足
        trg_lines = trg_lines[:100000]  # 取前10万行，防止训练时显存不足

        assert len(src_lines) == len(trg_lines)
        self.pairs = []                    # 用于保存处理后的输入和输出序列的token id列表
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        index = 0
        for src, trg in zip(src_lines, trg_lines):
            index += 1
            if index % 100000 == 0:
                print(index)          # 每10万行打印一次，观察数据加载进度
            # 每个句子前边增加<bos>后边增加<eos>
            src_ids = [BOS_ID] + self.src_tokenizer(src) + [EOS_ID]
            trg_ids = [BOS_ID] + self.trg_tokenizer(trg) + [EOS_ID]
            # 只保留输入和输出序列token数同时小于max_len的训练样本。
            if len(src_ids) <= max_len and len(trg_ids) <= max_len:
                self.pairs.append((src_ids, trg_ids))  # <-- 直接保存token id序列

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ids, trg_ids = self.pairs[idx]
        return torch.LongTensor(src_ids), torch.LongTensor(trg_ids)

    ## 对一个batch的输入和输出token序列，依照最长的序列长度，用<pad> token进行填充，确保一个batch的数据形状一致，组成一个tensor。
    @staticmethod
    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_lens = [len(x) for x in src_batch]
        trg_lens = [len(x) for x in trg_batch]
        ## 注意，Transformer里的tensor，设置batch_frist=True。
        src_pad = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
        trg_pad = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True,padding_value=PAD_ID)
        return src_pad, trg_pad, src_lens, trg_lens

# === 数据集定义 ===
def create_mask(src, tgt, pad_idx):     # 创建mask函数，接受输入序列、目标序列和<pad> token的索引，返回输入序列和目标序列的mask
    # mask <pad> token for encoder.
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
    # mask <pad> token for decoder.
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)

    tgt_len = tgt.size(1)
    # decoder mask 当前token后边的token。
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()  # (tgt_len, tgt_len)
    # decoder 同时mask <pad> token, 以及当前token后边的token。
    tgt_mask = tgt_pad_mask & tgt_sub_mask  # (batch, 1, tgt_len, tgt_len)
    return src_mask, tgt_mask

def train(model, dataloader, optimizer, criterion, pad_idx):
    model.train()
    total_loss = 0
    step = 0
    log_loss = 0  # 用于每100步统计

    for src, tgt, src_lens, tgt_lens in dataloader:
        step += 1

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask, tgt_mask = create_mask(src, tgt_input, pad_idx)

        optimizer.zero_grad()
        encoder_output = model.encode(src, src_mask)
        decoder_output = model.decode(encoder_output, src_mask, tgt_input, tgt_mask)
        output = model.project(decoder_output)

        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        log_loss += loss.item()

        if step % 100 == 0:
            avg_log_loss = log_loss / 100
            print(f"Step {step}: Avg Loss = {avg_log_loss:.4f}") 
            log_loss = 0  # 重置每100步的loss计数

    return total_loss / len(dataloader)

def main():
    # 超参数
    SRC_VOCAB_SIZE = 16000
    TGT_VOCAB_SIZE = 16000
    SRC_SEQ_LEN = 64     
    TGT_SEQ_LEN = 64
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LR = 8e-5

    # 数据集加载
    train_dataset = TranslationDataset('14RNN_lstm/en2cn/train_en.txt', '14RNN_lstm/en2cn/train_zh.txt',tokenize_en, tokenize_cn)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn)

    # 构建模型
    model = build_transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN).to(DEVICE)
    # 尝试加载已有模型，如果存在的话，这样可以继续之前的训练，而不是每次都从头开始训练。
    model_path = "15Transformer/transformer.pt"
    if os.path.exists(model_path):
        print("Loading existing model...")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("No saved model found, training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for epoch in range(NUM_EPOCHS):
        loss = train(model, train_dataloader, optimizer, criterion, PAD_ID)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss:.4f}")

        torch.save(model.state_dict(), model_path)
if __name__ == "__main__":
    main()

# 以下为截取前10万行数据，在3050ti上训练时，lr=2e-4,batch_size=16，seq_len=64，训练20轮的结果。
# Epoch 1/20 - Loss: 5.5160
# Epoch 2/20 - Loss: 4.7777
# Epoch 3/20 - Loss: 4.3692
# Epoch 4/20 - Loss: 4.0529
# Epoch 5/20 - Loss: 3.7852
# Epoch 6/20 - Loss: 3.5468
# Epoch 7/20 - Loss: 3.3274
# Epoch 8/20 - Loss: 3.1152
# Epoch 9/20 - Loss: 2.9119
# Epoch 10/20 - Loss: 2.7176
# Epoch 11/20 - Loss: 2.5288
# Epoch 12/20 - Loss: 2.3464
# Epoch 13/20 - Loss: 2.1758
# Epoch 14/20 - Loss: 2.0111
# Epoch 15/20 - Loss: 1.8614
# 发现step loss 在上升，初步推测是lr过大，所以将lr调整为8e-5，继续训练。
# Epoch 16/20 - Loss: 1.4596
# Epoch 17/20 - Loss: 1.2802
# Epoch 18/20 - Loss: 1.1691
# Epoch 19/20 - Loss: 1.0807
# Epoch 20/20 - Loss: 1.0031
# 调低学习率后，step loss反常上升有所改善，整体loss也继续下降，说明模型在继续学习。
# 但在loss达到1.1左右，仍有step loss上升的现象，说明模型在这个阶段可能遇到了局部最优或者过拟合的情况。
# 可能因为只截取100000条数据进行训练，导致模型在后期过拟合，或者学习率虽然调低了，但仍然不够小，无法稳定地继续下降。
# 可以考虑进一步降低学习率，或者增加正则化手段（如dropout）来缓解这个问题。同时，也可以观察验证集上的表现，看看是否存在过拟合的迹象。
