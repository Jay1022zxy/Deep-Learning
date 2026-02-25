import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

sp_en = spm.SentencePieceProcessor()        # 加载英文的BPE模型
sp_en.load('14RNN_lstm/en2cn/en_bpe.model')                  
sp_cn = spm.SentencePieceProcessor()        # 加载中文的BPE模型
sp_cn.load('14RNN_lstm/en2cn/zh_bpe.model')

# 定义英文和中文的分词函数，分别将文本转换为token ID列表。这里直接调用SentencePiece模型的encode方法，并指定输出类型为int（token ID）。
# 注意：英文和中文的BPE模型都使用了相同的特殊token ID（unk_id=0, pad_id=1, bos_id=2, eos_id=3），所以这里直接使用英文模型的特殊token ID定义PAD_ID、UNK_ID、BOS_ID和EOS_ID。
def tokenize_en(text):
    return sp_en.encode(text, out_type=int)

def tokenize_cn(text):
    return sp_cn.encode(text, out_type=int)

# 中文和英文一致,取英文。
PAD_ID = sp_en.pad_id()  # 1
UNK_ID = sp_en.unk_id()  # 0
BOS_ID = sp_en.bos_id()  # 2
EOS_ID = sp_en.eos_id()  # 3

class TranslationDataset(Dataset):
    # 初始化数据集，加载英文和中文的BPE编码数据,然后给每个句子前后增加<bos>和<eos>标记
    # 为了防止训练时显存不足，对于长度超过限制的句子进行过滤
    def __init__(self, src_file, trg_file,src_tokenizer,trg_tokenizer, max_length=100):
        with open(src_file, encoding='utf-8') as f:          # 打开英文数据文件
            src_lines = f.read().splitlines()               # 读取文件内容并按行分割成列表
        with open(trg_file, encoding='utf-8') as f:          # 打开中文数据文件
            trg_lines = f.read().splitlines()               # 读取文件内容并按行分割成列表
        assert len(src_lines) == len(trg_lines), "源语言和目标语言的句子数量不匹配！"  # 确保英文和中文句子数量一致
        self.pairs = []                                    # 初始化一个空列表用于存储英文和中文的句子对
        self.src_tokenizer = src_tokenizer                    # 存储英文分词器
        self.trg_tokenizer = trg_tokenizer                    # 存储中文分词器

        for src, trg in zip(src_lines, trg_lines):          # 遍历英文和中文句子列表
            # 给每个句子前后增加<bos>和<eos>标记
            src_ids = [BOS_ID] + self.src_tokenizer(src) + [EOS_ID]  # 将英文句子编码为ID列表，并在前后添加<bos>和<eos>标记
            trg_ids = [BOS_ID] + self.trg_tokenizer(trg) + [EOS_ID]  # 将中文句子编码为ID列表，并在前后添加<bos>和<eos>标记
            # 过滤掉长度超过限制的句子对
            if len(src_ids) <= max_length and len(trg_ids) <= max_length:
                self.pairs.append((src_ids, trg_ids))                # 将符合长度要求的句子对添加到列表中
        
    def __len__(self):
        return len(self.pairs)                                       # 返回数据集中句子对的数量
    
    def __getitem__(self, idx):
        src_ids, trg_ids = self.pairs[idx]                            # 获取指定索引处的英文和中文句子对
        return torch.LongTensor(src_ids), torch.LongTensor(trg_ids)   # 返回指定索引处的句子对
    
    # 对一个batch的输入和输出序列，依照最长的序列长度进行padding，保证每个batch中所有序列的长度一致，组成一个tensor
    @staticmethod          # 定义一个静态方法用于collate_fn函数，处理一个batch的数据
    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)                             # 将batch中的英文和中文句子对分开
        src_lens = [len(src) for src in src_batch]                     # 计算每个英文句子的长度
        trg_lens = [len(trg) for trg in trg_batch]                     # 计算每个中文句子的长度
        src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)  # 对英文句子进行padding，使用0作为填充标记
        trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=PAD_ID)  # 对中文句子进行padding，使用0作为填充标记
        return src_batch, trg_batch, src_lens, trg_lens                # 返回处理后的英文和中文句子批次
    
# 用DataLoader加载数据集，设置batch_size为8，并且使用上面定义的collate_fn函数进行批次处理
dataset = TranslationDataset('14RNN_lstm/en2cn/train_en.txt', '14RNN_lstm/en2cn/train_zh.txt',tokenize_en, tokenize_cn)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=TranslationDataset.collate_fn)
for src,trg, _, _ in dataloader:                      # 遍历DataLoader中的数据，获取每个批次的英文和中文句子批次
    print(src.shape, trg.shape)                       # 输出每个批次的英文和中文句子批次的形状
    print(src,trg)                                    # 输出每个批次的英文和中文句子批次的内容

    break                                             # 只查看第一个批次的数据，之后退出循环
