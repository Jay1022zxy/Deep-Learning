import torch
import sentencepiece as spm
from Transformer import build_transformer  # adjust import path as needed

# 1. 加载SentencePiece模型和定义特殊token ID
sp_en = spm.SentencePieceProcessor()
sp_en.load('14RNN_lstm/en2cn/en_bpe.model')
sp_cn = spm.SentencePieceProcessor()
sp_cn.load('14RNN_lstm/en2cn/zh_bpe.model')

PAD_ID = sp_en.pad_id()  # 1
UNK_ID = sp_en.unk_id()  # 0
BOS_ID = sp_en.bos_id()  # 2
EOS_ID = sp_en.eos_id()  # 3

# 2. 加载训练好的Transformer模型
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters (must match training)
SRC_VOCAB_SIZE = 16000
TGT_VOCAB_SIZE = 16000
SRC_SEQ_LEN = 64
TGT_SEQ_LEN = 64

model = build_transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN).to(DEVICE)
model.load_state_dict(torch.load('15Transformer/transformer.pt', map_location=DEVICE))  # load your trained model
model.eval()


# 3. 定义翻译函数
def create_mask(src, pad_idx):   # 创建mask函数，接受输入序列和<pad> token的索引，返回输入序列的mask
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (1, 1, 1, src_len)

def translate_sentence(sentence, max_len=100):
    # 将输入句子转换为token ID序列，并添加<bos>和<eos> token
    tokens = [BOS_ID] + sp_en.encode(sentence, out_type=int) + [EOS_ID]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)  # [1, src_len]
    src_mask = create_mask(src_tensor, PAD_ID)

    # 解码器输入初始化为<bos> token的ID
    trg_indices = [BOS_ID]

    with torch.no_grad():
        # 编码器前向传播，得到编码器输出
        encoder_output = model.encode(src_tensor, src_mask)

        # 循环生成目标序列，直到生成<eos> token或达到最大长度
        for _ in range(max_len):
            trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(DEVICE)  # [1, current_trg_len]

            # 创建目标序列的mask，使用下三角矩阵确保每个位置只能看到之前的位置
            trg_mask = torch.tril(torch.ones((len(trg_indices), len(trg_indices)), device=DEVICE)).bool()
            trg_mask = trg_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, trg_len, trg_len]

            # 解码器前向传播，得到当前目标序列的输出
            decoder_output = model.decode(encoder_output, src_mask, trg_tensor, trg_mask)
            output = model.project(decoder_output)

            # 获取当前时间步的预测结果，选择概率最高的token作为下一个输入
            pred_token = output.argmax(2)[:, -1].item()
            trg_indices.append(pred_token)

            if pred_token == EOS_ID:
                break

    # 将生成的token ID序列转换回文本，去掉<bos>和<eos> token
    translated = sp_cn.decode(trg_indices[1:-1])
    return translated


# 4. 交互式翻译
if __name__ == '__main__':
    print("Transformer Translator (type 'quit' or 'exit' to end)")
    while True:
        src_sent = input("\n请输入英文: ")
        if src_sent.lower() in ['quit', 'exit']:
            break
        translation = translate_sentence(src_sent)
        print(f"中文翻译为: {translation}")

# 以下为在3050ti上训练时，截取前10万行数据，训练20轮的结果。
# 请输入英文: hello,nice to meet you
# 中文翻译为: 一见面,你就发现
# 请输入英文: I love playing football
# 中文翻译为: 一直在仰望你
# 请输入英文: he is coming
# 中文翻译为: 一走了之
# 请输入英文: good morning
# 中文翻译为: 一大早唱
# 请输入英文: good afternoon
# 中文翻译为: 一清二楚
# 请输入英文: you
# 中文翻译为: 你
# 请输入英文: your
# 中文翻译为: 你的
# 请输入英文: I love you
# 中文翻译为: 说起你

# 表明代码逻辑正确，实际训练时可以使用全部数据，并且训练更多的epoch以获得更好的翻译质量。
