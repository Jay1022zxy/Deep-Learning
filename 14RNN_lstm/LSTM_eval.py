import torch
import torch.nn as nn
import sentencepiece as spm
from LSTM import Encoder, Decoder, Seq2Seq, Attention

# 1 加载 tokenizer
sp_en = spm.SentencePieceProcessor()
sp_en.load('14RNN_lstm/en2cn/spm_en.model')
sp_zh = spm.SentencePieceProcessor()
sp_zh.load('14RNN_lstm/en2cn/spm_zh.model')
PAD_ID = sp_en.pad_id()  # Padding token ID
BOS_ID = sp_en.bos_id()  # Beginning of Sequence token ID
EOS_ID = sp_en.eos_id()  # End of Sequence token ID

# 2 加载训练好的模型
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = sp_en.get_piece_size()   # 英文词表大小
OUTPUT_DIM = sp_zh.get_piece_size()  # 中文词表大小
ENC_EMB_DIM = 256                    # 编码器词嵌入维度 (必须和训练时保持一致)
DEC_EMB_DIM = 256                    # 解码器词嵌入维度
HID_DIM = 256                        # LSTM隐藏状态维度
N_LAYERS = 3

attention = Attention(HID_DIM).to(DEVICE)
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS).to(DEVICE)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS,attention).to(DEVICE)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
model.load_state_dict(torch.load('14RNN_lstm/en2cn/seq2seq_attention.pth', map_location=DEVICE))  # 加载训练好的模型参数
model.eval()  # 设置模型为评估模式

# 3 翻译方法
def translate_sentence(sentence,  max_len=100):
    # 将输入句子编码为token ID列表，并添加<bos>和<eos>标记
    tokens = [BOS_ID] + sp_en.encode(sentence,out_type=int) + [EOS_ID]   # 将输入句子编码为token ID列表，并在前后添加<bos>和<eos>标记
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)        # 将token ID列表转换为tensor，并添加batch维度
    src_len = [len(tokens)]                                              # 计算输入句子的长度，并转换为tensor

    # 调用Encoder获取编码器输出和初始隐状态
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    # 第一个输入token，序列上是<bos>标记
    trg_indexes = [BOS_ID]
    # 逐个token,循环调用Decoder进行翻译，直到生成<eos>标记或者达到最大长度
    for _ in range(max_len):
        # 最新生成的token ID作为Decoder的输入，获取下一个token的预测结果和新的隐状态
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(DEVICE)  # 获取当前输入token的ID，并转换为tensor
        with torch.no_grad():
            output, hidden, _ = model.decoder(trg_tensor, hidden, encoder_outputs,    # 调用Decoder获取下一个token的预测结果和新的隐状态，
                                              (src_tensor !=PAD_ID).permute(1, 0))    # 传入当前输入token、上一时刻的隐状态、编码器输出和源序列的mask

       # 获取预测结果中概率最高的token ID，并添加到生成的token列表中
        pred_token_id = output.argmax(1).item()    # 获取预测结果中概率最高的token ID
        trg_indexes.append(pred_token_id)          # 将预测的token ID添加到生成的token列表中
        if pred_token_id == EOS_ID:                # 如果生成了<eos>标记，停止翻译
            break

    # 将生成的token ID列表转换回文本，去掉<bos>和<eos>标记
    translated = sp_zh.decode(trg_indexes[1:-1])  # 将生成的token ID列表转换回文本，去掉<bos>和<eos>标记
    return translated

# 4 测试翻译
if __name__ == "__main__":
    while True:
        src_sentence = input("请输入英文句子进行翻译（输入exit退出）：")  # 提示用户输入英文句子
        if src_sentence.lower() == "exit":  # 如果用户输入exit，退出程序
            break
        translation = translate_sentence(src_sentence)  # 调用翻译方法获取翻译结果
        print(f"翻译结果：{translation}")  # 输出翻译结果
