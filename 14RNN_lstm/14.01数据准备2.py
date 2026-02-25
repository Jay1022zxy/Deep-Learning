import sentencepiece as spm
# 利用词典模型进行分词
sp_cn = spm.SentencePieceProcessor()            # 创建一个SentencePieceProcessor对象
sp_cn.Load('14RNN_lstm/en2cn/zh_bpe.model')     # 加载中文BPE模型

text = "今天天气非常好。"                 # 待分词的中文文本

encode_result = sp_cn.encode(text, out_type=int)     # 将文本编码为对应的ID列表，out_type=int表示输出整数类型的ID
print("编码结果:", encode_result)                    # 输出分词结果

decode_result = sp_cn.decode(encode_result)         # 将分词结果解码回原始文本
print("解码结果:", decode_result)                    # 输出解码结果
