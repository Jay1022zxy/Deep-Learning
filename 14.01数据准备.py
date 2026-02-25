import sentencepiece as spm
# 训练英文和中文的BPE模型
spm.SentencePieceTrainer.Train('--input=14RNN_lstm/en2cn/train_en.txt --model_prefix=en_bpe --vocab_size=16000 --model_type=bpe ' \
                               '--character_coverage=1.0 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3')
# character_coverage参数设置为1.0，因为英文单个字符的覆盖率较高，可以覆盖所有字符。
spm.SentencePieceTrainer.Train('--input=14RNN_lstm/en2cn/train_zh.txt --model_prefix=zh_bpe --vocab_size=16000 --model_type=bpe ' \
                               '--character_coverage=0.9995 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3' \
                               '--input_sentence_size=1000000 --shuffle_input_sentence=true')  # 只采样100万行数据进行训练，减少训练时间，并且打乱输入数据，增加模型的泛化能力。
# character_coverage参数设置为0.9995，表示覆盖99.95%的字符，这样可以更好地处理中文文本中的特殊字符和罕见字符。