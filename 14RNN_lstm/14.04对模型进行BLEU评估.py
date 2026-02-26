import sacrebleu
from LSTM_eval import translate_sentence

# 读取验证集的英文原文和中文参考
with open('14RNN_lstm/en2cn/valid_en.txt', 'r', encoding='utf-8') as f:
    src_sentences = [line.strip() for line in f.readlines()]

with open('14RNN_lstm/en2cn/valid_zh.txt', 'r', encoding='utf-8') as f:
    ref_sentences = [line.strip() for line in f.readlines()]

# 检查长度是否匹配
assert len(src_sentences) == len(ref_sentences), "源语言和参考翻译句子数不匹配"

# 用模型生成翻译
hypotheses = []
for i, src in enumerate(src_sentences):
    print(f"Translating {i+1}/{len(src_sentences)}...")
    translation = translate_sentence(src)
    print(ref_sentences[i], translation)
    hypotheses.append(translation.strip())

# 计算 BLEU
bleu = sacrebleu.corpus_bleu(hypotheses, [ref_sentences], tokenize='zh')

print("\n========== BLEU Evaluation Result ==========")
print(f"BLEU Score: {bleu.score:.2f}")

# 受限于配置，以下为截取前100000条数据，5个epochs的结果
# ========== BLEU Evaluation Result ==========
# BLEU Score: 3.33
# 可以表明代码逻辑正确，实际训练时可以使用全部数据，并且训练更多的epoch以获得更好的翻译质量和更高的BLEU分数。
