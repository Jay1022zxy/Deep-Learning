import sacrebleu
# 单个参考译文列表
references = [
    "今天天气很好。",
    "我喜欢在雨天散步。",
    "今天要早点下班。"
]

# 模型生成的候选译文列表
hypotheses = [
    "今日天气不错。",
    "下雨的时候我喜欢散步。",
    "今天下班要早些。"
]

# sacreBLEU计算需要 references 是 list[list[str]] 的格式(即每个参考译文是一个列表)，hypotheses 是 list[str] 的格式
# 即使只有一个参考译文，也需要将其包装成一个列表
references = [references]            # 变成:[[ref1, ref2, ref3]]

# 计算BLEU分数
bleu = sacrebleu.corpus_bleu(hypotheses, references, tokenize="zh")

print(f"BLEU score: {bleu.score:.2f}%")
