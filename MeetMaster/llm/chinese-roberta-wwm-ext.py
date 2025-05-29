from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 模型及类别映射定义
model_name = "hfl/chinese-roberta-wwm-ext"
label2id = {'A': 0, 'B': 1}
id2label = {v: k for k, v in label2id.items()}

# 加载tokenizer和预训练分类模型（假设你已fine-tune完成）
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# 初步prompt设计（提高模型判断的语境理解）
def build_prompt(query):
    prompt = f"请判断以下问题的复杂程度并给出类别A、B。如果是能靠LLM直接回复的简单问题请输出A，如果是需要多步推理的复杂问题请输出B。\n问题：{query}\n类别："
    return prompt

# 推理函数，输入问题query，输出label
def classify_query(query):
    prompt = build_prompt(query)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_label = id2label[torch.argmax(probs).item()]
    confidence = torch.max(probs).item()
    return pred_label, confidence

# 示例query测试
query_example = "你好交交，如何有效管理团队沟通？"
predicted_label, confidence = classify_query(query_example)

print(f"问题：「{query_example}」")
print(f"预测类别：{predicted_label}（置信度：{confidence:.2f}）")
