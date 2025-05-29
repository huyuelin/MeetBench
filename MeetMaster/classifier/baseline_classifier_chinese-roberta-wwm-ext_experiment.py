import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# 模型及类别映射定义
model_name = "hfl/chinese-roberta-wwm-ext"
label2id = {'A': 0, 'B': 1}
id2label = {v: k for k, v in label2id.items()}

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# Prompt设计
def build_prompt(query):
    return f"请判断以下问题的复杂程度并给出类别A、B。如果是能靠LLM直接回复的简单问题请输出A，如果是需要多步推理的复杂问题请输出B。\n问题：{query}\n类别："

# 推理函数
def classify_query(query):
    prompt = build_prompt(query)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_label = id2label[torch.argmax(probs).item()]
    confidence = torch.max(probs).item()
    return pred_label, confidence

# GT文件路径
# gt_paths = [
#     '/home/leon/agent/AISHELL_dataset/train_L/gpt_4o_ground_truth.jsonl',
#     '/home/leon/agent/AISHELL_dataset/train_S/gpt_4o_ground_truth_train_S.jsonl'
# ]

gt_paths = [
    '/home/leon/agent/AISHELL_dataset/train_M/gpt_4o_ground_truth_train_M.jsonl'
]

# 加载GT数据
data = []
for path in gt_paths:
    with open(path, 'r', encoding='utf-8') as f:
        data.extend([json.loads(line) for line in f])

# 分类并评估
correct = 0
results = []
for entry in data:
    query = entry['question']
    true_label = 'A' if entry['question_type'] == 'simple' else 'B'
    pred_label, confidence = classify_query(query)
    print(f"问题：「{query}」")
    print(f"预测类别：{pred_label}（置信度：{confidence:.2f}）")
    print(f"真实类别：{true_label}")
        
    if pred_label == true_label:
        correct += 1
    
    results.append({
        "query": query,
        "true_label": true_label,
        "predicted_label": pred_label,
        "confidence": confidence
    })

# 统计准确率
accuracy = correct / len(data)
print(f"分类准确数：{correct}/{len(data)}，准确率：{accuracy:.2%}")

# 结果输出路径
output_dir = Path('/home/leon/agent/classifier/roberta_classification_results')
os.makedirs(output_dir, exist_ok=True)

output_path = output_dir / 'baseline_roberta_classification_results.jsonl'
with open(output_path, 'w', encoding='utf-8') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"结果已保存到 {output_path}")
