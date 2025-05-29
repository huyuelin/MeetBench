import json
import os
from pathlib import Path

# 原始分类结果文件
input_path = '/home/leon/agent/classifier/roberta_classification_results/baseline_classification_results.jsonl'

# QLoRA微调数据输出路径
output_dir = Path('/home/leon/agent/classifier/qlora_finetune_data')
os.makedirs(output_dir, exist_ok=True)
output_path = output_dir / 'qlora_finetune_input.jsonl'

# 构建QLoRA训练数据的prompt模板
def build_qlora_entry(query, label):
    label_desc = '能靠LLM直接回复的简单问题' if label == 'A' else '需要多步推理的复杂问题'
    return {
        "instruction": "请判断以下问题的复杂程度并给出类别A、B。",
        "input": f"问题：{query}",
        "output": f"类别：{label}（{label_desc}）"
    }

# 转换数据并保存
with open(input_path, 'r', encoding='utf-8') as f_in, \
     open(output_path, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        entry = json.loads(line.strip())
        qlora_entry = build_qlora_entry(entry['query'], entry['true_label'])
        f_out.write(json.dumps(qlora_entry, ensure_ascii=False) + '\n')

print(f"QLoRA微调输入数据已保存到 {output_path}")
