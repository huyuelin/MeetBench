import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import numpy as np
from tqdm import tqdm
import os

# 模型定义与路径
model_name = "hfl/chinese-roberta-wwm-ext"
data_path = '/home/leon/agent/classifier/qlora_finetune_data/qlora_finetune_input.jsonl'
output_dir = './finetuned_model'
os.makedirs(output_dir, exist_ok=True)

# 类别映射\
label2id = {'A': 0, 'B': 1}
id2label = {v: k for k, v in label2id.items()}

# 超参数
max_length = 128
batch_size = 8
epochs = 5
learning_rate = 2e-5

# 加载tokenizer和model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# 构建Dataset和DataLoader
class ClassifyDataset:
    def __init__(self, tokenizer, file, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                query = item['instruction'] + item['input']
                label = label2id[item['output'][3]]  # 提取标签A/B
                encoded = tokenizer(query, padding='max_length', truncation=True, max_length=self.max_length)
                self.samples.append((encoded, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoded, label = self.samples[idx]
        return (
            torch.tensor(encoded['input_ids']),
            torch.tensor(encoded['attention_mask']),
            torch.tensor(label)
        )

# 加载数据
dataset = ClassifyDataset(tokenizer, data_path, max_length)
sampler = RandomSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

# 训练
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    total_loss = 0
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 10 == 0:
            print(f"Epoch: {epoch+1}, Step: {step+1}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

# 保存微调模型
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"模型已保存到 {output_dir}")
