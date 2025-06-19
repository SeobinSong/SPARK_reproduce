import torch
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

########################################
# BERT 모델 signed INT8 양자화 및 정확도 평가 (SPARK X)
########################################

# 1. 사전학습된 모델 로드 (SST-2 분류용)
model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-SST-2",
    trust_remote_code=True,
    use_safetensors=True
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 2. 양자화 적용 (signed INT8)
scale_map = {}

for name, param in model.named_parameters():
    if "weight" in name and param.requires_grad:
        max_val = param.data.abs().max()
        scale = 127 / max_val  # signed INT8은 -128 ~ 127
        scale_map[name] = scale

        # float → int8 양자화
        quantized = torch.round(param.data * scale).clamp(-128, 127).to(torch.int8)

        # int8 → float 복원
        dequantized = (quantized.float()) / scale
        param.data = dequantized

# 3. SST-2 데이터셋 로드
dataset = load_dataset("glue", "sst2")

def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 4. 정확도 metric
def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# 5. Trainer 설정 및 평가
training_args = TrainingArguments(output_dir="./int8_eval", per_device_eval_batch_size=64)
trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)

# 6. 정확도 평가
eval_result = trainer.evaluate(eval_dataset=encoded_dataset["validation"])
print(f"\n✅ INT8 Quantization 정확도: {eval_result['eval_accuracy']:.4f}")
