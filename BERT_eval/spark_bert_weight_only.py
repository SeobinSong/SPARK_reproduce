import torch
from function.spark_encode import spark_encode
from spark_decode import spark_decode
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

# model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-SST-2",
    trust_remote_code=True,
    use_safetensors=True
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

scale_map = {}
offset = 128

for name, param in model.named_parameters(): # reference (not copy)
    if "weight" in name and param.requires_grad:
        # 1. float → UINT8
        max_val = param.data.abs().max()
        scale = 127.5 / max_val
        scale_map[name] = scale

        quantized_tensor = torch.round(param.data * scale + offset).clamp(0, 255).to(torch.int)

        # 2. SPARK 인코딩 + 디코딩
        decoded_vals = []
        for v in quantized_tensor.view(-1):
            encoded, _, _ = spark_encode(int(v))  # v는 0~255
            decoded = spark_decode(encoded)
            restored = (decoded - offset) / scale
            decoded_vals.append(restored)

        decoded_tensor = torch.tensor(decoded_vals).reshape(param.shape)
        param.data = decoded_tensor.float()

# 평가 데이터셋 로드 : sentiment pos/neg classification
dataset = load_dataset("glue", "sst2")
def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 정확도 metric
def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Trainer 설정
training_args = TrainingArguments(output_dir="./spark_eval", per_device_eval_batch_size=64)
trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)

# 평가 실행
eval_result = trainer.evaluate(eval_dataset=encoded_dataset["validation"])
print(f"\n✅ SPARK 디코딩 후 정확도: {eval_result['eval_accuracy']:.4f}")
