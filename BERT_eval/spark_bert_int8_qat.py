import torch
from spark_encode import spark_encode
from spark_decode import spark_decode
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

# ✅ 1. BERT 모델 로드 (QAT 모델)
model = AutoModelForSequenceClassification.from_pretrained(
    "yujiepan/bert-base-uncased-sst2-int8-unstructured80-30epoch"
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ✅ 2. 모델 weight에 SPARK 인코딩/디코딩 적용
scale_map = {}
offset = 128  # zero-point 기준 (중심값)

for name, param in model.named_parameters():
    if "weight" in name and param.requires_grad:
        max_val = param.data.abs().max()
        scale = 127.5 / max_val
        scale_map[name] = scale

        # 양자화
        quantized_tensor = torch.round(param.data * scale + offset).clamp(0, 255).to(torch.int)

        # SPARK 인코딩 후 디코딩 → float 복원
        flat_tensor = quantized_tensor.view(-1)
        restored_tensor = torch.empty_like(flat_tensor, dtype=torch.float32)

        for i in range(flat_tensor.size(0)):
            encoded, _, _ = spark_encode(int(flat_tensor[i].item()))
            decoded = spark_decode(encoded)
            restored_tensor[i] = (decoded - offset) / scale

        param.data = restored_tensor.view(param.shape)

# ✅ 3. SST-2 데이터셋 로드 및 전처리
dataset = load_dataset("glue", "sst2")

def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ✅ 4. 정확도 metric 정의
def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# ✅ 5. Trainer 설정 및 평가 실행
training_args = TrainingArguments(output_dir="./spark_eval", per_device_eval_batch_size=64)
trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)

eval_result = trainer.evaluate(eval_dataset=encoded_dataset["validation"])
print(f"\n✅ SPARK 디코딩 후 정확도: {eval_result['eval_accuracy']:.4f}")
