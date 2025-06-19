import torch
from spark_encode_with_correction import spark_encode_with_correction_bit
from spark_decode_with_correction import spark_decode_with_correction_bit
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

# 1. BERT 모델 로드 (SST-2용)
model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-SST-2",
    trust_remote_code=True,
    use_safetensors=True
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
scale_map = {}

# 2. weight 양자화 + SPARK 인코딩 + 디코딩 (보정 포함)
for name, param in model.named_parameters():
    if "weight" in name and param.requires_grad:
        max_val = param.data.abs().max()
        scale = 127 / max_val
        scale_map[name] = scale

        # float → int8 양자화
        int8_tensor = torch.round(param.data * scale).clamp(-128, 127).to(torch.int8)

        # SPARK 인코딩 + 보정 디코딩
        decoded_vals = []
        for v in int8_tensor.view(-1):
            val = int(v.item())
            sign = 1 if val >= 0 else -1
            abs_val = abs(val)

            # 인코딩 (보정 비트 포함) → 디코딩 (보정 반영)
            encoded, _, _ = spark_encode_with_correction_bit(abs_val)
            decoded_abs = spark_decode_with_correction_bit(encoded)

            restored = sign * decoded_abs / scale  # 부호 복원
            decoded_vals.append(restored)

        # 디코딩된 weight로 갱신
        decoded_tensor = torch.tensor(decoded_vals).reshape(param.shape)
        param.data = decoded_tensor.float()

# 3. SST-2 데이터셋 로드 및 전처리
dataset = load_dataset("glue", "sst2")

def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 4. 정확도 metric
def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# 5. Trainer 설정
training_args = TrainingArguments(output_dir="./spark_eval_corrected", per_device_eval_batch_size=64)
trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)

# 6. 평가 실행
eval_result = trainer.evaluate(eval_dataset=encoded_dataset["validation"])
print(f"\n✅ SPARK(보정 비트 포함) 디코딩 후 정확도: {eval_result['eval_accuracy']:.4f}")
