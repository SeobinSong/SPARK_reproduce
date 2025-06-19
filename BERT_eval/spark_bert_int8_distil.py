from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from spark_encode import spark_encode
from spark_decode import spark_decode
import torch

# ✅ INT8 양자화된 모델 불러오기
model = AutoModelForSequenceClassification.from_pretrained(
    "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static-inc",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

offset = 128
scale_map = {}

# ✅ SPARK encoding + decoding 적용
for name, param in model.named_parameters():
    if "weight" in name and param.requires_grad and param.dtype == torch.float32:
        max_val = param.data.abs().max()
        scale = 127.5 / max_val
        scale_map[name] = scale

        quantized = torch.round(param.data * scale + offset).clamp(0, 255).to(torch.int)

        decoded_vals = []
        for v in quantized.view(-1):
            encoded, _, _ = spark_encode(int(v))
            decoded = spark_decode(encoded)
            restored = (decoded - offset) / scale
            decoded_vals.append(restored)

        param.data = torch.tensor(decoded_vals).reshape(param.shape).float()

# ✅ SST-2 평가
dataset = load_dataset("glue", "sst2")
def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

training_args = TrainingArguments(output_dir="./spark_int8_eval", per_device_eval_batch_size=64)
trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)

eval_result = trainer.evaluate(eval_dataset=encoded_dataset["validation"])
print(f"\n✅ INT8 모델 + SPARK 적용 후 정확도: {eval_result['eval_accuracy']:.4f}")
