import torch
from spark_encode import spark_encode
from spark_decode_update import spark_decode
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from collections import defaultdict


model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-SST-2",
    trust_remote_code=True,
    use_safetensors=True
)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

offset = 128
layer_buffers = defaultdict(list)  # layer별 파라미터 모으기

# 1. 파라미터 이름 기준으로 layer 단위로 묶기
for name, param in model.named_parameters():
    if param.requires_grad and ("weight" in name or "bias" in name):
        layer_name = ".".join(name.split(".")[:4])  # 예: 'bert.encoder.layer.0'
        layer_buffers[layer_name].append((name, param))

# 2. 각 layer 단위로 scale 계산 후 양자화 + SPARK 인코딩
for layer_name, params in layer_buffers.items():
    # 모든 텐서를 하나로 합쳐서 max 추출
    all_data = torch.cat([p.data.view(-1) for _, p in params])
    max_val = all_data.abs().max()
    scale = 127.5 / max_val

    for name, param in params:
        quantized_tensor = torch.round(param.data * scale + offset).clamp(0, 255).to(torch.int)

        decoded_vals = []
        for v in quantized_tensor.view(-1):
            encoded, _, _ = spark_encode(int(v))
            decoded = spark_decode(encoded)
            restored = (decoded - offset) / scale
            decoded_vals.append(restored)

        decoded_tensor = torch.tensor(decoded_vals).reshape(param.shape)
        param.data = decoded_tensor.float()

# ▽▽ 평가 파이프라인은 그대로 유지 ▽▽
dataset = load_dataset("glue", "sst2")
def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

training_args = TrainingArguments(output_dir="./spark_eval", per_device_eval_batch_size=64)
trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)

eval_result = trainer.evaluate(eval_dataset=encoded_dataset["validation"])
print(f"\n✅ SPARK (per-layer quant) 정확도: {eval_result['eval_accuracy']:.4f}")
