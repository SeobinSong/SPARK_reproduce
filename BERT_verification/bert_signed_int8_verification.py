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

log_lines = []

for name, param in model.named_parameters():
    if "weight" in name and param.requires_grad:
        max_val = param.data.abs().max()
        scale = 127 / max_val

        quantized = torch.round(param.data * scale).clamp(-128, 127).to(torch.int8)
        dequantized = (quantized.float()) / scale

        qmin, qmax = quantized.min().item(), quantized.max().item()
        error = (param.data - dequantized).abs()
        mean_err = error.mean().item()
        max_err = error.max().item()

        log_lines.append(f"[{name}] Quantized range: [{qmin}, {qmax}]")
        log_lines.append(f"[{name}] Quantization error: mean={mean_err:.6f}, max={max_err:.6f}")
        log_lines.append("")  # 줄바꿈

        param.data = dequantized

# 저장
with open("quantization_report.txt", "w") as f:
    f.write("\n".join(log_lines))

print("📄 quantization_report.txt 저장 완료!")

