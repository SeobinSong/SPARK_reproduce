import torch
from spark_encode import spark_encode
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-SST-2",
    trust_remote_code=True,
    use_safetensors=True
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# SPARK rounding이 발생할 수 있는 구간 정의
spark_lossy_range = set()
for start in [16, 48, 80, 112, 128, 160, 192, 224]:
    spark_lossy_range.update(range(start, start + 16))  # 각 구간 16개씩

count_lossy_prone = 0
count_safe = 0

# 양자화 및 구간별 통계
for name, param in model.named_parameters():
    if "weight" in name and param.requires_grad:
        max_val = param.data.abs().max()

        # 1. symmetric 정량화
        scale = max_val / 127
        int8_tensor = torch.round(param.data / scale).clamp(-128, 127).to(torch.int)

        # 2. unsigned int8로 offset
        quantized_tensor = int8_tensor + 128

        for v in quantized_tensor.view(-1):
            if int(v) in spark_lossy_range:
                count_lossy_prone += 1
            else:
                count_safe += 1

# 결과 출력
total = count_lossy_prone + count_safe
print("✅ SPARK 손실 가능성 있는 파라미터 통계")
print(f"총 파라미터 수   : {total}")
print(f"Lossy 가능 구간  : {count_lossy_prone} ({100 * count_lossy_prone / total:.2f}%)")
print(f"Lossless 구간     : {count_safe} ({100 * count_safe / total:.2f}%)")
