import torch
from spark_encode import spark_encode
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

lossless = 0
lossy = 0
lossy_list = []

for name, param in model.named_parameters():
    if "weight" in name and param.requires_grad:
        max_val = param.data.abs().max()
        scale = 127.5 / max_val
        quantized_tensor = torch.round(param.data * scale + offset).clamp(0, 255).to(torch.int)

        for v in quantized_tensor.view(-1):
            v_int = int(v)
            encoded, _, _ = spark_encode(v_int)
            decoded = spark_decode(encoded)

            if v_int == decoded:
                lossless += 1
            else:
                lossy += 1
                # 디버깅 정보만 따로 저장
                lossy_list.append((v_int, decoded, abs(v_int - decoded)))

# 통계 출력
total = lossless + lossy
print("\n✅ SPARK 인코딩 손실 통계")
print(f"총 weight 수 : {total}")
print(f"Lossless 수  : {lossless} ({100 * lossless / total:.2f}%)")
print(f"Lossy 수     : {lossy} ({100 * lossy / total:.2f}%)")

# 디버깅 출력 (일부만 보여주기)
print("\n🔍 디버깅용 손실 항목 (상위 10개만 표시):")
for i, (v, decoded, err) in enumerate(lossy_list[:10]):
    print(f"[{i+1}] v={v}, decoded={decoded}, error={err}")
