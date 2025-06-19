##########################################################################
# SPARK BERT 파라미터 분포 분석 및 시각화
# 분석 대상1 : BERT 모델의 파라미터 분포
# 분석 대상2 : SPARK Encoding 후 손실(Lossy)/무손실(Lossless) 구간 통계
##########################################################################
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from transformers import BertForSequenceClassification, BertTokenizer

# # 모델 불러오기
# model = BertForSequenceClassification.from_pretrained(
#     "textattack/bert-base-uncased-SST-2",
#     trust_remote_code=True,
#     use_safetensors=True
# )

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # SPARK 손실 가능 영역 정의 (논문 Table II 기준)
# spark_lossy_range = set()
# for start in [16, 48, 80, 112, 128, 160, 192, 224]:
#     spark_lossy_range.update(range(start, start + 16))

# # 레이어별 처리
# for name, param in model.named_parameters():
#     if "weight" not in name or not param.requires_grad:
#         continue

#     print(f"\n📌 [Layer] {name}")

#     # (1) FP32 분포 시각화
#     data_fp32 = param.data.cpu().view(-1).numpy()
#     plt.hist(data_fp32, bins=100, color='skyblue')
#     plt.title(f"{name} - FP32 Parameter Distribution")
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # (2) INT8 양자화
#     max_val = param.data.abs().max()
#     scale = max_val / 127
#     int8_tensor = torch.round(param.data / scale).clamp(-128, 127).to(torch.int)
#     data_int8 = int8_tensor.cpu().view(-1).numpy()

#     plt.hist(data_int8, bins=256, range=(-128, 127), color='orange')
#     plt.title(f"{name} - INT8 Quantized Distribution")
#     plt.xlabel("Quantized Value")
#     plt.ylabel("Frequency")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # (3) SPARK 손실 통계 계산
#     quantized_tensor = int8_tensor + 128  # unsigned로 변환
#     count_lossy = 0
#     count_safe = 0

#     for v in quantized_tensor.view(-1):
#         if int(v) in spark_lossy_range:
#             count_lossy += 1
#         else:
#             count_safe += 1

#     total = count_lossy + count_safe
#     print(f"   ▶ Total Params     : {total}")
#     print(f"   ▶ Lossy Range      : {count_lossy} ({100 * count_lossy / total:.2f}%)")
#     print(f"   ▶ Lossless Range   : {count_safe} ({100 * count_safe / total:.2f}%)")

#     # (4) SPARK 손실 히스토그램
#     plt.figure(figsize=(5, 4))
#     plt.bar(['Lossy', 'Lossless'], [count_lossy, count_safe], color=['red', 'green'])
#     plt.yscale('log')
#     plt.title(f"{name} - SPARK Encoding Zones (Log Scale)")
#     plt.ylabel("Log-scaled Number of Params")
#     plt.tight_layout()
#     plt.grid(axis='y', which='both')
#     plt.savefig(f"{name.replace('.', '_')}_spark_encoding_log_clean.png")
#     plt.close()



#     # ✅ 한 개 레이어만 보고 싶을 경우 break
#     break

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import BertForSequenceClassification

# 모델 로딩
model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-SST-2",
    trust_remote_code=True,
    use_safetensors=True
)

# 특정 레이어만 선택
param = dict(model.named_parameters())["bert.embeddings.word_embeddings.weight"]
data_fp32 = param.data.cpu().view(-1).numpy()

# 히스토그램 + KDE (커널 밀도 추정) 같이 시각화
plt.figure(figsize=(8, 4))
sns.histplot(data_fp32, bins=100, kde=True, color='blue')
plt.title("FP32 Weight Distribution - bert.embeddings.word_embeddings.weight")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("fp32_distribution_word_embedding.png")
plt.show()
