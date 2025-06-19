# from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained(
#     "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static-inc",
#     trust_remote_code=True
# )

# # 파라미터 dtype 확인
# for name, param in model.named_parameters():
#     print(f"{name}: {param.dtype}")

#######################
from transformers import AutoModelForSequenceClassification

# HuggingFace 양자화된 모델 불러오기
model = AutoModelForSequenceClassification.from_pretrained(
    "yujiepan/bert-base-uncased-sst2-int8-unstructured80-30epoch"
)

# 가중치 dtype 확인
for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")

########################
# from transformers.utils.import_utils import is_torch_available
# import torch
# print("Torch Available:", is_torch_available())
# print("Torch Version:", torch.__version__)



