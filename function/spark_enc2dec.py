from function.spark_encode import spark_encode
from spark_decode import spark_decode

errors = []
for v in range(256):
    enc, _, _ = spark_encode(v)
    dec = spark_decode(enc)
    if abs(dec - v) > 1:
        errors.append((v, dec))

if errors:
    print("❌ Too much error in SPARK:")
    for v, dec in errors[:10]:
        print(f"v={v}, decoded={dec}, error={abs(v - dec)}")
else:
    print("✅ SPARK decode error within ±1 → OK")
