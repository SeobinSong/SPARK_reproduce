def spark_decode_with_correction_bit(bin_code: str) -> int:
    """
    SPARK 디코딩 함수 (보정 비트 반영)
    입력: SPARK 인코딩된 4bit (lossless) 또는 9bit (보정 비트 포함)
    출력: 디코딩된 정수 (0~255 내)
    """
    if len(bin_code) == 4:
        # Case 1: 4bit, identifier = 0
        assert bin_code[0] == "0"
        return int(bin_code[1:], 2)

    elif len(bin_code) == 9:
        assert bin_code[0] == "1"
        core = bin_code[:-1]  # 앞 8비트
        corr_bit = int(bin_code[-1])  # 마지막 1비트

        c0 = int(bin_code[0])
        c3 = int(bin_code[3])

        if c0 ^ c3 == 1:
            # lossy zone → 보정 비트 사용
            value = int(core, 2)
            if c3 == 1:
                # Case 2: 보정 방향 +8
                return value + 8 if corr_bit == 1 else value
            else:
                # Case 3: 보정 방향 -8
                return value - 8 if corr_bit == 1 else value
        else:
            # lossless
            return int(core, 2)

    else:
        raise ValueError(f"Invalid SPARK bit code: {bin_code}")

# 테스트
from pprint import pprint

def spark_encode_with_correction_bit(value: int) -> tuple[str, int, bool]:
    assert 0 <= value <= 255
    if value <= 7:
        bin_code = "0" + format(value, "03b")
        return bin_code, 4, False
    elif value <= 127:
        bin_8 = format(value, "08b")
        b0, b3 = int(bin_8[0]), int(bin_8[3])
        if b0 ^ b3 == 0:
            return "1" + bin_8[1:] + "0", 9, False
        else:
            return "1" + bin_8[1:3] + "01111" + "1", 9, True
    else:
        bin_8 = format(value, "08b")
        b0, b3 = int(bin_8[0]), int(bin_8[3])
        if b0 ^ b3 == 0:
            return "1" + bin_8[1:] + "0", 9, False
        else:
            return "1" + bin_8[1:3] + "10000" + "1", 9, True

# # 확인
# test_vals = [5, 88, 154, 138]
# results = []
# for v in test_vals:
#     encoded, _, _ = spark_encode_with_correction_bit(v)
#     decoded = spark_decode_with_correction_bit(encoded)
#     results.append((v, encoded, decoded))

# results
