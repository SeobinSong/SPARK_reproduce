from pprint import pprint
from spark_encode import spark_encode


def spark_decode(bin_code: str) -> int:
    """
    SPARK 디코딩 함수 (특정 예외 변형 포함)
    입력: SPARK 인코딩된 4bit 또는 8bit 비트 문자열
    출력: 디코딩된 정수 (0~255 내)
    """
    if len(bin_code) == 4:
        # Case 1: identifier = 0, 3bit 값
        assert bin_code[0] == "0"
        value = int(bin_code[1:], 2)
        return value

    elif len(bin_code) == 8:
        assert bin_code[0] == "1"
        c0 = int(bin_code[0])
        c3 = int(bin_code[3])
        payload = bin_code[1:]

        # ✅ 예외 case: 1xx0 1111 → 0xx1 1000
        if c0 == 1 and c3 == 0 and payload[-4:] == "1111":
            bin_code = "0" + bin(int("1000", 2))[2:].zfill(3)
            return int(bin_code[1:], 2)

        # ✅ 예외 case: 1xx1 0000 → 1xx0 1000
        if c0 == 1 and c3 == 1 and payload[-4:] == "0000":
            # 수정된 비트 문자열로 대체
            bin_code = bin(int("10001000", 2))[2:].zfill(8)

        # 일반 case
        if c0 ^ c3 == 1:
            value = int(payload, 2)
            return value
        else:
            value = int(bin_code, 2)
            return value

    else:
        raise ValueError(f"Invalid SPARK bit code: {bin_code}")


# pprint(spark_decode("10001111"))  # 1xx0 1111 → 0xx1 1000 → 8
# pprint(spark_decode("10010000"))  # 1xx1 0000 → 1xx0 1000 → 136


# Example usage of spark_decode function
# test_vals = [5, 88, 154, 138]
# results = []

# for v in test_vals:
#     encoded, _, _ = spark_encode(v)
#     decoded = spark_decode(encoded)
#     results.append((v, encoded, decoded))

# pprint(results) # list안의 tuple을 예쁘게 보여줌
