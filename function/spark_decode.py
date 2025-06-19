from pprint import pprint

def spark_decode(bin_code: str) -> int:
    """
    SPARK 디코딩 함수
    입력: SPARK 인코딩된 4bit 또는 8bit 비트 문자열
    출력: 디코딩된 정수 (0~255 내)
    """
    if len(bin_code) == 4:
        # Case 1: identifier = 0, 3bit 값
        assert bin_code[0] == "0"
        value = int(bin_code[1:], 2) # 7-bit 2진수를 10진수 정수로 변환
        return value

    elif len(bin_code) == 8:
        assert bin_code[0] == "1"
        payload = bin_code[1:]  # 7비트 부분
        c0 = int(bin_code[0])
        c3 = int(bin_code[3])
        # c0 ^ c3 == 1 → [7, 127]
        if c0 ^ c3 == 1:
            value = int(payload, 2)
            return value
        # c0 ^ c3 == 0 → [128, 255]
        else :
            value = int(bin_code, 2)  # 7-bit 2진수를 10진수 정수로 변환
            return value

    else:
        raise ValueError(f"Invalid SPARK bit code: {bin_code}")


# Example usage of spark_decode function
# test_vals = [5, 88, 154, 138]
# results = []

# for v in test_vals:
#     encoded, _, _ = spark_encode(v)
#     decoded = spark_decode(encoded)
#     results.append((v, encoded, decoded))

# pprint(results) # list안의 tuple을 예쁘게 보여줌
