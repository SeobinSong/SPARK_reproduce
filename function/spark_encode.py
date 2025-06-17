def spark_encode(value: int) -> tuple[str, int, bool]:
    """
    SPARK 논문 기반 정확한 인코딩 구현
    입력: 0~255 정수
    출력: (인코딩된 비트 문자열, 총 비트 수)
    """
    assert 0 <= value <= 255, "Value must be in [0, 255]"

    # 0~7: Case 1 - 4bit, identifier 0 + 3bit
    if value <= 7:
        bin_code = "0" + format(value, "03b") # valid 3-bit
        return bin_code, 4, False

    # 8~127: Case 2 - 8bit, identifier 1 + 7bit
    elif value <= 127:
        bin_8 = format(value, "08b") # valid 7-bit
        b0 = int(bin_8[0])  # 첫 번째 비트
        b3 = int(bin_8[3])  # 네 번째 비트
        if b0 ^ b3 == 0:
            # b0=0, b3=0 : "1" + 하위 7비트 그대로 사용
            bin_code = "1" + bin_8[1:]
            return bin_code, 8, False
        else:
            # b0=0, b3=1 : 1xx0 1111 (내림)
            bin_code = "1" + bin_8[1:3] + "01111"
            return bin_code, 8, True 

    # 128~255: Case 3 - 8bit, identifier 1 + 7bit
    else:
        bin_8 = format(value, "08b") # valid 7-bit
        b0 = int(bin_8[0])  # 첫 번째 비트
        b3 = int(bin_8[3])  # 네 번째 비트
        if b0 ^ b3 == 0:
            # b0=1, b3=1 : "1" + 하위 7비트 그대로 사용
            bin_code = "1" + bin_8[1:]
            return bin_code, 8, False
        else:
            # b0=1, b3=0 : 1xx1 0000 (올림)
            bin_code = "1" + bin_8[1:3] + "10000"
            return bin_code, 8, True

# Example usage of spark_encode function
if __name__ == "__main__":
    # Test cases
    print(spark_encode(5))    # Case 1 : 0101
    print(spark_encode(72))   # Case 2 : 0100 1000 >> 1100 1000
    print(spark_encode(88))   # Case 2 : 0101 1000 >> 1100 1111 spark error (9)
    print(spark_encode(154))  # Case 3 : 1001 1010 >> 1001 1010
    print(spark_encode(138))  # Case 3 : 1000 1010 >> 1001 0000 spark error (6)

    case2_rounded = []
    case3_rounded = []

    for value in range(256):
        _, _, rounded = spark_encode(value)
        if 8 <= value <= 127 and rounded:
            case2_rounded.append(value)
        elif 128 <= value <= 255 and rounded:
            case3_rounded.append(value)
    print(
        f"Case 2에서 rounding된 값: {len(case2_rounded)}개"
        f" ({round(len(case2_rounded)/120*100, 2)}%) → {case2_rounded}"
    )
    print(
        f"Case 3에서 rounding된 값: {len(case3_rounded)}개"
        f" ({round(len(case3_rounded)/128*100, 2)}%) → {case3_rounded}"
    )



