def spark_encode_with_correction_bit(value: int) -> tuple[str, int, bool]:
    """
    SPARK 인코딩 + 보정 비트 추가 (1비트)
    입력: 0~255 정수
    출력: (인코딩된 비트 문자열, 총 비트 수, lossy 여부)
    """
    assert 0 <= value <= 255, "Value must be in [0, 255]"

    # Case 1: 0~7 (4bit, lossless)
    if value <= 7:
        bin_code = "0" + format(value, "03b")  # identifier 0 + 3-bit
        return bin_code, 4, False

    # Case 2: 8~127
    elif value <= 127:
        bin_8 = format(value, "08b")
        b0 = int(bin_8[0])
        b3 = int(bin_8[3])
        if b0 ^ b3 == 0:
            # 그대로 사용 (lossless)
            bin_code = "1" + bin_8[1:] + "0"  # 마지막 1비트: 보정 없음
            return bin_code, 9, False
        else:
            # lossy encoding: 내림 (원래는 1xx01111), 보정 방향: +8
            bin_code = "1" + bin_8[1:3] + "01111" + "1"  # correction bit = 1
            return bin_code, 9, True

    # Case 3: 128~255
    else:
        bin_8 = format(value, "08b")
        b0 = int(bin_8[0])
        b3 = int(bin_8[3])
        if b0 ^ b3 == 0:
            bin_code = "1" + bin_8[1:] + "0"  # lossless + correction bit 0
            return bin_code, 9, False
        else:
            # lossy encoding: 올림 (원래는 1xx10000), 보정 방향: -8
            bin_code = "1" + bin_8[1:3] + "10000" + "1"  # correction bit = 1
            return bin_code, 9, True

# 테스트
# test_outputs = []
# for val in [5, 72, 88, 154, 138]:
#     test_outputs.append((val, spark_encode_with_correction_bit(val)))

# test_outputs
