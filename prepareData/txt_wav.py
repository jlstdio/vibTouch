import numpy as np
from scipy.io.wavfile import write

# 데이터 불러오기
name = 'audio_1730377940974' #audio_1730377940974.txt
with open(f"{name}.txt", "r") as f:
    data = f.read()

# 텍스트를 리스트로 변환하고 부동소수점 값으로 변환
data = np.array([float(x) for x in data.split(",")])

# 정규화
normalized_data = data / np.max(np.abs(data))

# 샘플링 주파수 설정
sampling_rate = 1000  # Hz

# WAV 파일로 저장
write(f"{name}.wav", sampling_rate, normalized_data.astype(np.float32))

print("WAV 파일이 'output.wav'에 저장되었습니다.")
