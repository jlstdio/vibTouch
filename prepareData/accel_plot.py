import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
name = 'accel_1730377940974' # accel_1730377940974
data = pd.read_csv(f'{name}.csv')

# 시간 생성 (100 Hz, 즉, 0.01초마다 데이터가 하나씩 있음)
time = [i * 0.01 for i in range(len(data))]  # 시간 벡터 생성

# 플롯 생성
plt.figure(figsize=(12, 6))
plt.plot(time, data['x'], label='X-axis')
plt.plot(time, data['y'], label='Y-axis')
plt.plot(time, data['z'], label='Z-axis')

# 그래프 설정
plt.title('Accelerometer Data at 100 Hz')
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration')
plt.legend()
plt.grid(True)

# 그래프 보여주기
plt.show()
