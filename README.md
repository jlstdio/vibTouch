# vibTouch


# Prompted Structure
- chatGPT used

너의 역할은 인공지능 모델링이다.

사용할 도구는 Pytorch이며, Transformer 기반의 인공지능 모델을 만들어라.

다음은 네가 사용하게 될 데이터에 대한 설명이다.
1. 모델 학습에 사용될 input은 4 채널이다.
2. 각각 채널0,1,2에는 가속도계 센서 데이터(100hz)가 그리고 채널 3에는 마이크 소리 데이터 (16000hz)가 들어간다.
3. 데이터의 경우 다음과 같다
   1. slide 제스쳐의 경우. 클래스는 3_0, 3_1, 3_2, 3_3이 있다. data/sliced/slide/3_0/acc, data/sliced/slide/3_0/audio, data/sliced/slide/3_1/acc, data/sliced/slide/3_1/audio, data/sliced/slide/3_2/acc, data/sliced/slide/3_2/audio, data/sliced/slide/3_3/acc, data/sliced/slide/3_3/audio 파일 경로가 있다. acc의 경우, acc_0.csv, acc_1.csv...과 같은 형태로 0부터 1씩 커지는 숫자로 넘버링되도록 파일이 있고. audio의 경우에도 audio_0.wav, audio_1.wav...과 같은 형태로 0부터 1씩 커지는 숫자로 넘버링되도록 파일이 있다. 
   2. Tap 제스쳐의 경우. 클래스는 0,1,2,3이 있다. data/sliced/slide/0/acc, data/sliced/slide/0/audio, data/sliced/slide/1/acc, data/sliced/slide/1/audio, data/sliced/slide/2/acc, data/sliced/slide/2/audio, data/sliced/slide/3/acc, data/sliced/slide/3/audio 파일 경로가 있다. acc의 경우, acc_0.csv, acc_1.csv...과 같은 형태로 0부터 1씩 커지는 숫자로 넘버링되도록 파일이 있고. audio의 경우에도 audio_0.wav, audio_1.wav...과 같은 형태로 0부터 1씩 커지는 숫자로 넘버링되도록 파일이 있다.
   3. audio 데이터의 경우 sample rate는 16khz에 bits per sample은 24bit이다. 데이터 길이는 12800 이다.
   4. acc 데이터의 경우 길이가 80이다. csv파일 형태이며 column은 time,x,y,z이다. x,y,z데이터를 각각 추출하여 채널 0,1,2로 삼는다.
다음은 모델의 목적 및 기타 사항에 대해서 설명하겠다.
1. 모델은 시계열 데이터에 대한 Classification 모델이다.
2. 스마트폰의 마이크와 가속도계를 활용하여 스마트폰을 책상에 둔 채로 오른쪽, 왼쪽, 위, 아래를 탭(두드리는)하는 동작과 스마트폰의 오른쪽에서 손가락으로 오른쪽, 왼쪽, 위, 아래 방향으로 쓸어서 나는 소리와 진동 데이터를 실시간으로 구분하는 인공지능 모델을 만들려고 한다.
3. 각 데이터를 받아 실시간으로 제스쳐를 구분하는 인공지능 모델을 만들면 된다.
4. 각 데이터는 1초 단위로 달려서 학습 데이터가 만들어져있다.
6. cuda를 쓸 수 있다면 써라.

네가 해야할 일은 다음과 같다.
1. 특정 위치에 있는 소리 (wav)및 가속도계(csv) 데이터를 읽어들여 dataloader를 만든다.
2. 데이터 라벨은 0,1,2,3,4,5와 같이 숫자로 라벨링한다
3. 준비된 데이터는 8:2의 비율로 각각 train와 validation dataset을 만든다.
4. Transformer 기반의 인공지능 학습 모델을 만든다.
5. Batch size 16으로 학습 데이터를 준비하고, 100epoch로 학습하는 학습 코드를 만든다.

너는 이 과제를 달성하기 위해 최선을 다 해야한다.

