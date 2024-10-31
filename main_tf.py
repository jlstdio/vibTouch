import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import wandb
import random
from modelArch.tensorFlow.MultimodalClassifier1D_tf import MultimodalClassifier1D
from modelManager_tf import ModelManager
from prepareData.dataloader_tf import GestureDataset


def main():
    # 랜덤 시드 설정
    seed = 1234
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 웨이트 앤 바이어스 초기화
    config = {
        "learning_rate": 0.001,
        "architecture": "MultimodalClassifier1D",
        "dataset": ['tap', 'slide'],  # ['tap'] || ['slide']
        "modelType": "typeClassifier",  # "tapClassifier" || "slideClassifier"
        "epochs": 300,
        "batchSize": 64,
    }

    wandb.init(
        project="vibTouch",
        config=config
    )

    # 데이터셋 로드
    gesture_types = config['dataset']
    print(f'초기 제스처 유형: {gesture_types}')
    # full_dataset = GestureDataset(root_dir='data/train', gesture_types=gesture_types, batch_size=config['batchSize'])
    train_dataset = GestureDataset(root_dir='data/train', gesture_types=gesture_types, batch_size=config['batchSize'])
    train_dataset.display_class_counts()

    # 데이터셋 분할 (90% 훈련, 10% 검증)
    # total_samples = len(full_dataset)
    # print(f'samples : {total_samples}')
    # all_indices = list(range(total_samples))
    # train_indices, val_indices = train_test_split(all_indices, test_size=0.1, random_state=seed)

    # 훈련 및 검증 데이터셋 생성
    # train_dataset = GestureDataset(root_dir='data/train', gesture_types=gesture_types, batch_size=config['batchSize'], indices=train_indices)
    # val_dataset = GestureDataset(root_dir='data/train', gesture_types=gesture_types, batch_size=config['batchSize'], indices=val_indices)
    val_dataset = GestureDataset(root_dir='data/test', gesture_types=gesture_types, batch_size=config['batchSize'], shuffle=False)

    val_dataset.display_class_counts()

    print('데이터 로드 완료')

    num_samples = len(train_dataset.data)
    print(f"Total number of train_dataset samples: {num_samples}\n")

    num_samples = len(val_dataset.data)
    print(f"Total number of val_dataset samples: {num_samples}\n")

    accel_data, audio_data = train_dataset.data[0]
    print(f"Accelerometer data shape: {accel_data.shape}")  # 예: (12800, 3)
    print(f"Audio data shape: {audio_data.shape}\n")  # 예: (12800,)

    # 모델 생성
    model = MultimodalClassifier1D(num_subclasses=4)
    # model.summary()  # 모델 요약 출력

    # 모델 매니저 초기화
    manager = ModelManager(model=model, device='GPU', lr=config['learning_rate'])

    # 모델 학습 및 저장
    history = manager.train_model(train_dataset, val_dataset, num_epochs=config['epochs'], checkpoint_dir='pths')

    # 테스트 데이터셋 로드 (예: data/test 디렉토리 사용)
    test_gesture_types = config['dataset']
    test_dataset = GestureDataset(root_dir='data/test', gesture_types=test_gesture_types,
                                  batch_size=config['batchSize'], shuffle=False)

    # 모델 평가
    f1_main, f1_sub, f1_sub_slide, f1_sub_tap = manager.evaluate_model(test_dataset)

    # 결과 로그
    wandb.log({
        "Test F1 Main": f1_main,
        "Test F1 Sub": f1_sub,
        "Test F1 Sub Slide": f1_sub_slide,
        "Test F1 Sub Tap": f1_sub_tap
    })

    wandb.finish()


if __name__ == "__main__":
    main()