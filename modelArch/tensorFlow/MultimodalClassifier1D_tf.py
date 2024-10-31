import tensorflow as tf
from tensorflow.keras import layers, models


def MultimodalClassifier1D(num_subclasses=4):
    # 가속도계 입력
    accel_input = layers.Input(shape=(None, 3), name='accel_input')  # [batch, time, channels]
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(accel_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)  # [batch, 128]
    accel_feat = layers.Flatten()(x)

    # 오디오 입력
    audio_input = layers.Input(shape=(None, 1), name='audio_input')  # [batch, time, channels]
    y = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(audio_input)
    y = layers.BatchNormalization()(y)
    y = layers.MaxPooling1D(pool_size=2)(y)

    y = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.MaxPooling1D(pool_size=2)(y)

    y = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.GlobalAveragePooling1D()(y)  # [batch, 128]
    audio_feat = layers.Flatten()(y)

    # 피처 결합
    combined = layers.concatenate([accel_feat, audio_feat], axis=1)  # [batch, 256]

    # 메인 분류기
    main = layers.Dense(128, activation='relu')(combined)
    main = layers.Dropout(0.5)(main)
    main_output = layers.Dense(1, activation='sigmoid', name='main_output')(main)  # Tap vs Slide

    # 서브 분류기
    sub = layers.Dense(128, activation='relu')(combined)
    sub = layers.Dropout(0.5)(sub)
    sub_output = layers.Dense(num_subclasses * 2, activation='softmax', name='sub_output')(sub)  # 8 클래스

    model = models.Model(inputs=[accel_input, audio_input], outputs=[main_output, sub_output])

    return model