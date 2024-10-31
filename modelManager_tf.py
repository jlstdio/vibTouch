import os
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, losses, callbacks
import tensorflow as tf
from wandb.integration.keras import WandbCallback


class ModelManager:
    def __init__(self, model, device='GPU', lr=1e-4):
        self.model = model
        self.device = device
        self.lr = lr

        # 옵티마이저
        self.optimizer = optimizers.Adam(learning_rate=self.lr)

        # 모델 컴파일
        self.model.compile(
            optimizer=self.optimizer,
            loss={
                'main_output': losses.BinaryCrossentropy(),
                'sub_output': losses.SparseCategoricalCrossentropy()
            },
            metrics={
                'main_output': ['accuracy'],
                'sub_output': ['accuracy']
            },
            run_eagerly=False  # 디버깅 완료 후 False 유지
        )

    def train_model(self, train_dataset, val_dataset, num_epochs, checkpoint_dir='pths'):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # 체크포인트 콜백 (최고 성능 모델만 저장)
        checkpoint_cb = callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=False  # 전체 모델 저장
        )

        # 조기 종료 콜백 (선택 사항)
        early_stop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

        wandb_cb = WandbCallback(save_weights=True)

        # 학습
        history = self.model.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=val_dataset,
            callbacks=[checkpoint_cb, wandb_cb, early_stop_cb],
            verbose=1
        )

        # 훈련 완료 후 모델 저장 (SavedModel 형식)
        saved_model_dir = os.path.join(checkpoint_dir, 'saved_model')
        self.model.save(saved_model_dir, save_format='tf')
        print(f"SavedModel saved at {saved_model_dir}")

        # TensorFlow Lite 변환 및 저장
        self.convert_to_tflite(saved_model_dir, os.path.join(checkpoint_dir, 'model.tflite'))

        return history

    def convert_to_tflite(self, saved_model_dir, tflite_save_path):
        # TensorFlow Lite 변환기 초기화
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

        # 양자화 등 최적화 설정 (선택 사항)
        # 예: 정수 양자화
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8

        try:
            # 변환 실행
            tflite_model = converter.convert()

            # .tflite 파일 저장
            with open(tflite_save_path, 'wb') as f:
                f.write(tflite_model)

            print(f"TensorFlow Lite model saved at {tflite_save_path}")
        except Exception as e:
            print(f"Failed to convert to TFLite: {e}")

    def evaluate_model(self, test_dataset):
        # 모델 평가
        results = self.model.evaluate(test_dataset, verbose=1)
        print(f"Test Loss and Metrics: {results}")

        # 예측 및 F1 스코어 계산
        all_main_preds = []
        all_main_labels = []
        all_sub_preds = []
        all_sub_labels = []

        for batch in test_dataset:
            inputs, labels = batch
            preds = self.model.predict(inputs, verbose=0)
            main_pred = (preds[0] >= 0.5).astype(int).flatten()
            sub_pred = np.argmax(preds[1], axis=1)

            all_main_preds.extend(main_pred)
            all_main_labels.extend(labels['main_output'].numpy())
            all_sub_preds.extend(sub_pred)
            all_sub_labels.extend(labels['sub_output'].numpy())

        # F1 스코어 계산
        f1_main = f1_score(all_main_labels, all_main_preds, average='binary')
        f1_sub = f1_score(all_sub_labels, all_sub_preds, average='weighted')

        # 서브 레이블 분리 (Slide: <4, Tap: >=4)
        sub_labels_slide = (np.array(all_sub_labels) < 4).astype(int)
        sub_labels_tap = (np.array(all_sub_labels) >= 4).astype(int)
        sub_preds_slide = (np.array(all_sub_preds) < 4).astype(int)
        sub_preds_tap = (np.array(all_sub_preds) >= 4).astype(int)

        f1_sub_slide = f1_score(sub_labels_slide, sub_preds_slide, average='binary')
        f1_sub_tap = f1_score(sub_labels_tap, sub_preds_tap, average='binary')

        print(f"Main F1 Score: {f1_main:.4f}, Sub F1 Score: {f1_sub:.4f}")
        print(f"Sub Slide F1 Score: {f1_sub_slide:.4f}, Sub Tap F1 Score: {f1_sub_tap:.4f}")

        # 혼동 행렬 시각화 및 저장
        self.plot_confusion_matrix(all_main_labels, all_main_preds, ['Slide', 'Tap'], 'confusion_matrix_main.png')
        self.plot_confusion_matrix(all_sub_labels, all_sub_preds,
                                   ['Slide0', 'Slide1', 'Slide2', 'Slide3', 'Tap1', 'Tap2', 'Tap3', 'Tap4'],
                                   'confusion_matrix_sub.png')
        self.plot_confusion_matrix(sub_labels_slide, sub_preds_slide, ['Not Slide', 'Slide'],
                                   'confusion_matrix_sub_slide.png')
        self.plot_confusion_matrix(sub_labels_tap, sub_preds_tap, ['Not Tap', 'Tap'], 'confusion_matrix_sub_tap.png')

        return f1_main, f1_sub, f1_sub_slide, f1_sub_tap

    def plot_confusion_matrix(self, y_true, y_pred, labels, save_path):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

        fmt = '.2f'
        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                plt.text(j, i, format(cm_normalized[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm_normalized[i, j] > thresh else "black")

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()