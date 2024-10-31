import tensorflow as tf

# Load the Keras model from the H5 file
root_h5 = './h5'
h5_name = 'multimodal_classifier1D_epoch_271.h5'

root_tflite = './tflite'
tflite_name = 'vibTouch.tflite'
keras_model = tf.keras.models.load_model(f'{root_h5}/{h5_name}')

# Convert the Keras model to a TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open(f'{root_tflite}/{tflite_name}', 'wb') as f:
    f.write(tflite_model)