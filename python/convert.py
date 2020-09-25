import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import load_model


DATASET_DIR = "dataset"
RESULT_DIR = "results"
MODEL_DIR = "models"

def load_dataset(filename, label_col='label'):
    df = pd.read_csv(DATASET_DIR + '/' + filename + '.csv', index_col=None)
    # df = df.drop('Unnamed: 0', axis=1)
    # Get label col only
    y = df[label_col].values
    # Get features without index and label col
    X = df.iloc[:,0:-1].values
    # Convert to float32
    X = np.float32(X)
    # Mod dimension
    X.shape = X.shape + (1,)
    y.shape = y.shape + (1,)
    return df, X, y

def save_lite_model(model, filename, model_dir=MODEL_DIR+'/lite/'):
    full_name = model_dir + filename + '.tflite'

    with open(full_name, "wb") as f:
        f.write(model)

def representative_data_gen():
    # data = X_test[:100,:,:].astype(np.float32)
    for input_value in X_repr.astype(np.float32):
        input_value = np.expand_dims(input_value, axis=0)
        yield [input_value]

def lite_convert(model, opt_mode, int_mode=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Do something
    # quantization
    # etc

    # optimization
    if opt_mode.lower() == 'storage':
        optimization = tf.lite.Optimize.OPTIMIZE_FOR_SIZE
    elif opt_mode.lower() == 'speed':
        optimization = tf.lite.Optimize.OPTIMIZE_FOR_LATENCY
    else:
        optimization = tf.lite.Optimize.DEFAULT
        
    converter.optimizations = [optimization]
    
    if int_mode:
        # Representative data
        converter.representative_dataset = representative_data_gen
        # Limit to Int8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8

    tflite_model = converter.convert()

    return tflite_model

def convert_saved_model(filename, int_mode=False, model_dir=MODEL_DIR+'/lite/quant'):
    model = load_model(MODEL_DIR + '/' + filename + '.h5')
    lite_model = lite_convert(model, 'speed', int_mode)
    print("Model dir: " +model_dir)
    save_lite_model(lite_model, filename + '_int', model_dir)
    return lite_model


# Set the representative data first
_, X_repr, _ = load_dataset('test__default_split__260')

# Convert the model
lite_model = convert_saved_model('acharya__default_ovr_train__16-06-2020_11-30-10', True, '')