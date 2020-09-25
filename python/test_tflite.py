from os.path import splitext
from os import walk
import csv
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix

#import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite
# print(tf.__version__)


def evaluate_metrics(confusion_matrix, y_test, y_pred, print_result=False, f1_avg='macro'):
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    
    TP = np.diag(confusion_matrix)
    FP = confusion_matrix.sum(axis=0) - TP
    FN = confusion_matrix.sum(axis=1) - TP    
    TN = confusion_matrix.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # ACC_micro = (sum(TP) + sum(TN)) / (sum(TP) + sum(FP) + sum(FN) + sum(TN))
    ACC_macro = np.mean(
        ACC)  # to get a sense of effectiveness of our method on the small classes we computed this average (macro-average)

    f1 = f1_score(y_test, y_pred, average=f1_avg)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    if (print_result):
        print("\n")
        print("\n")
        print("============ METRICS ============")
        print(confusion_matrix)
        print("Accuracy (macro) : ", ACC_macro)        
        print("F1 score         : ", f1)
        print("Cohen Kappa score: ", kappa)
        print("======= Per class metrics =======")
        print("Accuracy         : ", ACC)
        print("Sensitivity (TPR): ", TPR)
        print("Specificity (TNR): ", TNR)
        print("Precision (+P)   : ", PPV)
    
    return ACC_macro, ACC, TPR, TNR, PPV, f1, kappa


# tflite_load = "Loaded" #@param ["Loaded", "File"]

def load_lite_model(lite_model=None, tflite_load='Loaded', filename=None):
    print(filename)
    # Load TFLite model
    if tflite_load == 'Loaded':
        interpreter = tflite.Interpreter(model_content=lite_model)
    elif tflite_load == 'File':
        interpreter = tflite.Interpreter(model_path=filename + ".tflite")

    # allocate tensors
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    return interpreter, input_index, output_index


def test_lite_model(X_test, y_test, interpreter, input_index, output_index):
    predictions = []
    test_labels = y_test
    exec_times  = []
    segments = X_test #.astype(np.float32)

    for i in range(segments.shape[0]):
        segment = segments[[i],:,:]
        start_time = time.time()
        interpreter.set_tensor(input_index, segment)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_index)
        end_time = time.time()
        exec_times.append(end_time-start_time)
        predictions.append(pred)

    # convert predictions to classes
    predictions = np.vstack(predictions)
    y_pred = np.argmax(predictions,1)

    # convert to miliseconds
    exec_times = np.array(exec_times) * 1000

    # evaluate
    cm = confusion_matrix(y_test, y_pred)
    ACC_macro, ACC, TPR, TNR, PPV, f1, kappa = evaluate_metrics(cm, y_test, y_pred, True)

    return y_pred, exec_times, (cm, ACC_macro, ACC, TPR, TNR, PPV, f1, kappa)


def save_lite_test_result(y_test, y_pred, exec_times, filename):
    df = pd.DataFrame({'label': y_test[:,0], 'pred': y_pred, 'exec_time': exec_times})
    df.to_csv(filename + '__lt_test_result.csv')
    
    return df


def load_test_data(filename, label_col='label'):
    df = pd.read_csv(filename, index_col=False)
    # df = df.drop('Unnamed: 0', axis=1)
    # Get label col only
    y_test = df[label_col].values
    # Get features without index and label col
    X_test = df.iloc[:,0:-1].values
    # Convert to float32
    X_test = np.float32(X_test)
    # Mod dimension
    X_test.shape = X_test.shape + (1,)
    y_test.shape = y_test.shape + (1,)
    
    # ~ print(df.head())
    
    return df, X_test, y_test



FILENAME = 'models/lite/acharya__default_ovr_train__16-06-2020_11-30-10'

_, X_test_1, y_test_1 = load_test_data('dataset/test__default_split__260.csv')

# Load lite model
interpreter_1, input_idx_1, output_idx_1 = load_lite_model(tflite_load='File', filename=FILENAME)
# Test
y_pred_1, exec_times_1, cm = test_lite_model(X_test_1, y_test_1, interpreter_1, input_idx_1, output_idx_1)

print("Mean of execution times: " , np.mean(exec_times_1))

plt.boxplot(exec_times_1, showfliers=False)
plt.show()
# Extract metrics
#cm_lt_1, ACC_macro_lt_1, ACC_lt_1, TPR_lt_1, TNR_lt_1, PPV_lt_1, f1_lt_1, kappa_lt_1 = metrics_lt_1

# Save exec_time & prediction results
# save_lite_test_result(y_test_1, y_pred_1, exec_times_1, FILENAME)
