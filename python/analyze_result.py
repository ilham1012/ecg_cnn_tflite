from os.path import splitext
import os
import csv
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix

CLASS_NAME = ['N', 'S', 'V', 'F', 'Q']


def load_all_pd(dir_name, index_col=False):
    dfs = {}
    for entry in os.scandir(dir_name):
        if (entry.path.endswith(".csv") and entry.is_file()):
            df = pd.read_csv(entry.path, index_col=index_col)
            key = os.path.splitext(entry.name)[0]
            dfs.update({key : df})

    return dfs

def load_results(dir_name, index_col=False, print_metrics=False):
    result_dfs = load_all_pd(dir_name, index_col)

    metrics = {}

    for conf in result_dfs:
        if print_metrics:
    	    print("==== " + dir_name + " ====")
    	    print("---- " + conf + " ----")
    
        metrics.update({conf:
            evaluate_metrics(result_dfs[conf]['label'], result_dfs[conf]['pred'], print_metrics)})

    return result_dfs, metrics


def plot_accuracy_hist(history, title='Model', acc_ymin=None, loss_ymax=None, save=False, filename='loss'):
    if acc_ymin is None:
        acc_ymin = .8

    axes = plt.gca()
    axes.set_ylim([acc_ymin, 1.005])

    # Plot Accuracy
    plt.plot(history['accuracy'], linewidth=2, label='Train')
    plt.plot(history['val_accuracy'], linewidth=2, label='Valid')
    plt.legend(loc='upper right')
    plt.title(title + ' Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    
    if save:
        plt.savefig('results/train_hist/'+filename+'.svg', bbox_inches='tight')
    else:
        plt.show()

    plt.clf()


def plot_loss_hist(history, title='Model', acc_ymin=None, loss_ymax=None, save=False, filename='loss'):
    if loss_ymax is None:
        loss_ymax = .2

    axes = plt.gca()
    axes.set_ylim([-0.01, loss_ymax])

    # Plot Loss
    plt.plot(history['loss'], linewidth=2, label='Train')
    plt.plot(history['val_loss'], linewidth=2, label='Valid')
    plt.legend(loc='upper right')
    plt.title(title + ' Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    if save:
        plt.savefig('results/train_hist/'+filename+'.svg', bbox_inches='tight')
    else:
        plt.show()

    plt.clf()
    

def plot_training_hist(history, title='Model', acc_ymin=None, loss_ymax=None, save=False, filename='model'):
    plot_accuracy_hist(history, title, acc_ymin, loss_ymax, save, filename+'_acc')
    plot_loss_hist(history, title, acc_ymin, loss_ymax, save, filename+'_loss')


def plot_all_hist(hist_list, save=False):
    acc_ymin = []
    loss_ymax = []

    # get the min and max to standarize y-axis
    for _, hist in hist_list.items():
        trn_acc = hist['accuracy'].min()
        val_acc = hist['val_accuracy'].min()

        acc_ymin.append(trn_acc)
        acc_ymin.append(val_acc)

        trn_loss = hist['loss'].max()
        val_loss = hist['val_loss'].max()

        loss_ymax.append(trn_loss)
        loss_ymax.append(val_loss)

    acc_ymin = np.array(acc_ymin)
    loss_ymax = np.array(loss_ymax)

    print("acc min : ", acc_ymin.min(), "loss max : ", loss_ymax.max())

    for key, hist in hist_list.items():
        plot_training_hist(hist, key, acc_ymin.min(), loss_ymax.max(), save, key)


def plot_all_time(dfs, col='exec_time_ms', title='Device', showfliers=False, save=False, filename='device'):
    plt.boxplot([df[col] for _, df in dfs.items()], labels=[key for key in dfs], showfliers=showfliers)
    plt.legend(loc='upper right')
    plt.title(title + ' Execution Time')
    plt.ylabel('Time (ms)')
    plt.xlabel('Configuration')

    if save:
        plt.savefig('results/'+filename+'__exec_time.svg', bbox_inches='tight')
    else:
        plt.show()

    plt.clf()


def evaluate_metrics(y_test, y_pred, print_result=False, f1_avg='macro'):
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    cm = confusion_matrix(y_test, y_pred)

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP    
    TN = cm.sum() - (FP + FN + TP)
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

    DOR = (TP/FN) / (FP/TN)

    f1 = f1_score(y_test, y_pred, average=f1_avg)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    if (print_result):
        print("\n")
        print("\n")
        print("============ METRICS ============")
        print(cm)
        print("Accuracy (macro) : ", ACC_macro)        
        print("F1 score         : ", f1)
        print("Cohen Kappa score: ", kappa)
        print("======= Per class metrics =======")
        print("Accuracy         : ", ACC)
        print("Sensitivity (TPR): ", TPR)
        print("Specificity (TNR): ", TNR)
        print("Precision (+P)   : ", PPV)
        print("Diagnostics Odds : ", DOR)
    
    metrics = {
                'cm': cm,
                'single': {
                    'cm': cm,
                    'acc_macro': ACC_macro,
                    'f1': f1,
                    'kappa': kappa,
                },

                'multi': {
                    'acc': ACC,
                    'sensitivity': TPR,
                    'specitivity': TNR,
                    'precision': PPV,
                    'dor': DOR
                }

            }

    return metrics


# Train Histories
hist_dfs = load_all_pd('results/train_hist')

# Test Results
jetson_dfs, jetson_metrics = load_results('results/jetson_nano', 0, print_metrics=True)
raspi_dfs, raspi_metrics = load_results('results/rasp_pi', print_metrics=True)
mi5_dfs, mi5_metrics = load_results('results/mi5', print_metrics=True)

# savefig('foo.png', bbox_inches='tight')
# plot_all_hist([])
plot_all_hist(hist_dfs, save=False)

plot_all_time(jetson_dfs, title='Jetson Nano', save=False, filename='jetson')
plot_all_time(raspi_dfs, title='Raspberry Pi 3+ B', save=False, filename='raspi')
plot_all_time(mi5_dfs, title='Android - Xiaomi Mi 5', save=False, filename='mi5')
