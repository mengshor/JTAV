from __future__ import print_function
from sklearn.metrics import roc_auc_score, precision_score, f1_score
import numpy as np
import keras.backend as K


def mean_pred(y_pred):
    return K.mean(y_pred)


def evaluation(results, labels_test, name):
    threshold = 0.3
    predicted_labels = np.zeros(results.shape)
    predicted_labels[results >= threshold] = 1
    print('###############################################')
    print('threshold: ', threshold)
    weighted_f1 = f1_score(labels_test, predicted_labels, average='weighted')
    print('weighted-f1:{:.3f}'.format(weighted_f1))
    roc_auc = roc_auc_score(labels_test, predicted_labels, average='weighted')
    print('ROC-AUC: %.3f' % roc_auc)
    precision = precision_score(labels_test, predicted_labels, average='weighted')
    print('precision:{:.3f}'.format(precision))

    if roc_auc > 0.515 and weighted_f1 > 0.55 and precision > 0.5:
        with open('log', 'a') as log_out:
            import time

            log_out.write('\n############### {} #############'.format(name))
            log_out.write("\n{}\n".format(time.asctime(time.localtime(time.time()))))
            log_out.write('=====================================\n')
            log_out.write('ROC-AUC: {roc:.3f}\n'.format(roc=roc_auc))
            log_out.write('weighted-f1:{f1:.3f}\n'
                          'precision: {precision:.3f}\n'.format(f1=weighted_f1, precision=precision))
    return roc_auc, weighted_f1, precision
