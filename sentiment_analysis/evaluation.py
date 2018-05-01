from sklearn.metrics import roc_auc_score, f1_score, precision_score


def evaluation(results, test_label, thres):
    prediction = []
    for r in results:
        if r < thres:
            prediction.append(0)
        else:
            prediction.append(1)

    auc = roc_auc_score(test_label, prediction, average='weighted')
    print('auc: ' + str(auc))

    f1 = f1_score(test_label, prediction, average='weighted')
    print('f1: ' + str(f1))

    precision = precision_score(test_label, prediction, average='weighted')
    print('precision: ' + str(precision))

    return auc, f1, precision
