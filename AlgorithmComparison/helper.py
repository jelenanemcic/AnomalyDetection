from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, \
    f1_score, precision_score, recall_score
import numpy as np


def calculate_metrics(y_true, y_pred):
    fractions = np.zeros(len(y_pred))
    number_of_positives = np.zeros(len(y_pred))
    total_number = np.zeros(len(y_pred))
    auc_roc = np.zeros(len(y_pred))
    average_precisions = np.zeros(len(y_pred))
    precisions = np.zeros(len(y_pred))
    recalls = np.zeros(len(y_pred))
    f1_scores = np.zeros(len(y_pred))
    accuracies = np.zeros(len(y_pred))

    for i in range(0, len(y_pred)):
        fractions[i] = y_true[i][y_pred[i] == 1].mean()
        number_of_positives[i] = y_true[i][y_pred[i] == 1].sum()
        total_number[i] = np.sum(y_pred[i] == 1)
        auc_roc[i] = roc_auc_score(y_true[i], y_pred[i])
        average_precisions[i] = average_precision_score(y_true[i], y_pred[i])
        precisions[i] = precision_score(y_true[i], y_pred[i])
        recalls[i] = recall_score(y_true[i], y_pred[i])
        f1_scores[i] = f1_score(y_true[i], y_pred[i])
        accuracies[i] = accuracy_score(y_true[i], y_pred[i])

    return fractions.mean(), number_of_positives.mean(), total_number.mean(), auc_roc.mean(), average_precisions.mean(),\
           precisions.mean(), recalls.mean(), f1_scores.mean(), accuracies.mean()


def save(name, *to_save):
    means = np.zeros(len(to_save))
    for i in range(0, len(to_save)):
        means[i] = to_save[i].mean()
    np.savetxt('stats/stats_' + name + '.txt', means)


def print_all(*to_print):
    for i in range(0, len(to_print)):
        print(to_print[i])
