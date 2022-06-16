import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, \
    f1_score, precision_score, recall_score, silhouette_score, davies_bouldin_score, rand_score


def calculate_metrics(y_true, y_pred, name):
    number_of_positives = np.zeros(len(y_pred))
    total_number = np.zeros(len(y_pred))
    auc_roc = np.zeros(len(y_pred))
    average_precisions = np.zeros(len(y_pred))
    precisions = np.zeros(len(y_pred))
    recalls = np.zeros(len(y_pred))
    f1_scores = np.zeros(len(y_pred))
    accuracies = np.zeros(len(y_pred))
    rand = np.zeros(len(y_pred))

    for i in range(0, len(y_pred)):
        number_of_positives[i] = y_true[i][y_pred[i] == 1].sum()
        total_number[i] = np.sum(y_pred[i] == 1)
        auc_roc[i] = roc_auc_score(y_true[i], y_pred[i])
        average_precisions[i] = average_precision_score(y_true[i], y_pred[i])
        precisions[i] = precision_score(y_true[i], y_pred[i])
        recalls[i] = recall_score(y_true[i], y_pred[i])
        f1_scores[i] = f1_score(y_true[i], y_pred[i])
        accuracies[i] = accuracy_score(y_true[i], y_pred[i])
        rand[i] = rand_score(y_true[i], y_pred[i])

    np.savetxt('stats/f1_' + name + '.txt', f1_scores)
    return number_of_positives.mean(), total_number.mean(), auc_roc.mean(), average_precisions.mean(), \
        precisions.mean(), recalls.mean(), f1_scores.mean(), accuracies.mean(), rand.mean()


def calculate_clustering_metrics(X, y_pred):
    silhouette = silhouette_score(X, y_pred)
    davies = davies_bouldin_score(X, y_pred)
    return silhouette, davies


def save(name, *to_save):
    means = np.zeros(len(to_save))
    for i in range(0, len(to_save)):
        means[i] = to_save[i].mean()
    np.savetxt('stats/stats_' + name + '.txt', means)


def print_all(*to_print):
    for i in range(0, len(to_print)):
        print(to_print[i])


def plot(parameter, score, name):
    plt.figure(figsize=(16, 8))
    plt.plot(parameter, score, 'bx-')
    plt.xlabel(name)
    plt.ylabel('F1-score')
    plt.show()
