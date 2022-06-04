import random

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

import algorithms
import helper


def run_DBSCAN(df):
    eps = [2, 5]
    samples = [25, 50]
    max_j = 1

    y_trues = []
    y_predictions = []

    silhouette, calinski, davies = np.zeros(max_j), np.zeros(max_j), np.zeros(max_j)

    for e in eps:
        for sample in samples:
            for j in range(0, max_j):
                X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.1, frac_negative=0.1, verbose=1,
                                                            random_state=random.randint(0, 10000), scaler=RobustScaler)

                #   algorithms.calculate_Nearest_Neighbors(X,25)
                y_pred = algorithms.calculate_DBSCAN(X, y, e, sample)

                y_trues.append(y)
                y_predictions.append(y_pred)

                s, c, d = helper.calculate_clustering_metrics(X, y_pred)
                silhouette[j] = s
                calinski[j] = c
                davies[j] = d

            number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
            accuracies = helper.calculate_metrics(y_trues, y_predictions)

            helper.save("dbscan-cover", number_of_positives, total_number, auc_roc, average_precisions, precisions,
                        recalls, f1_scores, accuracies, silhouette.mean(), calinski.mean(), davies.mean())

            helper.print_all(number_of_positives, total_number, auc_roc, average_precisions, precisions,
                             recalls, f1_scores, accuracies, silhouette.mean(), calinski.mean(), davies.mean())


def run_KMeans(df):
    k = 1
    max_j = 3

    y_trues = []
    y_predictions = []

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.5, frac_negative=0.1, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=RobustScaler)

        y_pred = algorithms.calculate_KMeans(X, y, k, 5)

        y_trues.append(y)
        y_predictions.append(y_pred)

    fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("kmeans-cover", fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions,
                recalls, f1_scores, accuracies)

    helper.print_all(fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                     f1_scores, accuracies)


if __name__ == '__main__':
    X = pd.read_csv('datasets/cover_X.csv', header=None)
    y = pd.read_csv('datasets/cover_y.csv', header=None)
    y.columns = ["class"]

    num_neg = len(y[y == 0])
    num_pos = int(y[y == 1].sum())
    print('Number of positive / negative samples: {} / {}'.format(num_pos, num_neg))
    print('Fraction of positives: {:.2%}'.format(num_pos / num_neg))

    # run_KMeans(pd.concat([X, y], axis=1))
    run_DBSCAN(pd.concat([X, y], axis=1))
