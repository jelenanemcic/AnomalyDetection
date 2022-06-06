import random

import pandas as pd
import numpy as np

import algorithms
import helper


def run_DBSCAN():
    eps = 0.25
    samples = 25
    max_j = 5

    y_trues = []
    y_predictions = []
    silhouette, davies = np.zeros(max_j), np.zeros(max_j)

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.2, frac_negative=0.1, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=None)

        y_pred = algorithms.calculate_DBSCAN(X, y, eps, samples)

        y_trues.append(y)
        y_predictions.append(y_pred)

        s, d = helper.calculate_clustering_metrics(X, y_pred)
        silhouette[j] = s
        davies[j] = d

    number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies, rand = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("dbscan-credit", number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                f1_scores, accuracies, rand, silhouette.mean(), davies.mean())

    helper.print_all(number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores,
                     accuracies, rand, silhouette.mean(), davies.mean())


def run_GaussianMixture():
    max_j = 5
    n_components = 10
    percentile = 1

    y_trues = []
    y_predictions = []
    silhouette, davies = np.zeros(max_j), np.zeros(max_j)

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.2, frac_negative=0.1, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=None)

        y_pred = algorithms.calculate_Gaussian(X, y, n_components, percentile)

        y_trues.append(y)
        y_predictions.append(y_pred)

        s, d = helper.calculate_clustering_metrics(X, y_pred)
        silhouette[j] = s
        davies[j] = d

    number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies, rand = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("gaussian-credit", number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                f1_scores, accuracies, rand, silhouette.mean(), davies.mean())

    helper.print_all(number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores,
                     accuracies, rand, silhouette.mean(), davies.mean())


def run_KMeans():
    k = 1
    max_j = 1
    threshold = 0.52

    y_trues = []
    y_predictions = []
    silhouette, davies = np.zeros(max_j), np.zeros(max_j)

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.2, frac_negative=0.1, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=None)

        y_pred = algorithms.calculate_KMeans(X, y, k, threshold)

        y_trues.append(y)
        y_predictions.append(y_pred)

        s, d = helper.calculate_clustering_metrics(X.to_numpy(), y_pred)
        silhouette[j] = s
        davies[j] = d

    number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies, rand = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("kmeans-credit", number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                f1_scores, accuracies, rand, silhouette.mean(), davies.mean())

    helper.print_all(number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores,
                     accuracies, rand, silhouette.mean(), davies.mean())


if __name__ == '__main__':
    df = pd.read_csv('datasets/creditcardfraud_normalised.csv')
    num_neg = (df["class"] == 0).sum()
    num_pos = df["class"].sum()
    print('Number of positive / negative samples: {} / {}'.format(num_pos, num_neg))
    print('Fraction of positives: {:.2%}'.format(num_pos / num_neg))

    #    run_KMeans()
    run_DBSCAN()
    #   run_GaussianMixture()
