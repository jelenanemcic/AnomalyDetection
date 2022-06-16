import random
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler

import algorithms
import helper


def run_DBSCAN():
    eps = 5.3
    samples = 60
    max_j = 20

    y_trues = []
    y_predictions = []
    silhouette, davies = np.zeros(max_j), np.zeros(max_j)

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.1, frac_negative=1, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=RobustScaler)

        y_pred = algorithms.calculate_DBSCAN(X, y, eps, samples, "cancer")

        y_trues.append(y)
        y_predictions.append(y_pred)

        s, d = helper.calculate_clustering_metrics(X, y_pred)
        silhouette[j] = s
        davies[j] = d

    number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
        accuracies, rand = helper.calculate_metrics(y_trues, y_predictions, "dbscan-cancer")

    helper.save("dbscan-cancer", number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                f1_scores, accuracies, rand, silhouette.mean(), davies.mean())

    helper.print_all(number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores,
                     accuracies, rand, silhouette.mean(), davies.mean())


def run_GaussianMixture():
    n_components = 1
    percentile = 6.5
    max_j = 20

    y_trues = []
    y_predictions = []
    silhouette, davies = np.zeros(max_j), np.zeros(max_j)

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.1, frac_negative=1, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=RobustScaler)

        y_pred = algorithms.calculate_Gaussian(X, y, n_components, percentile, "cancer")

        y_trues.append(y)
        y_predictions.append(y_pred)

        s, d = helper.calculate_clustering_metrics(X, y_pred)
        silhouette[j] = s
        davies[j] = d

    number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
        accuracies, rand = helper.calculate_metrics(y_trues, y_predictions, "gaussian-cancer")

    helper.save("gaussian-cancer", number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                f1_scores, accuracies, rand, silhouette.mean(), davies.mean())

    helper.print_all(number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores,
                     accuracies, rand, silhouette.mean(), davies.mean())


def run_KMeans():
    k = 1
    percentile = 93.5
    max_j = 20

    y_trues = []
    y_predictions = []
    silhouette, davies = np.zeros(max_j), np.zeros(max_j)

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.1, frac_negative=1, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=RobustScaler)

        y_pred = algorithms.calculate_KMeans(X, y, k, percentile, "cancer")

        y_trues.append(y)
        y_predictions.append(y_pred)

        s, d = helper.calculate_clustering_metrics(X, y_pred)
        silhouette[j] = s
        davies[j] = d

    number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
        accuracies, rand = helper.calculate_metrics(y_trues, y_predictions, "kmeans-cancer")

    helper.save("kmeans-cancer", number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                f1_scores, accuracies, rand, silhouette.mean(), davies.mean())

    helper.print_all(number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores,
                     accuracies, rand, silhouette.mean(), davies.mean())


if __name__ == '__main__':
    X = pd.read_csv('datasets/cancer_X.csv')
    y = pd.read_csv('datasets/cancer_y.csv')
    y.columns = ["class"]

    num_neg = len(y[y == 0])
    num_pos = int(y[y == 1].sum())
    print('Number of positive / negative samples: {} / {}'.format(num_pos, num_neg))
    print('Fraction of positives: {:.2%}'.format(num_pos / num_neg))

    df = pd.concat([X, y], axis=1)

    run_KMeans()
    # run_DBSCAN()
    # run_GaussianMixture()
