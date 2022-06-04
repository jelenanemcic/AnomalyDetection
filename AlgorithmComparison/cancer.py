import random

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

import algorithms
import helper


def run_DBSCAN(df):
    eps = 5.5
    samples = 60 # najbolje eps 3.5, samples 3; eps 5 ili 5.5, s 60 najbolje
    max_j = 10

    y_trues = []
    y_predictions = []
    silhouette, davies = np.zeros(max_j), np.zeros(max_j)

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.1, frac_negative=1, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=RobustScaler)

        y_pred = algorithms.calculate_DBSCAN(X, y, eps, samples)

        y_trues.append(y)
        y_predictions.append(y_pred)

        s, d = helper.calculate_clustering_metrics(X, y_pred)
        silhouette[j] = s
        davies[j] = d

    number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies, rand = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("dbscan-cancer", number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                f1_scores, accuracies, rand, silhouette.mean(), davies.mean())

    helper.print_all(number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores,
                     accuracies, rand, silhouette.mean(), davies.mean())


def run_GaussianMixture(df):
    max_j = 10
    n_components = 2
    percentile = 15

    y_trues = []
    y_predictions = []
    silhouette, davies = np.zeros(max_j), np.zeros(max_j)

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.1, frac_negative=1, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=RobustScaler)

        y_pred = algorithms.calculate_Gaussian(X, y, n_components, percentile)

        y_trues.append(y)
        y_predictions.append(y_pred)

        s, d = helper.calculate_clustering_metrics(X, y_pred)
        silhouette[j] = s
        davies[j] = d

    number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies, rand = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("gaussian-cancer", number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                f1_scores, accuracies, rand, silhouette.mean(), davies.mean())

    helper.print_all(number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores,
                     accuracies, rand, silhouette.mean(), davies.mean())


def run_KMeans(df):
    k = 1
    max_j = 1
    percentile = 90

    y_trues = []
    y_predictions = []
    silhouette, calinski, davies = np.zeros(max_j), np.zeros(max_j), np.zeros(max_j)

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.1, frac_negative=1, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=RobustScaler)

        y_pred = algorithms.calculate_KMeans(X, y, k, percentile)

        y_trues.append(y)
        y_predictions.append(y_pred)

        s, c, d = helper.calculate_clustering_metrics(X, y_pred)
        silhouette[j] = s
        calinski[j] = c
        davies[j] = d

    number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("kmeans-cancer", number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
         f1_scores, accuracies, silhouette.mean(), calinski.mean(), davies.mean())

    helper.print_all(number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores,
                     accuracies, silhouette.mean(), calinski.mean(), davies.mean())


if __name__ == '__main__':
    X = pd.read_csv('datasets/cancer_X.csv')
    y = pd.read_csv('datasets/cancer_y.csv')
    y.columns = ["class"]

    num_neg = len(y[y == 0])
    num_pos = int(y[y == 1].sum())
    print('Number of positive / negative samples: {} / {}'.format(num_pos, num_neg))
    print('Fraction of positives: {:.2%}'.format(num_pos / num_neg))

  #  run_KMeans(pd.concat([X, y], axis=1))
    run_DBSCAN(pd.concat([X, y], axis=1))
  #  run_GaussianMixture(pd.concat([X, y], axis=1))