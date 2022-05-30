import random

import pandas as pd

import algorithms
import helper


def run_DBSCAN():
    eps = 0.001     # 0.001 i 1000
    sample = 1000
    max_j = 1

    y_trues = []
    y_predictions = []

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.5, frac_negative=0.5, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=None)

        y_pred = algorithms.calculate_DBSCAN(X, y, eps, sample)

        y_trues.append(y)
        y_predictions.append(y_pred)

    fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("dbscan-probe", fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions,
                recalls,
                f1_scores, accuracies)

    helper.print_all(fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                     f1_scores, accuracies)


def run_GaussianMixture():
    max_j = 5
    n_components = 8
    percentile = 10

    y_trues = []
    y_predictions = []

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.5, frac_negative=0.5, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=None)

        y_pred = algorithms.calculate_Gaussian(X, y, n_components, percentile)

        y_trues.append(y)
        y_predictions.append(y_pred)

    fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("gaussian-probe", fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions,
                recalls,
                f1_scores, accuracies)

    helper.print_all(fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                     f1_scores, accuracies)


def run_KMeans():
    k = 1
    max_j = 3
    threshold = 2.5

    y_trues = []
    y_predictions = []

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.5, frac_negative=0.5, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=None)

        y_pred = algorithms.calculate_KMeans(X, y, k, threshold)

        y_trues.append(y)
        y_predictions.append(y_pred)

    fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("kmeans-probe", fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
         f1_scores, accuracies)

    helper.print_all(fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, accuracies)


if __name__ == '__main__':
    X = pd.read_csv('datasets/probe_fixed2.csv')
    y = pd.read_csv('datasets/probe_labels.csv').squeeze()

    num_neg = len(y[y == 0])
    num_pos = int(y[y == 1].sum())
    print('Number of positive / negative samples: {} / {}'.format(num_pos, num_neg))
    print('Fraction of positives: {:.2%}'.format(num_pos / num_neg))

    df = pd.concat([X, y], axis=1)

  #  run_KMeans()
 #   run_DBSCAN()
    run_GaussianMixture()

