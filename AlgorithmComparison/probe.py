import random

import pandas as pd

import algorithms
import helper


def run_DBSCAN():
    eps = 0.1  # 0.1 i 1300
    sample = 1300
    max_j = 3

    y_trues = []
    y_predictions = []

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.5, frac_negative=0.5, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=None)

        y_pred = algorithms.calculate_DBSCAN(X, y, eps, sample)

        y_trues.append(y)
        y_predictions.append(y_pred)

    number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies, rand = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("dbscan-probe", number_of_positives, total_number, auc_roc, average_precisions, precisions,
                recalls, f1_scores, accuracies, rand)

    helper.print_all(number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                     f1_scores, accuracies, rand)


def run_GaussianMixture():
    max_j = 10
    n_components = 4
    percentile = 12

    y_trues = []
    y_predictions = []

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.5, frac_negative=0.5, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=None)

        y_pred = algorithms.calculate_Gaussian(X, y, n_components, percentile)

        y_trues.append(y)
        y_predictions.append(y_pred)

    number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies, rand = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("gaussian-probe", number_of_positives, total_number, auc_roc, average_precisions, precisions,
                recalls, f1_scores, accuracies, rand)

    helper.print_all(number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                     f1_scores, accuracies, rand)


def run_KMeans():
    k = 1
    max_j = 10
    percentile = 90

    y_trues = []
    y_predictions = []

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.5, frac_negative=0.5, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=None)

        y_pred = algorithms.calculate_KMeans(X, y, k, percentile)

        y_trues.append(y)
        y_predictions.append(y_pred)

    number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies, rand = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("kmeans-probe", number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                f1_scores, accuracies, rand)

    helper.print_all(number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
                     f1_scores, accuracies, rand)


if __name__ == '__main__':
    X = pd.read_csv('datasets/probe_X.csv')
    y = pd.read_csv('datasets/probe_y.csv').squeeze()

    num_neg = len(y[y == 0])
    num_pos = int(y[y == 1].sum())
    print('Number of positive / negative samples: {} / {}'.format(num_pos, num_neg))
    print('Fraction of positives: {:.2%}'.format(num_pos / num_neg))

    df = pd.concat([X, y], axis=1)

    #  run_KMeans()
    #  run_DBSCAN()
    run_GaussianMixture()
