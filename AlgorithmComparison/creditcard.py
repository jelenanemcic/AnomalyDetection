import random

import pandas as pd

import algorithms
import helper


def run_DBSCAN():
    eps = 0.25
    samples = 25
    max_j = 5

    y_trues = []
    y_predictions = []

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.2, frac_negative=0.1, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=None)

        y_pred = algorithms.calculate_DBSCAN(X, y, eps, samples)

        y_trues.append(y)
        y_predictions.append(y_pred)

    fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("dbscan-credit", fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
         f1_scores, accuracies)

    helper.print_all(fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, accuracies)


def run_KMeans():
    k = 1
    max_j = 3
    threshold = 0.52

    y_trues = []
    y_predictions = []

    for j in range(0, max_j):
        X, y = algorithms.downsample_scale_split_df(df, frac_positive=0.2, frac_negative=0.1, verbose=1,
                                                    random_state=random.randint(0, 10000), scaler=None)

        y_pred = algorithms.calculate_KMeans(X, y, k, threshold)

        y_trues.append(y)
        y_predictions.append(y_pred)

    fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, \
    accuracies = helper.calculate_metrics(y_trues, y_predictions)

    helper.save("kmeans-credit", fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls,
         f1_scores, accuracies)

    helper.print_all(fractions, number_of_positives, total_number, auc_roc, average_precisions, precisions, recalls, f1_scores, accuracies)


if __name__ == '__main__':
    df = pd.read_csv('datasets/creditcardfraud_normalised.csv')
    num_neg = (df["class"] == 0).sum()
    num_pos = df["class"].sum()
    print('Number of positive / negative samples: {} / {}'.format(num_pos, num_neg))
    print('Fraction of positives: {:.2%}'.format(num_pos / num_neg))

    run_KMeans()
  #  run_DBSCAN()


