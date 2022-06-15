import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import helper

from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score

times = []


def calculate_Gaussian(X, y, n_components, percentile, name):
    start = time.process_time()

    gaussianMixture = GaussianMixture(n_components=n_components, covariance_type='spherical', init_params='kmeans').fit(X)

    total_time = time.process_time() - start
    if len(times) == 0 and os.path.exists('stats/' + name + '-gauss-time.txt'):
        os.remove('stats/' + name + '-gauss-time.txt')
    file_object = open('stats/' + name + '-gauss-time.txt', 'a')
    file_object.write(str(total_time) + "\n")
    file_object.close()
    times.append(total_time)
    if len(times) == 20:
        print(np.mean(times))

    log_prob = gaussianMixture.score_samples(X)
    density_threshold = np.percentile(log_prob, percentile)
    print(density_threshold)
    indices = log_prob < density_threshold

    # draw_graph(0, percentile, 0.1, log_prob, y, "percentile")

    print('Number of anomalies {:d}, number of true positives {} (fraction: {:.3%})'.format(
        indices[indices == True].sum(), y[indices == 1].sum(), y[indices == 1].mean()))

    return indices


def calculate_DBSCAN(X, y, eps, min_pts, name):
    if type(min_pts) == list:
        f1_scores = []
        for p in min_pts:
            dbscan = DBSCAN(eps=eps, min_samples=p).fit(X)
            y_pred = set_outlier_labels_dbscan(dbscan.labels_)
            f1_scores.append(f1_score(y, y_pred))
        helper.plot(min_pts, f1_scores, "minPts")

    elif type(eps) == list:
        f1_scores = []
        for e in eps:
            dbscan = DBSCAN(eps=e, min_samples=min_pts).fit(X)
            y_pred = set_outlier_labels_dbscan(dbscan.labels_)
            f1_scores.append(f1_score(y, y_pred))
        helper.plot(eps, f1_scores, "eps")
    else:
        # calculate_Nearest_Neighbors(X, min_samples)
        start = time.process_time()

        dbscan = DBSCAN(eps=eps, min_samples=min_pts).fit(X)

        total_time = time.process_time() - start
        if len(times) == 0 and os.path.exists('stats/' + name + '-dbscan-time.txt'):
            os.remove('stats/' + name + '-dbscan-time.txt')
        file_object = open('stats/' + name + '-dbscan-time.txt', 'a')
        file_object.write(str(total_time) + "\n")
        file_object.close()
        times.append(total_time)
        if len(times) == 20:
            print(np.mean(times))

        for i in set(dbscan.labels_):
            print('class {}: number of points {:d}, number of positives {} (fraction: {:.3%})'.format(
                i, np.sum(dbscan.labels_ == i), y[dbscan.labels_ == i].sum(), y[dbscan.labels_ == i].mean()))

        n_noise = np.where(dbscan.labels_ == -1)
        print('Found {:d} outliers'.format(len(n_noise[0])))
        y_pred = set_outlier_labels_dbscan(dbscan.labels_)

    return y_pred


def calculate_KMeans(X, y, k, percentile, name):
    start = time.process_time()

    kmeans = KMeans(n_clusters=k).fit(X)

    total_time = time.process_time() - start
    if len(times) == 0 and os.path.exists('stats/' + name + '-kmeans-time.txt'):
        os.remove('stats/' + name + '-kmeans-time.txt')
    file_object = open('stats/' + name + '-kmeans-time.txt', 'a')
    file_object.write(str(total_time) + "\n")
    file_object.close()
    times.append(total_time)
    if len(times) == 20:
        print(np.mean(times))

    distances = kmeans.transform(X)
    min_distances = np.min(distances, axis=1)
    threshold = np.percentile(min_distances, percentile)
    print(threshold)
    indices = np.argwhere(min_distances > threshold).flatten()

    # draw_graph(percentile, 100, 0.1, distances, y, "percentile")

    print('Found {:d} outliers'.format(len(indices)))
    y_pred = np.zeros(len(y))
    y_pred[indices] = 1

    return y_pred


def set_outlier_labels_dbscan(labels):
    # y_pred = labels
    # if len(np.unique(labels)) > 2:
    #     y_pred[labels == 1] = len(np.unique(labels)) - 1
    y_pred = np.zeros(labels.shape)
    y_pred[labels == -1] = 1
    y_pred = y_pred.astype(int)

    return y_pred


def draw_graph(start, end, step, distances, y, name):
    f1_scores = []
    for i in np.arange(start, end, step):
        threshold = np.percentile(distances, i)
        if end == 100:
            indices = np.argwhere(distances > threshold).flatten()
        else:
            indices = np.argwhere(distances < threshold).flatten()
        y_pred = np.zeros(len(y))
        y_pred[indices] = 1
        f1_scores.append(f1_score(y, y_pred))

    helper.plot(np.arange(start, end, step), f1_scores, name)


def calculate_Nearest_Neighbors(X, k):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.show()


def find_best_k(X):
    distortions = []
    K = range(1, 100, 5)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(X)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def find_best_n_components(X):
    bic = []
    K = range(1, 15, 1)
    for k in K:
        gaussian_mixture = GaussianMixture(n_components=k, covariance_type='full', init_params='random')
        gaussian_mixture.fit(X)
        bic.append(gaussian_mixture.bic(X))

    plt.figure(figsize=(16, 8))
    plt.plot(K, bic, 'bx-')
    plt.xlabel('k')
    plt.ylabel('BIC')
    # plt.title('The elbow method showing the optimal number of components')
    plt.show()


# from: https://donernesto.github.io/blog/outlier-detection-data-preparation/
def downsample_scale_split_df(df_full, y_column='class', frac_negative=1, frac_positive=1, scaler=RobustScaler,
                              random_state=1, verbose=False):
    """ Returns downsampled X, y DataFrames, with prescribed downsampling of positives and negatives
    The labels (y's) should have values 0, 1 and be located in y_column
    X will additionally be scaled using the passed scaler

    Arguments
    =========
    df_full (pd.DataFrame) : data to be processed
    y_column (str) : name of the column containing the Class
    frac_negative (int): fraction of negatives in returned data
    frac_positive (int): fraction of negatives in returned data
    scaler (sci-kit learn scaler object)

    Returns
    ========
    downsampled and scaled X (DataFrame) and downsampled y (Series)
    """
    df_full = df_full.sample(frac=1)  # Shuffle the data set
    df_downsampled = (pd.concat([df_full.loc[df_full[y_column] == 0].sample(frac=frac_negative,
                                                                            random_state=random_state),
                                 df_full.loc[df_full[y_column] == 1].sample(frac=frac_positive,
                                                                            random_state=random_state)])
                      .sample(frac=1, random_state=random_state))  # a random shuffle to mix both classes
    X_downsampled = df_downsampled.loc[:, df_full.columns != y_column]
    y_downsampled = df_downsampled.loc[:, y_column]
    if scaler is not None:
        X_downsampled = scaler().fit_transform(X_downsampled)  # Scale the data
    if verbose:
        print('Number of points: {}, number of positives: {} ({:.2%})'.format(
            len(y_downsampled), y_downsampled.sum(), y_downsampled.mean()))
    return X_downsampled, y_downsampled
