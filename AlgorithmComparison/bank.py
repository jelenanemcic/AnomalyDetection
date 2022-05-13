import sklearn
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.decomposition import PCA

if __name__ == '__main__':
    data = arff.loadarff('bank-additional-ful-nominal.arff')
    df = pd.DataFrame(data[0])
    df = df.select_dtypes([object])
    df = df.stack().str.decode('utf-8').unstack()
    print(df.head())

    X = df.iloc[:, 0:-1]
    print(X.head())

    y = df.iloc[:, -1]
    print(y.head())

    oe_style = OneHotEncoder(dtype=int)
  #  oe_style = OrdinalEncoder(dtype=int)

    for column in X.columns:
        #X[column] = oe_style.fit_transform(X[[column]])
        oe_results = oe_style.fit_transform(X[[column]])
        column_names = [column + "_" + cat for cat in oe_style.categories_]
        X = X.join(pd.DataFrame(oe_results.toarray(), columns=column_names))

    print(X.shape)

    X = X.iloc[:, 10:63]
    X.rename(columns=''.join, inplace=True)
    print(X.shape)
    print(X.head(10))

    y_copy = y
    y_copy = y_copy.where(y == "no", 1)
    y = y_copy.where(y == "yes", 0)

    y = y.astype(int)
    print(y)
    #
    ysmall = y.iloc[:]
    # print(y.dtype)
    ytrue = ysmall.loc[ysmall == 1].index.values.astype(int)
    print(ytrue[0:60])
    print(len(ytrue))

    pca = PCA(n_components=20, svd_solver='full')
    principalComponents = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))

    principalX = pd.DataFrame(data=principalComponents)

    epss = [0.1]
    samples = [3]
    for sample in samples:
        for i in epss:
            clustering = DBSCAN(eps=i, min_samples=sample).fit(principalX)
            # print(clustering.labels_)
            # print(np.where(clustering.labels_ == -1))

            n_noise_ = list(clustering.labels_).count(-1)
            print(n_noise_)

            ypred = np.zeros(clustering.labels_.shape)
            ypred[clustering.labels_ == -1] = 1
            ypred = ypred.astype(int)
            #    print(ypred[ytrue])
            #    print(ysmall[ytrue])
            print(sklearn.metrics.accuracy_score(ysmall[ytrue], ypred[ytrue]))
            print(sklearn.metrics.accuracy_score(y, clustering.labels_))
    #         break
