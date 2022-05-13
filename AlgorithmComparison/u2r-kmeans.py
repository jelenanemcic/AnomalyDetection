import sklearn
from kmodes import kmodes
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OrdinalEncoder

if __name__ == '__main__':
    data = arff.loadarff('u2rvsnormal.arff')
    df = pd.DataFrame(data[0])
    df = df.select_dtypes([object])
    df = df.stack().str.decode('utf-8').unstack()
    print(df.head())

    X = df.iloc[:, 0:6]
    print(X.head())

    y = df.iloc[:, 6]
    print(y.head())
    #
    from sklearn.preprocessing import OneHotEncoder

    # oe_style = OneHotEncoder(dtype=int)
    # oe_results = oe_style.fit_transform(X[["protocol_type"]])
    # column_names = ["protocol_type_" + cat for cat in oe_style.categories_]
    # X = X.join(pd.DataFrame(oe_results.toarray(), columns=column_names))
    #
    # oe_results = oe_style.fit_transform(X[["service"]])
    # column_names = ["service_" + cat for cat in oe_style.categories_]
    # X = X.join(pd.DataFrame(oe_results.toarray(), columns=oe_style.categories_))
    #
    # oe_results = oe_style.fit_transform(X[["flag"]])
    # column_names = ["flag_" + cat for cat in oe_style.categories_]
    # X = X.join(pd.DataFrame(oe_results.toarray(), columns=oe_style.categories_))

  #  X = X.iloc[0:20000, 3:-1]
  #  X.rename(columns=''.join, inplace=True)

    X = X.iloc[0:30000]
    print(X.shape)

    print(X.head(100))
    y = y.astype(int)

    ysmall = y.iloc[0:30000]
    print(y.dtype)
    ytrue = ysmall.loc[ysmall == 1].index.values.astype(int)
    print(ytrue[0:60])

  #  kmode = kmodes.KModes(n_clusters=2, init = "random", n_init = 50, verbose=1)
#    kmode.fit_predict(X)
 #   print(kmode.cost_)
 #   ypred = np.zeros(kmeans.labels_.shape)
  #  ypred[clustering.labels_ == -1] = 1
   # ypred = ypred.astype(int)
  #  print(sklearn.metrics.accuracy_score(ysmall, kmode.labels_))
