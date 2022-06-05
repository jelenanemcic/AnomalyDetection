from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def encode_ordinal(X):
    oe_style = OrdinalEncoder(dtype=int)
    X_arr = oe_style.fit_transform(X)
    X = pd.DataFrame(X_arr, columns=X.columns)
    return X


def encode_one_hot(X):
    oe_style = OneHotEncoder(dtype=int)
    stringColumns = X.columns
    print(stringColumns)
    for column in stringColumns:
        oe_results = oe_style.fit_transform(X[[column]])
        column_names = [column + "_" + cat for cat in oe_style.categories_]
        X = X.join(pd.DataFrame(oe_results.toarray(), columns=column_names))

    X.rename(columns=''.join, inplace=True)
    return X.iloc[:, 3:]


def combine_services(X):
    m = dict()
    n = dict()
    print(X)
    for index, row in X.iterrows():
        if row["service"] != "http" and row["service"] != "smtp" and row["service"] != "domain_u" \
                and row["service"] != "ftp_data" and row["service"] != "private":
            row["service"] = "other"

        if row["flag"] != "SF":
            row["flag"] = "other"

        if row["service"] in m:
            m[row["service"]] = m[row["service"]]+1
        else:
            m[row["service"]] = + 1
        if row["flag"] in n:
            n[row["flag"]] = n[row["flag"]] + 1
        else:
            n[row["flag"]] = + 1
    print(sorted(m.items(), key=lambda item: item[1]))
    print(sorted(n.items(), key=lambda item: item[1]))
    return X


def read_breast_dataset(df):
    X = df.iloc[:, 2:32]
    y = df.iloc[:, 1]
    y = pd.Series(np.where(y.values == 'M', 1, 0), y.index)
    return X, y


def save(X, y, name):
    X.to_csv('datasets/' + name + '_X.csv', index=False)
    y.to_csv('datasets/' + name + '_y.csv', index=False)


def read_dataset(df, y_column):
    print(df.head)
    X = df.iloc[:, 0:y_column]
    print(X.head())

    y = df.iloc[:, y_column]
    y = y.astype(int)
    print(y.head())

    return X, y


if __name__ == '__main__':
    #df = pd.read_csv('datasets/thyroid.csv')
    data = arff.loadarff('datasets/u2rvsnormal.arff')
    df = pd.DataFrame(data[0])
    df = df.select_dtypes([object])
    df = df.stack().str.decode('utf-8').unstack()

    name = 'u2r'
    X, y = read_dataset(df, y_column=6)

    X2 = combine_services(X.iloc[:, 0:3])
    X2 = encode_one_hot(X2)
    print(X2)
    X = pd.concat([X.iloc[:, 3:], X2], axis=1)
    save(X, y, name)
