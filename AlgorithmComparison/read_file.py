from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def encode_ordinal(X):
    oe_style = OrdinalEncoder(dtype=int)
    stringColumns = X.columns[0:3]
    X_arr = oe_style.fit_transform(X)
  #  for column in X.columns:
   #     X[column] = oe_style.fit_transform([X[column]])
    #   column = oe_style.fit_transform(X[stringColumns])
    #      oe_results = oe_style.fit_transform(X[[column]])
    # column_names = [column + "_" + cat for cat in column]
    #       X = X.join(pd.DataFrame(oe_results.toarray(), columns=column_names))
    # X = X.join(pd.DataFrame(column, columns=column_names))

    #   X = oe_style.fit_transform(X)

    X = pd.DataFrame(X_arr, columns=X.columns)
  #  X = X.iloc[:, 3:15]
  #  X.rename(columns=''.join, inplace=True)
    X.to_csv('u2r_fixed-noother.csv', index=False)

    return X


def encode_onehot(X):
    oe_style = OneHotEncoder(dtype=int)
    stringColumns = X.columns[0:3]
    print(stringColumns)
    for column in stringColumns:
        oe_results = oe_style.fit_transform(X[[column]])
        column_names = [column + "_" + cat for cat in oe_style.categories_]
        X = X.join(pd.DataFrame(oe_results.toarray(), columns=column_names))

    print(X.shape)
    X = X.iloc[:, 3:23]
    X.rename(columns=''.join, inplace=True)
    X.to_csv('u2r_fixed.csv', index=False)

    return X


if __name__ == '__main__':
    data = arff.loadarff('datasets/u2rvsnormal.arff')
    df = pd.DataFrame(data[0])
    df = df.select_dtypes([object])
    df = df.stack().str.decode('utf-8').unstack()
    print(df.head())

    X = df.iloc[:, 0:6]
    print(X.head())

    y = df.iloc[:, 6]
    print(y.head())

    #   count = 0
    #   for index, row in X.iterrows():
    #       if row["service"] != "http" and row["service"] != "smtp" and row["service"] != "ftp" \
    #               and row["service"] != "ftp_data" and row["service"] != "private":
    #           row["service"] = "other"
    #           count = count + 1

    X = encode_ordinal(X)
    y = y.astype(int)
    y.to_csv('u2r_labels.csv', index=False)
