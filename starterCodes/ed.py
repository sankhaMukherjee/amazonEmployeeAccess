""" Amazon Access Challenge Starter Code

These files provide some starter code using
the scikit-learn library. It provides some examples on how
to design a simple algorithm, including pre-processing,
training a logistic regression classifier on the data,
assess its performance through cross-validation and some
pointers on where to go next.

Paul Duan <email@paulduan.com>
"""

from __future__ import division

import numpy as np
import pandas as pd
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
from itertools import combinations, imap, count
from collections import defaultdict

SEED = 19  # always use a seed for randomized procedures

def load_data(filename, use_labels=True):
    """
    Load data from CSV files and return them as numpy arrays
    The use_labels parameter indicates whether one should
    read the first column (containing class labels). If false,
    return all 0s.
    """

    # load column 1 to 8 (ignore last one)
    data = np.loadtxt(open("data/" + filename), delimiter=',',
                      usecols=range(1, 9), skiprows=1)
    if use_labels:
        labels = np.loadtxt(open("data/" + filename), delimiter=',',
                            usecols=[0], skiprows=1)
    else:
        labels = np.zeros(data.shape[0])
    return labels, data


def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))

def smallify(X, X_test):
    """ Reduce the numeric value of the labels """
    labeller = preprocessing.LabelEncoder()
    labeller.fit(np.vstack((X, X_test)))
    return((labeller.transform(X), labeller.transform(X_test)))

def id_map_cols(values):
    """values: some iterable thing that has a length
    (Series, DataFrame, list, etc.).

    Returns: A numpy array where each distinct row in input will map
    to an integer. Integers used are contiguous and start at zero.
    """
    to_return = np.zeros(len(values), dtype=np.int32)

    # To make values stuff hashable might
    # need to make rows into tuples:
    if type(values) == pd.DataFrame:
        values = imap(tuple, values.values)
    elif type(values) == np.ndarray:
        values = imap(tuple, values)

    counter = count()
    counting_map = defaultdict(counter.next)
    for (i, v) in enumerate(values):
        to_return[i] = counting_map[v]
    return to_return

def add_tuples(X):
    """Add in the 2- and 3-tuples combinations """

    _, p = X.shape

    # Doubles
    for i in combinations(range(p), 2):
        X = np.column_stack((X, id_map_cols(X[:,i])))

    # Uncomment for extra fun
    #Triples
    #for i in combinations(range(p), 3):
    #    X = np.column_stack((X, id_map_cols(X[:,i])))

    return(X)

def main():
    """
    Fit models and make predictions.
    We'll use one-hot encoding to transform our categorical features
    into binary features.
    y and X will be numpy array objects.
    """
    model = linear_model.LogisticRegression(C=3)  # the classifier we'll use

    # === load data in memory === #
    print "loading data"
    y, X = load_data('train.csv')
    y_test, X_test = load_data('test.csv', use_labels=False)

    # === Reduce the numeric vals of the labels ===
    X, X_test = smallify(X, X_test)

    # === Add the tuples ===
    X      = add_tuples(X)
    X_test = add_tuples(X_test)

    # === TODO ADD FEATURE SELECTION HERE IF USE TRIPLES ===

    # === one-hot encoding === #
    # we want to encode the category IDs encountered both in
    # the training and the test set, so we fit the encoder on both
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_test = encoder.transform(X_test)

    # if you want to create new features, you'll need to compute them
    # before the encoding, and append them to your dataset after

    # === training & metrics === #
    mean_auc = 0.0
    n = 10  # repeat the CV procedure 10 times to get more precise results
    for i in range(n):
        # for each iteration, randomly hold out 20% of the data as CV set
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.20, random_state=i*SEED)

        # if you want to perform feature selection / hyperparameter
        # optimization, this is where you want to do it

        # train model and make predictions
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:, 1]

        # compute AUC metric for this CV fold
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
        roc_auc = metrics.auc(fpr, tpr)
        print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        mean_auc += roc_auc

    print "Mean AUC: %f" % (mean_auc/n)

    # === Predictions === #
    # When making predictions, retrain the model on the whole training set
    model.fit(X, y)
    preds = model.predict_proba(X_test)[:, 1]
    filename = raw_input("Enter name for submission file: ")
    save_results(preds, filename + ".csv")

if __name__ == '__main__':
    main()
