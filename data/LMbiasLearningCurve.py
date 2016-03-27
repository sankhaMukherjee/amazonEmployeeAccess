import pandas  as pd
import sklearn as sk
import numpy   as np
import seaborn as sns
from   sklearn import cross_validation, linear_model, metrics, preprocessing
from   sklearn.learning_curve import learning_curve
from itertools import combinations

import matplotlib.pyplot as plt
import plots



def loadFile(fileName):
    '''
        Over here, we shall neglect the column that 
        says ROLE_TITLE. We shall keep all of the
        rest of the features in. 
    '''
    df      = pd.read_csv(fileName)
    columns = list(df.columns)
    columns = [ c for c in columns if c not in ['ROLE_CODE', 'ACTION', 'id']]

    X = np.array(df[columns])
    if 'ACTION' in df.columns:
        y = np.array(df['ACTION'])
    else:
        y = np.zeros(X.shape[0])


    return y, X


if __name__ == '__main__':

    plt.ion()
    
    y, X      = loadFile('data/train.csv')
    _, X_test = loadFile('data/test.csv')

    X      = np.hstack((X, np.load('data/trainF_02.npy')))
    X_test = np.hstack((X_test, np.load('data/testF_02.npy')))

    # X      = np.hstack((X, np.load('data/trainF_03.npy')))
    # X_test = np.hstack((X_test, np.load('data/testF_03.npy')))

    print 'Encoding the data ...'
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X      = encoder.transform(X) 
    X_test = encoder.transform(X_test)

    # Split in training and testing sets ...
    print 'Generalting the testing and training sets ...'
    random_state=147    
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.20, random_state=int(random_state))

    model = linear_model.LogisticRegression(C=0.51)
    plots.plot_trainingCurve(model, X, y, 10)
    