'''
    regularization ...
'''

import pandas  as pd
import sklearn as sk
import numpy   as np
import seaborn as sns
from   sklearn import cross_validation, linear_model, metrics, preprocessing
from   sklearn.learning_curve import learning_curve
from itertools import combinations

import matplotlib.pyplot as plt
import plots

'''
http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
'''
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.svm                   import SVC
from sklearn.tree                  import DecisionTreeClassifier
from sklearn.ensemble              import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes           import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


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

    X      = np.hstack((X, np.load('data/trainF_03.npy')))
    X_test = np.hstack((X_test, np.load('data/testF_03.npy')))

    print 'Encoding the data ...'
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X      = encoder.transform(X) 
    X_test = encoder.transform(X_test)


    reg = np.linspace(0.1, 1, 20)
    rocs = []
    for r in reg:
        model = linear_model.LogisticRegression(C=r)

        random_state=147
        # random_state=100
        
            
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                X, y, test_size=.20, random_state=int(random_state))
        
        model.fit(X_train, y_train) 
        preds = model.predict_proba(X_cv)[:, 1]

        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
        roc_auc = metrics.auc(fpr, tpr)
        print r, roc_auc
        rocs.append( roc_auc )

    # plots.plot_trainingCurve(model, X, y, 10)

    plt.figure(figsize=(4, 3))
    plt.plot(reg, rocs, 's')
    plt.xlabel(r'$1/\lambda$')
    plt.ylabel(r'$AUC$')
        

    plt.show()

    


    print 'done'
