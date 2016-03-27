import pandas  as pd
import sklearn as sk
import numpy   as np
from   sklearn import cross_validation, linear_model, metrics, preprocessing
from   sklearn.learning_curve import learning_curve
from   sklearn.metrics        import roc_curve

import matplotlib.pyplot as plt

def loadFile(fileName, polyFeatures = None):
    '''
        Over here, we shall neglect the column that 
        says ROLE_TITLE. We shall keep all of the
        rest of the features in. 

        This will optionally create polynomial features
        as required ...
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

def plot_trainingCurve(model, X, y, N=5):
    '''
        This function will plot the training curve. This is 
        useful to see if we have a bias/variance problem at
        our hands ...
    '''

    plt.figure(figsize=(4, 3))
    plt.axes([0.17, 0.18, 0.94-0.17, 0.96-0.18])

    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, 
        train_sizes = np.linspace(0.1, 1.0, N), 
        cv=5)

    meanTraining = train_scores.mean(axis=1)
    stdTraining  = train_scores.std(axis=1)
    meanValid = valid_scores.mean(axis=1)
    stdValid  = valid_scores.std(axis=1)

    plt.plot(np.log10(train_sizes), 
        meanTraining, 's-', mfc='red', mec='black', 
        color='red', label='training')

    plt.fill_between( np.log10(train_sizes), 
        meanTraining+stdTraining*3, 
        meanTraining-stdTraining*3, 
        facecolor='red', alpha=0.5)

    plt.plot(np.log10(train_sizes), meanValid, 
        's-', mfc='green', mec='black', 
        color='green', label='validation')

    plt.fill_between( np.log10(train_sizes), 
        meanValid+stdValid*3, 
        meanValid-stdValid*3, 
        facecolor='green', alpha=0.5)

    plt.xlabel(r'$\log_{10}(size)$')
    plt.ylabel(r'accuracy')

    return

def plot_roc(fpr, tpr):
    plt.figure(figsize=(4, 3))
    plt.axes([0.17, 0.18, 0.94-0.17, 0.96-0.18])
    plt.plot(fpr, tpr)

    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    return

if __name__ == '__main__':

    plt.ion()
    
    y, X      = loadFile('data/train.csv')
    _, X_test = loadFile('data/test.csv')
    
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X      = encoder.transform(X) 
    X_test = encoder.transform(X_test)

    model = linear_model.LogisticRegression(C=3)

    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.20, random_state=100)
    
    model.fit(X_train, y_train) 
    preds = model.predict_proba(X_cv)[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
    roc_auc = metrics.auc(fpr, tpr)
    print roc_auc

    plot_roc(fpr, tpr)

    plt.show()

    


    print 'done'
