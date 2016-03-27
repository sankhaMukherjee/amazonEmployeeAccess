'''
    saving models once done ...
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
from sklearn.ensemble              import ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.externals import joblib


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

models = [
    # linear_model.LogisticRegression(C=3),
    # KNeighborsClassifier(13),
    # SVC(kernel="linear", C=1, probability=True),
    # SVC(gamma=2, C=1, probability=True),
    # DecisionTreeClassifier(max_depth=1000),
    # RandomForestClassifier(n_estimators=1999, max_features='sqrt', max_depth=None, min_samples_split=9, random_state=8803),#8803
    RandomForestClassifier(max_depth=1000, n_estimators=20, max_features=15),
    # AdaBoostClassifier()
    # ExtraTreesClassifier(n_estimators=1999, max_features='sqrt', max_depth=None, min_samples_split=8, random_state=8903), #8903, 
    ExtraTreesClassifier(max_depth=2000, n_estimators=20, max_features=15),
    # GradientBoostingClassifier(n_estimators=50, learning_rate=0.20, max_depth=20, min_samples_split=9, random_state=8749)  #8749
    ]

modelNames = [
    # 'Logistic_Reg',
    # 'KNN_13',
    # 'SVC_Linear',
    # 'SVC',
    # 'DecisionTree',
    # 'RandomForest',
    'RandomForest1',
    # 'AdaBoost',
    # 'ExtraTrees',
    'ExtraTrees1',
    # 'GradBoost'

    ]

def ROC():

    return

if __name__ == '__main__':

    plt.ion()
    
    y, X      = loadFile('data/train.csv')
    _, X_test = loadFile('data/test.csv')

    # Add polynomial features if necessary
    # X      = np.hstack((X, np.load('data/trainF_02.npy')))
    # X_test = np.hstack((X_test, np.load('data/testF_02.npy')))

    # X      = np.hstack((X, np.load('data/trainF_03.npy')))
    # X_test = np.hstack((X_test, np.load('data/testF_03.npy')))

    print 'Generalting the testing and training sets ...'
    random_state=147    
    X_train, X_ps, y_train, y_ps = cross_validation.train_test_split(
            X, y, test_size=.20, random_state=int(random_state))

    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X_train, y_train, test_size=.20, random_state=int(random_state))

    # Lets save the data first ...
    np.save('y_ps.npy', y_ps)
    np.save('X_ps.npy', X_ps)
    np.save('y_cv.npy', y_cv)
    np.save('X_cv.npy', X_cv)
    np.save('y.npy', y)
    np.save('X.npy', X)
    np.save('X_test.npy', X_test)


    print 'Encoding the data ...'
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X       = encoder.transform(X) 
    X_test  = encoder.transform(X_test)
    X_train = encoder.transform(X_train)
    X_ps    = encoder.transform(X_ps)
    X_cv    = encoder.transform(X_cv)

    rocs = []
    fprs, tprs, thresholdss = [], [], []
    fitModels = []
    for model, name in zip(models, modelNames):
        
        print 'Now doing:', name,

        model.fit(X_train, y_train) 
        preds = model.predict_proba(X_cv)[:, 1]
        
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
        roc_auc = metrics.auc(fpr, tpr)
        print roc_auc

        # rocs.append( roc_auc )
        # fprs.append( fpr )
        # tprs.append( tpr )
        # thresholdss.append( thresholds )
        # fitModels.append( model )

        joblib.dump(model, 'models/'+name+'_model.pkl') 


        # plots.plot_trainingCurve(model, X, y, 3, title=name)
        # plt.savefig(name+'_TrainingCurve.png')

        


    # for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
    #     plots.plot_roc(fpr, tpr, roc_auc, newPlot=(i==0), 
    #         color=(float(i)/(len(modelNames)-1),0.5,1-float(i)/(len(modelNames)-1)),
    #         label=modelNames[i])
    # plt.legend(loc = 'lower right')
    # plt.savefig('allRocs.png')

    plt.show()

    


    print 'done'
