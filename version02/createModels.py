import pandas  as pd
import sklearn as sk
import numpy   as np
import seaborn as sns
import matplotlib.pyplot as plt
import plots

from   itertools import combinations

from   sklearn                import cross_validation, linear_model, metrics, preprocessing
from   sklearn.ensemble       import RandomForestClassifier, AdaBoostClassifier
from   sklearn.learning_curve import learning_curve
from   sklearn.externals      import joblib


'''
http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
'''


models = [
    linear_model.LogisticRegression(C=0.6),
    linear_model.LogisticRegressionCV(Cs=np.linspace(0.1, 2, 10), tol=1e-6),
    AdaBoostClassifier()
    ]

N       = 100
models += [ RandomForestClassifier(max_depth=1000, 
    n_estimators=e, 
    max_features=f) for (e, f) in zip(
        map(int, np.random.normal(20, 2, N)),
        map(int, np.random.normal(15, 2, N))
    ) ]

modelNames = [
    'Logistic_Reg',
    'Logistic_RegCV',
    # 'AdaBoost'
    ]

modelNames += [ 'RF%03d'%i for i in range(N) ]

if __name__ == '__main__':

    plt.ion()
    
    y      = np.load('data/y.npy')
    X      = np.load('data/XtrainExt.npy')
    X_test = np.load('data/XtestExt.npy')

    print 'Generalting the testing and training sets ...'
    random_state=147    
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.20, random_state=int(random_state))

    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.01, random_state=int(random_state))

    # Lets save the data first ...
    # np.save('data/y_ps.npy', y_ps)
    # np.save('data/X_ps.npy', X_ps)
    np.save('data/y_cv.npy', y_cv)
    np.save('data/X_cv.npy', X_cv)
    np.save('data/y.npy', y)
    np.save('data/X.npy', X)
    np.save('data/X_test.npy', X_test)


    print 'Encoding the data ...'
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X       = encoder.transform(X) 
    X_test  = encoder.transform(X_test)
    X_train = encoder.transform(X_train)
    X_cv    = encoder.transform(X_cv)

    rocs = []
    fprs, tprs, thresholdss = [], [], []
    fitModels = []
    for model, name in zip(models, modelNames):
        
        print 'Simple Estimate[%s]: '%name ,

        model.fit(X_train, y_train) 
        preds                = model.predict_proba(X_cv)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
        roc_auc              = metrics.auc(fpr, tpr)
        print roc_auc, 

        print 'Fitting Model',
        model.fit(X, y) 

        print 'Saving the model file ...'
        joblib.dump(model, '../../models1/'+name+'_model.pkl') 

    plt.show()

    


    print 'done'
