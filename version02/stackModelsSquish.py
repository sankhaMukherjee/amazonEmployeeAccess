'''
    This program is going to be used for stacking models that 
    have been previously saved as pickle object. 
'''
import numpy as np
import pandas as pd
import matplotlib.pyploy as plt

from sklearn.externals import joblib
from sklearn           import preprocessing, metrics

from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model

from datetime import datetime as dt


modelNames = [
    'Logistic_Reg',
    'Logistic_RegCV',
    # 'AdaBoost'
    ]

N = 100
modelNames += [ 'RF%03d'%i for i in range(N) ]

modelFiles = [  ('../../models1/'+n+'_model.pkl') for n in modelNames]

def createSubmission(meanPreds, fileName='result.csv'):
    result = pd.DataFrame({'Action': meanPreds, 'Id':(np.arange(len(meanPreds))+1)})
    result = result[['Id', 'Action']]
    print result.head()
    result.to_csv(fileName, index=False)

    return

def ROC(y, preds):
    fpr, tpr, thresholds = metrics.roc_curve(y, preds)
    roc_auc              = metrics.auc(fpr, tpr)
    return roc_auc

def atan(x, w=1, c=0):
    '''
        w = width parameter
        c = center
    '''
    x = x*2 -1
    v = np.arctan( 2*(x-c)/w )
    v = (v - v.min())/( v.max() - v.min() )
    return v

if __name__ == '__main__':

    X_test = np.load('data/X_test.npy')
    X      = np.load('data/X.npy')
    y      = np.load('data/y.npy')

    print 'Encoding the data ...'
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X = encoder.transform(X) 



    totalS = []
    for w in np.logspace(-1, 1, 5):
        temp = []
        for c in [-0.75, -0.5, 0, 0.5, 0.75]:

            print
            print w, c
            
            # Now we predict on X rather than X_test
            allPredicts = []
            models = [joblib.load(m) for m in modelFiles]
            for i, (m, modelName) in enumerate(zip(models, modelNames)):
                print i,
                pred  = m.predict_proba(X)[:, 1]
                allPredicts.append(pred)

            allPredicts = np.array(allPredicts).T
            meanPreds   = allPredicts.mean(axis=1)

            temp.append(ROC(y, meanPreds))

        totalS.append( temp )

    totalS = np.array( totalS )

    plt.imgshow(totalS)
    plt.show()




    # tmStr = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
    # createSubmission(meanPreds, fileName='predictions/mean_%s.csv'%tmStr)




    print 'done'
