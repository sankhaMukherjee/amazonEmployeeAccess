'''
    This program is going to be used for stacking models that 
    have been previously saved as pickle object. 
'''
import numpy as np
import pandas as pd 

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
    
N = 40
modelNames += [ 'RF%03d'%i for i in range(N) ]

modelFiles = [  ('../../models1/'+n+'_model.pkl') for n in modelNames]


def createSubmission(meanPreds, fileName='result.csv'):
    result = pd.DataFrame({'Action': meanPreds, 'Id':(np.arange(len(meanPreds))+1)})
    result = result[['Id', 'Action']]
    print result.head()
    result.to_csv(fileName, index=False)

    return

if __name__ == '__main__':

    X_test = np.load('data/X_test.npy')
    X      = np.load('data/X.npy')

    print 'Encoding the data ...'
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X_test = encoder.transform(X_test) 


    allPredicts = []
    models = [joblib.load(m) for m in modelFiles]
    for m, modelName in zip(models, modelNames):
        print modelName
        tmStr = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
        pred  = m.predict_proba(X_test)[:, 1]
        createSubmission(pred, fileName='predictions/%s_%s.csv'%(tmStr, modelName))
        allPredicts.append(pred)

    allPredicts = np.array(allPredicts).T
    meanPreds   = allPredicts.mean(axis=1)

    tmStr = dt.now().strftime('%Y-%m-%d-%H-%M-%S')

    createSubmission(meanPreds, fileName='predictions/mean_%s.csv'%tmStr)




    print 'done'
