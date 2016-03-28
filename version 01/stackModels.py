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

'''
                            2nd order      normal
    'Logistic_Reg',          0.86          0.875       
    'KNN_13'                 0.86          0.858
    'SVC_Linear',            0.846791      0.857
    # 'SVC',                 0.5           0.869
    # 'DecisionTree',        0.6823409     0.754
    'RandomForest',          0.85211       0.862
    # 'RandomForest1',       #0.8146       0.831
    'AdaBoost',              0.75059       0.76
    'ExtraTrees',            0.85997       
    'ExtraTrees1',           0.8611        0.833
                           
'''


modelNames = [
    'Logistic_Reg',    # 
    'SVC_Linear',      # 
    # 'SVC',           # 
    # 'DecisionTree',  # 
    # 'RandomForest',    # 
    # 'RandomForest1', # some problem here for poly features ... 
    'AdaBoost',        # 
    # 'ExtraTrees',      # 
    'ExtraTrees1',     # 
    ]

modelFiles = [  ('../../models/'+n+'_model.pkl').replace(' ', '\ ') for n in modelNames]

def AUC(y, predictions):
    fpr, tpr, thresholds = metrics.roc_curve(y, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def predictionArray(models, X, verbose=True):
    '''
        This function takes a list of models, predicts 
        the values of each model in turn, and returns 
        an array of predicted values ...
    '''

    allPredicts = []
    for i, model in enumerate(models):

        if verbose: 
            print 'Model :', i+1, 'of', len(models)

        preds = model.predict_proba(X)[:, 1]
        allPredicts.append(preds)

    allPredicts = np.array(allPredicts).T    

    return allPredicts

def createSubmission(meanPreds, fileName='result.csv'):
    result = pd.DataFrame({'Action': meanPreds, 'Id':(np.arange(len(meanPreds))+1)})
    result = result[['Id', 'Action']]
    print result.head()
    result.to_csv(fileName, index=False)

    return

def logTransform(predictions):
    predictions1 = predictions.copy()
    predictions1[ predictions1 > 0.9999999  ] = 0.9999999
    predictions1[ predictions1 < 0.0000001  ] = 0.0000001
    predictions1 = -np.log( (1-predictions1)/predictions1  )
    return predictions1

if __name__ == '__main__':

    X_ps = np.load('data/X_ps.npy')
    y_ps = np.load('data/y_ps.npy')

    X_test = np.load('data/X_test.npy')
    X      = np.load('data/X.npy')

    print 'Encoding the data ...'
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X_ps   = encoder.transform(X_ps) 
    X_test = encoder.transform(X_test) 


    models = [joblib.load(m) for m in modelFiles]

    allPredicts = predictionArray(models, X_ps)
    meanPreds   = allPredicts.mean(axis=1)

    print 'mean Predicted AUC: ', AUC(y_ps, meanPreds)

    lr      = linear_model.LinearRegression()
    model   = lr.fit(logTransform(allPredicts), y_ps)
    lrPreds = model.predict(logTransform(allPredicts))

    print 'lr Predicted AUC: ', AUC(y_ps, lrPreds)

    # This is where we train the entire test set
    allPredicts = predictionArray(models, X_test)
    meanPreds   = allPredicts.mean(axis=1)
    lrPreds     = model.predict(logTransform(allPredicts))

    tmStr = dt.now().strftime('%Y-%m-%d-%H-%M-%S')

    createSubmission(meanPreds, fileName='predictions/mean_%s.csv'%tmStr)
    createSubmission(lrPreds, fileName='predictions/lr_%s.csv'%tmStr)




    print 'done'
