'''
    In this file, we start by generating polynomial features
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

def combineCols(X, X_test, cols):
    '''
        This function takes the X and the X_test matrices, and 
        creates a new column by combining the columns specified
        ...
    '''

    X_all = np.vstack((X, X_test))

    vals = map( lambda m: '-'.join(map(str, m)), zip( *(X_all[:,c] for c in cols)) )
    newCol, _ = pd.factorize(  vals  )
    newCol    = np.array(newCol)
    
    splitAt = X.shape[0]
    
    return newCol[:splitAt], newCol[splitAt:]

def polyFeatures(X, X_test, polyOrder=2, verbose=True):
    '''
        Given a set of matrices, we shall add a number of features,
        dependent upon 
    '''

    X_all = np.vstack((X, X_test))

    def colStack( cols ):
        strs = map( lambda m: '-'.join(map(str, m)), zip( *(X_all[:,c] for c in cols)) )
        return strs
    
    if verbose: print 'polynomial order: ', polyOrder

    orders = map(list, combinations(range(np.shape(X)[1]), polyOrder))
    N      = len(orders)

    if verbose: print 'Number of orders: ', N

    allLists = []
    for i, cols in enumerate(orders):
        print i+1, 'of', N, cols 
        vals = colStack( cols )
        newCol, _ = pd.factorize(  vals  )
        allLists.append(newCol)

    allLists = np.array(allLists)
        
    return allLists.T

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

def newFeatureCount(df, featureName, group):
    '''
        Inputs:
            df, 
            featureName --> the feature to count
            group       --> groupby this feature

        For a particular dataframe, this function is 
        going to count the number of times this particular
        feature has occured. This will return a list of int's.

        if a groupBy feature is present, then groupby this 
        feature, and then take the mean of the counts.


        For example, 

        (df, 'RESOURCE', 'MGR_ID')

        will count the total number of resources by manager ...


    '''
    return

def newFeaturePercent(df, featureName, groups, value=0):
    '''
        Inputs:
            df, 
            featureName --> the feature to count
            groups      --> groupby these features

        For a particular dataframe, this function is 
        going to find the percentage of the feature within
        the combination of groups present ...

        if a groupBy feature is present, then groupby this 
        feature, and then take the mean of the counts.


        For example, 

        (df, 'RESOURCE', 'MGR_ID')

        will count the total number of resources by manager ...


    '''
    return


if __name__ == '__main__':

    plt.ion()
    
    y, X      = loadFile('data/train.csv')
    _, X_test = loadFile('data/test.csv')

    X_all = np.vstack((X, X_test))


    for order in range(2, 7):
        features = polyFeatures(X, X_test, polyOrder=order, verbose=True)
        trainF = features[:32769,:]
        testF  = features[32769:,:]

        np.save('data/trainF_%02d.npy'%order, trainF)
        np.save('data/testF_%02d.npy'%order, testF)


    print 'done'
