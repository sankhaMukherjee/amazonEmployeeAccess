
import pandas as pd 
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt

def appendCounts(dfTotal, feature):
    '''
        This function adds the count of a feature to 
        the dataframe ...
    '''
    dfTotal['count-'+feature] = 0
    for k, v in  dfTotal.groupby(feature):
        dfTotal['count-'+feature].ix[v.index] = len(v)
    return dfTotal 

def appendAllCounts(dfTotal):
    '''
        This function adds the counts of all the 
        features to the DataFrame ...
    '''
    for feature in dfTotal.columns:
        print 'Feature Count: ', feature
        dfTotal = appendCounts( dfTotal, feature )    

    return dfTotal

def mergeFeatures(dfTotal, features):
    '''

    ##
    ## We want to count by combinations of features ... 
    ## For example, if we want to know how many of a 
    ## particular resource is used by a particular manager,
    ## we can simply group managers and resources together
    ## and count unique values fro these ...
    ##
    ## This can then be easily generalized ...
    ##
    
    '''

    fMerge = '-'.join(features)
    dfTotal['count-'+fMerge] = 0
    for i, (k, v) in enumerate(dfTotal.groupby(features)):
        dfTotal[ 'count-'+fMerge ].ix[v.index] = len(v)

    # This assumes that the counts for a particular 
    # feature is already present ...
    ################################################################
    for f in features:
        temp = np.array(dfTotal['count-'+fMerge])*100.0/np.array(dfTotal['count-'+f])

        dfTotal['count-'+fMerge+'-by-'+f] = map(int, temp)
    
    return dfTotal

'''
    These pairs have a correlation coefficient of greater than 
    0.8 ...

    [('count-ROLE_FAMILY', 'count-ROLE_TITLE'),
     ('count-ROLE_TITLE-RESOURCE', 'count-ROLE_FAMILY_DESC-RESOURCE'),
     ('count-ROLE_FAMILY-RESOURCE', 'count-ROLE_TITLE-RESOURCE'),
     ('count-ROLE_ROLLUP_1-RESOURCE', 'count-RESOURCE'),
     ('count-ROLE_ROLLUP_1-RESOURCE-by-ROLE_ROLLUP_1',
      'count-ROLE_ROLLUP_2-RESOURCE-by-ROLE_ROLLUP_2'),
     ('ROLE_ROLLUP_2', 'ROLE_ROLLUP_1'),
     ('count-ROLE_FAMILY_DESC-RESOURCE', 'count-ROLE_TITLE-RESOURCE'),
     ('count-ROLE_TITLE-RESOURCE', 'count-ROLE_FAMILY-RESOURCE'),
     ('count-ROLE_ROLLUP_2-RESOURCE-by-ROLE_ROLLUP_2',
      'count-ROLE_ROLLUP_1-RESOURCE-by-ROLE_ROLLUP_1'),
     ('ROLE_ROLLUP_1', 'ROLE_ROLLUP_2'),
     ('count-RESOURCE', 'count-ROLE_ROLLUP_1-RESOURCE'),
     ('count-ROLE_TITLE', 'count-ROLE_FAMILY')]
'''

if __name__ == '__main__':

    plt.ion()

    fileNames = [
        'data/test.csv',
        'data/train.csv'
    ]

    dfTest, dfTrain = map(pd.read_csv, fileNames)

    print 'Test:', ', '.join(list(dfTest.columns))
    print 'Train', ', '.join(list(dfTrain.columns))
    print 
    print len(dfTest)
    print len(dfTrain)

    N = len(dfTrain)

    dfTotal = pd.concat( 
            [   dfTrain.drop(['ACTION', 'ROLE_CODE'], axis=1),
                dfTest.drop(['id', 'ROLE_CODE'], axis=1) ],
                ignore_index=True )

    # relabeling the factors ...
    for c in list(dfTotal.columns):
        temp, _ = pd.factorize(dfTotal[c])
        dfTotal[c] = temp

    dfTotal = appendAllCounts(dfTotal)

    
    features = ['MGR_ID', 'RESOURCE']
    combinations = combinations = [ 
        'MGR_ID',
        'ROLE_ROLLUP_1',
        'ROLE_ROLLUP_2',
        'ROLE_DEPTNAME',
        'ROLE_TITLE',
        'ROLE_FAMILY_DESC',
        'ROLE_FAMILY']

    combinations = [ [c, 'RESOURCE'] for c in combinations]

    for c in combinations:
        print 'Now doing:', ', '.join(c)
        dfTotal = mergeFeatures(dfTotal, c)

    # print dfTotal.head()
    # Plot the clustermapif you like ...
    # gc = sns.clustermap(dfTotal.corr())
    # for text in gc.ax_heatmap.get_yticklabels():
    #     text.set_rotation('horizontal')

    # for text in gc.ax_heatmap.get_xticklabels():
    #     text.set_rotation('vertical')

    # Find the lists that are highly correlated (correlation >= 0.8)
    temp = dfTotal.corr()
    allCorrs = []
    for c in temp.columns:
        ms = list(temp[c][ temp[c]>0.8  ].index)
        ms = [(c, m) for m in ms if c != m]
        allCorrs += ms

    allCorrs = list(set(allCorrs))
    print allCorrs


    # These parameters are correlated to other parameters with a
    # correlation codeeicient of greater than 0.8
    toRemove = ['ROLE_ROLLUP_1', 
        'count-ROLE_ROLLUP_1-RESOURCE',
        'count-ROLE_TITLE',
        'count-ROLE_TITLE-RESOURCE',
        'count-ROLE_ROLLUP_1-RESOURCE-by-ROLE_ROLLUP_1']

    # Create the array and save them ...
    for tR in toRemove:
        dfTotal.drop(tR, axis=1, inplace=True)

    dfTotalArr = np.array(dfTotal)
    XtrainExt = dfTotalArr[ :N , :]
    XtestExt  = dfTotalArr[ N: , :]

    np.save('data/XtrainExt.npy', XtrainExt)
    np.save('data/XtestExt.npy',  XtestExt)
    np.save('data/y.npy',         dfTrain.ACTION.values)


    
    plt.show()

    print 'done'


