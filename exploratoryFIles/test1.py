import pandas as pd
# import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
# import seaborn as sns
 
# First, find the unique values of the different samples ..
 
 
def normalize(x):
    return 2*((x - x.min())/(x.max() - x.min()) - 0.5)
 
def findMultiplicity(df):
    '''
        This finds the multiplicity of different
        columns. If a multiplicity is 1, it is going to
        be safe to just neglect one of the categorical
        variables
    '''
 
    # Find the multiplicity ....
    mult = {}
    for c in df.columns:
        temp = df.pivot_table( index=c,
                                  aggfunc=lambda x:len(x.unique()),
                                  fill_value=0).apply(np.max)
        mult[c] = temp
 
    mult1 = pd.DataFrame(mult)
    mult1 = mult1[ sorted(mult1.columns) ]
   
    return mult1

def plotDistribution(df, name=''):
    for c in df.columns:
        countData  = df[c].value_counts()
        plt.figure(figsize=(4,3))
        plt.plot( countData.index, list(countData), '+' )
        plt.title(c)
        plt.yscale('log')
        ymin, ymax = plt.ylim()
        plt.ylim([0, ymax])
        plt.savefig(name + '_' + c + '.png')


if __name__ == '__main__':

    # plt.ion()

    train = pd.read_csv('data/train.csv')
    test  = pd.read_csv('data/test.csv')

    # plotDistribution(train, 'train')
    # plotDistribution(test,   'test')

    # train1 = train.apply(normalize)
    # sns.pairplot(data=train1, hue='ACTION')

    mTrain = findMultiplicity(train)
    mTest = findMultiplicity(test)
     
    mTrain.to_csv('trainM.csv')
    mTest.to_csv('testM.csv')
     
 
    # plt.show()
    print 'done'

