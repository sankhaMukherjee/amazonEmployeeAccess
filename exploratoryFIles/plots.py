import matplotlib.pyplot as plt
import seaborn as sns

from   sklearn.learning_curve import learning_curve
import numpy as np
import pandas as pd 
from   sklearn import metrics

def plot_trainingCurve(model, X, y, N=5, title=None):
    '''
        This function will plot the training curve. This is 
        useful to see if we have a bias/variance problem at
        our hands ...
    '''

    plt.figure(figsize=(4, 3))

    if title is None:
        plt.axes([0.17, 0.18, 0.94-0.17, 0.96-0.18])
    else:
        plt.axes([0.17, 0.18, 0.94-0.17, 0.75-0.18])

    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, 
        train_sizes = np.linspace(0.1, 1.0, N), 
        cv=5)

    meanTraining = train_scores.mean(axis=1)
    stdTraining  = train_scores.std(axis=1)
    meanValid = valid_scores.mean(axis=1)
    stdValid  = valid_scores.std(axis=1)

    train_sizes /= 1000

    plt.plot(train_sizes, 
        meanTraining, 's-', mfc='red', mec='black', 
        color='red', label='training')

    plt.fill_between( train_sizes, 
        meanTraining+stdTraining*3, 
        meanTraining-stdTraining*3, 
        facecolor='red', alpha=0.5)

    plt.plot(train_sizes, meanValid, 
        's-', mfc='green', mec='black', 
        color='green', label='validation')

    plt.fill_between( train_sizes, 
        meanValid+stdValid*3, 
        meanValid-stdValid*3, 
        facecolor='green', alpha=0.5)

    plt.xlabel(r'size / 1000')
    plt.ylabel(r'accuracy')

    if title is not None:
        plt.title(title)

    return

def plot_roc(fpr, tpr, roc_auc, newPlot=True, color='blue', label=None):
    '''
        This is just going to plot the 
        roc curve. 
    '''

    if newPlot:
        plt.figure(figsize=(4, 3))
        plt.axes([0.17, 0.18, 0.94-0.17, 0.96-0.18])

    if label is None:
        plt.plot(fpr, tpr, color=color)
    else:
        plt.plot(fpr, tpr, color=color, label=label)

    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')

    return

def plot_distribution(y_cv, preds):
    '''
        This just plots a violin plot showing the distribution 
        of the data between zero and one. Depicts data for a 
        zero-one criterion
    '''

    result = []
    result.append(pd.DataFrame( {'values': y_cv,  'type':'actual'} ))
    result.append(pd.DataFrame( {'values': preds, 'type':'predicted'} ))
    result = pd.concat(result)


    plt.figure(figsize=(4, 3))
    plt.axes([0.17, 0.18, 0.94-0.17, 0.96-0.18])
    sns.violinplot(data=result, x='type', y='values')

    return

def plot_skew(y_cv, preds, N = 50, Nmax = 20, start=0, detailed=True):
    '''
        plot the roc curves with different skews
        to see what the distribution of the data 
        is ...
    '''
    powers = np.linspace(start, N, Nmax)[1:]
    aucs   = []
    
    if detailed:
        plot_distribution(y_cv, preds**N)
    
    for xx, i in enumerate(powers):
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds**i)
        roc_auc = metrics.auc(fpr, tpr)
        if detailed:
            plot_roc(fpr, tpr, roc_auc, newPlot=(xx==0), label='%.1f'%i, color=(i/N,0.5,1-i/N))
        aucs.append( roc_auc )

    if detailed:
        plt.legend()

    
    plt.figure(figsize=(4, 3))
    plt.axes([0.17, 0.18, 0.94-0.17, 0.96-0.18])
    plt.plot(powers, aucs, 's')
    intPowers = np.linspace(start, N, 100)[1:]

    # plt.plot(intPowers, np.poly1d(np.polyfit(powers, aucs, 2))( intPowers ), color='black' )
    plt.xlabel('power')
    plt.ylabel('AUC')

    return powers, aucs
