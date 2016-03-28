# Changes in this model 

1. Feature generation is dont in a separate files. 
    
    `interestingFeatureS.py`

    This file generates interesting features. Hopefully this is going to be better. It does the 
    following differently:

    1. Refactors all the features
    2. Generates features based on counts of different features, 
    3. Generates features based on counts of a combination of 
       `'RESOURCE'`s and other features and takes percentages 
       WRT counts of both the `'RESOURCE'`s and the particular 
       feature. 
    4. It looks for correlations of different features, and tried 
       to determine if different features are correlated by greater 
       than 80%. If they are, then some features are eliminated 
       to reduce the cirrelations. 
    5. It is directly going to create the arrays `X`, `X_test` and 
       `y` so subsequent programs can directly read the arrays ...


2. `createModels.py` now workds differently. 

    1. Use the other program to generate features 
    and directly load the numpy arrays over here. 
    There is no need for loading features directly 
    from the csv files ...

    2. Concentrate on only two different models rather 
    than a lot of different models. The `LogisticRegression`
    and the `RandomForest` classifiers ...

3. `stackModels.py` Here, we can no longer do a linear fit for the 
   stacking parameters, simply because we have trained for the entire 
   dataset rather than over a limited amount of the data ...

   We now save prediction files for _all_ the models along with the 
   prediction file for the mean model ...


