

### Changes in this model 

This model is a complete do-over. We neeed to rethink our strategy for this one. 

<svg width="400" height="180">
  <rect x="50" y="20" rx="20" ry="20" width="150" height="150"
  style="fill:red;stroke:black;stroke-width:5;opacity:0.5" />
</svg>

1. Feature generation is done in a separate files. 
    
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

### Lessons Learned:
  a. Intelligent feature generation immediately improved the ranking from 800
     to 500. Intelligent feature selection is very important. 
  b. *Need to check if deleting highly correlated features made a difference*
  c. *Need to check if refactoring made a difference*
  d. Decision Trees and Random Forests mostly overfit data. However, they are 
     very good ensemble methods. Hence, averaging descisions from a lot of 
     Random Forests is a very good way of reducing the overfitting inherent in 
     them.
  e. Always perform the final training on the entire dataset before making a 
     prediction ...
  f. Some models are faster to train ()

### Things to check in the Future:
  a. Does adding features change things
  b. How ot effectively do a grid search