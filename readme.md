# House Price Prediction
## Preprocess
In preprocess.py, we smooth the numerical data points by normalizing (function: normalize). Also, we transfer categorical data points to one hot vectors in buildmodels.py. Then, we can concatenate one hot vectors to get a sparse vector which can be added after numerical data points to build our x_train. Models can be established here in buildmodels.py. numerical.csv and categorical.csv are columns splitted by numerical data points and categorical data points obtained from preprocess.py.
###TODOs for Tang:
Following instructions in https://towardsdatascience.com/mercari-price-suggestion-97ff15840dbd, section 4.2 seperates our dataset into two parts, x_train and x_cv, respectly y_train and y_cv, to get the best alpha in Ridge model. A naive approach has been implemented in buildmodels.py. When we get the best alpha, apply the Ridge model with the best alpha to x_test in test.csv to predict house prices and submit to get a score. Evaluations should follow the section 6.1 in https://towardsdatascience.com/mercari-price-suggestion-97ff15840dbd.
##TODOs for All:
###Sunday 12.05
1. build LightGBM Regression model and optimize input.
2. abstract and introduction
3. related works(references)
4. each one of us explain one model. (methdology part)
###Monday 12.06
1. conclusion discussion