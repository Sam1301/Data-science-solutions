import numpy as np
import xgboost as xgb
import pandas as pd

# reading data from train.csv and test.csv
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

featureNames = train.columns[:-1]

xg_train = xgb.DMatrix(train[featureNames].values, label=train['price_range'].values)
xg_test = xgb.DMatrix(test[featureNames].values)

# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['num_class'] = 4 # number of classes
param['max_depth'] = 7
param['eta'] = 0.3
param['silent'] = 1
param['nthread'] = 4
num_round = 90 # nnumber of booster rounds

bst = xgb.train(param, xg_train, num_round)

# get prediction
pred = bst.predict(xg_test).astype(int)

# write submission.csv
pd.DataFrame({'id': test['id'], 'price_range': pred}).to_csv('submission.csv', index=False)
