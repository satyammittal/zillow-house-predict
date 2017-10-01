import pandas as pd 
import numpy as np 
import csv
import sklearn
from utils import cleanData 
from utils import createTrainingMatrices 
from utils import createKaggleSubmission
from utils import findBestMLModel
from utils import xgBoost
from xgboost import XGBClassifier

import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
import os.path
from sklearn import linear_model

## Dataset Cleaning TODOs ##
# How do you save a dataframe in Numpy

if (os.path.isfile('xTrain.npy') and os.path.isfile('yTrain.npy') and os.path.isfile('properties.npy')):
	print 'Loading in precomputed xTrain, yTrain, and properties'
	xTrain = np.load('xTrain.npy')
	yTrain = np.load('yTrain.npy')
	propertiesDataFrame = pd.read_csv('properties_2016.csv', low_memory=False)
	properties = np.load('properties.npy')
	cleanedPropertyData = pd.np.array(properties)
else:
	# Load in the data
	print 'Loading in data'
	propertiesDataFrame = pd.read_csv('properties_2016.csv', low_memory=False)
	trainDataFrame = pd.read_csv('train_2016_v2.csv')
	sampleSubDataFrame = pd.read_csv('sample_submission.csv')

	# Clean the data
	print 'Cleaning the data'
	properties = cleanData(propertiesDataFrame)
	cleanedPropertyData = pd.np.array(properties)
	print 'Shape of the cleaned data matrix:', cleanedPropertyData.shape

	print 'Computing xTrain and yTrain'
	xTrain, yTrain = createTrainingMatrices(properties, trainDataFrame)
	print 'Shape of the xTrain matrix:', xTrain.shape
	print 'Shape of the yTrain matrix:', yTrain.shape
	np.save('xTrain', xTrain)
	np.save('yTrain', yTrain)
	np.save('properties', properties)

# Train the model
#print 'Finding the best model'
#model = findBestMLModel(xTrain, yTrain)
#print 'The best model is:', model
print "Getiing xgb Result"
xgb_res = xgBoost(xTrain,yTrain, properties)
print 'Training the model'
print('Training LGBM model...')
ltrain = lgb.Dataset(xTrain, label = yTrain)
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.345    
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3

lgb_model = lgb.train(params, ltrain, verbose_eval=0, num_boost_round=2930)

createKaggleSubmission(lgb_model, xgb_res, properties, cleanedPropertyData)






