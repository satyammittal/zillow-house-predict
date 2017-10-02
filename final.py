import pandas as pd 
import numpy as np 
import csv
import sklearn
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
import os.path
from sklearn import linear_model
import pandas as pd 
import numpy as np 
import csv
import operator
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor
import keras
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.cross_validation import StratifiedKFold

# Preprocess the data
def cleanData(properties):
	for feature_name in ['airconditioningtypeid','bathroomcnt', 'bedroomcnt', 'buildingclasstypeid', 'buildingqualitytypeid', 'calculatedbathnbr','fullbathcnt', 'garagecarcnt','garagetotalsqft','heatingorsystemtypeid','lotsizesquarefeet','fips']
		Feature_max = properties[feature_name].value_counts().argmax()
		properties[feature_name] = properties[feature_name].fillna(Feature_max)

	print('Memory usage reduction...')
	latitudeMax = properties['latitude'].value_counts().argmax()
	properties['latitude'] = properties['latitude'].fillna(latitudeMax)

	longitudeMax = properties['longitude'].value_counts().argmax()
	properties['longitude'] = properties['longitude'].fillna(longitudeMax)
	properties[['latitude', 'longitude']] /= 1e6
	properties['censustractandblock'] /= 1e12
	properties = properties.drop(['yardbuildingsqft26', 'yardbuildingsqft17',
		'storytypeid','pooltypeid2','pooltypeid10','poolsizesum','hashottuborspa','finishedsquarefeet6','finishedsquarefeet13',
		'decktypeid','buildingclasstypeid','basementsqft','architecturalstyletypeid'],
		axis=1)


	for feature_name in ['finishedfloor1squarefeet', 'calculatedfinishedsquarefeet', 'finishedsquarefeet12','finishedsquarefeet15','finishedsquarefeet50']:
		Feature_mean = properties[feature_name].mean()
		properties[feature_name] = properties[feature_name].fillna(Feature_mean)


	# Making fireplace count\ a binary label
	#properties['fireplacecnt'] = properties['fireplacecnt'].replace([2,3,4,5,6,7,8,9],1)
	properties['fireplacecnt'] = properties['fireplacecnt'].fillna(0) 

	# Making pool a binary label
	properties['poolcnt'] = properties['poolcnt'].fillna(0)

	properties = properties.drop('pooltypeid7', axis=1)

	# Why would these even impact the price (but idk, maybe they're important)?
	properties = properties.drop('propertycountylandusecode', axis=1)	
	properties = properties.drop('propertylandusetypeid', axis=1)
	properties = properties.drop('propertyzoningdesc', axis=1)
	#properties = properties.drop('rawcensustractandblock', axis=1)
	#properties = properties.drop('censustractandblock', axis=1)

	properties['regionidcounty'] = properties['regionidcounty'].replace([3101, 1286, 2061],[0,1,2])
	properties['regionidcounty'] = properties['regionidcounty'].fillna(0) 

	properties = properties.drop('threequarterbathnbr', axis=1)

	properties = properties.drop('typeconstructiontypeid', axis=1)

	unitMode = properties['unitcnt'].value_counts().argmax()
	properties['unitcnt'] = properties['unitcnt'].fillna(unitMode)

	yearBuilt = properties['yearbuilt'].mean()
	properties['yearbuilt'] = properties['yearbuilt'].fillna(yearBuilt)

	properties['numberofstories'] = properties['numberofstories'].fillna(0)

	# Fireplace count already does this
	properties = properties.drop('fireplaceflag', axis=1)

	structureTax = properties['structuretaxvaluedollarcnt'].mean()
	properties['structuretaxvaluedollarcnt'] = properties['structuretaxvaluedollarcnt'].fillna(structureTax)

	landTax = properties['landtaxvaluedollarcnt'].mean()
	properties['landtaxvaluedollarcnt'] = properties['landtaxvaluedollarcnt'].fillna(landTax)

	tax = properties['taxamount'].mean()
	properties['taxamount'] = properties['taxamount'].fillna(tax)

	properties['tax_per_liv_area']=properties['taxamount']/properties['calculatedfinishedsquarefeet']
	properties['tax_per_liv_area2']=properties['taxamount']/properties['finishedsquarefeet12']
	properties['tax_per_lot_size']=properties['taxamount']/properties['lotsizesquarefeet']

	return properties

# Create xTrain and yTrain
def createTrainingMatrices(properties, labels):
	numTrainExamples = labels.shape[0]
	numFeatures = properties.shape[1]
	xTrain = np.zeros([numTrainExamples, numFeatures])
	yTrain = np.zeros([numTrainExamples])
	arr = []
	propertiesIds = properties['parcelid']
	log_errors = labels['logerror']
	upper_limit = np.percentile(log_errors, 99.5)
	lower_limit = np.percentile(log_errors, 0.5)
	for index, row in labels.iterrows():
		if row['logerror']<upper_limit and row['logerror']> lower_limit:
			xTrain[index] = properties[properties['parcelid'] == row['parcelid']]
			yTrain[index] = row['logerror']
		else:
			arr.append(index)
	xTrain = np.delete(xTrain, arr, axis=0)
	yTrain = np.delete(yTrain, arr)
	return xTrain, yTrain

def createKaggleSubmission(model,model2, properties, cleanedPropertyData):
	XGB_WEIGHT = 0.6415
	BASELINE_WEIGHT = 0.0056
	OLS_WEIGHT = 0.0828

	XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models

	BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg
	lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)
	xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
	baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)
	print 'Test prediction'
	preds = model.predict(properties);
	pred0 = xgb_weight0*model2 + baseline_weight0*BASELINE_PRED + lgb_weight*preds
	print( "\nCombined XGB/LGB/baseline predictions:" )
	print( pd.DataFrame(pred0).head() )
	numTestExamples = properties.shape[0]
	numPredictionColumns = 7
	predictions = []
	for index, pred in enumerate(pred0):
		parcelNum = int(cleanedPropertyData[index][0])
		predictions.append([parcelNum,pred,pred,pred,pred,pred,pred])
	firstRow = [['ParcelId', '201610', '201611', '201612', '201710', '201711', '201712']]
	print 'Writing results to CSV'
	with open("preds.csv", "wb") as f:
	    writer = csv.writer(f)
	    writer.writerows(firstRow)
	    writer.writerows(predictions)


def xgBoost(xTrain, yTrain, xTest):
	y_mean = np.mean(yTrain)
	XGB_WEIGHT = 0.6415
	XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models
	print("\nSetting up data for XGBoost ...")
	# xgboost params
	xgb_params = {
	    'eta': 0.037,
	    'max_depth': 5,
	    'subsample': 0.80,
	    'objective': 'reg:linear',
	    'eval_metric': 'mae',
	    'lambda': 0.8,   
	    'alpha': 0.4, 
	    'base_score': y_mean,
	    'silent': 1
	}

	dtrain = xgb.DMatrix(xTrain, yTrain)
	dtest = xgb.DMatrix(xTest)

	num_boost_rounds = 250
	print("num_boost_rounds="+str(num_boost_rounds))

	# train model
	print( "\nTraining XGBoost ...")
	model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

	print( "\nPredicting with XGBoost ...")
	xgb_pred1 = model.predict(dtest)

	print( "\nFirst XGBoost predictions:" )
	print( pd.DataFrame(xgb_pred1).head() )



	##### RUN XGBOOST AGAIN
	print("\nSetting up data for XGBoost ...")
	# xgboost params
	xgb_params = {
	    'eta': 0.033,
	    'max_depth': 6,
	    'subsample': 0.80,
	    'objective': 'reg:linear',
	    'eval_metric': 'mae',
	    'base_score': y_mean,
	    'silent': 1
	}

	num_boost_rounds = 150
	print("num_boost_rounds="+str(num_boost_rounds))

	print( "\nTraining XGBoost again ...")
	model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

	print( "\nPredicting with XGBoost again ...")
	xgb_pred2 = model.predict(dtest)

	print( "\nSecond XGBoost predictions:" )
	print( pd.DataFrame(xgb_pred2).head() )



	##### COMBINE XGBOOST RESULTS

	xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2
	del dtest
	del dtrain
	del xgb_pred1
	del xgb_pred2 
	return xgb_pred

def neuralNetwork(xTrain, yTrain):
	skf = StratifiedKFold(labels, n_folds=10, shuffle=True)
	loss=[]
	for train, test in kfold.split(xTrain, yTrain):
  		# create model
		model = Sequential()
		model.add(Dense(128, init='normal', input_dim = dim))
		model.add(Activation('relu'))
		model.add(Dropout(0.2))
		model.add(Dense(64, init='normal'))
		model.add(Activation('relu'))
		model.add(Dropout(0.1))
		model.add(Dense(16, init='normal'))
		model.add(Activation('relu'))
		model.add(Dense(1, init='normal'))
		model.add(Activation('softmax'))
		# Compile model
		model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['loss'])
		# Fit the model
		model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
		# evaluate the model
		scores = model.evaluate(X[test], Y[test], verbose=0)
		# TODO Add scores loss to loss list
	return sum(loss)/len(loss)


def baseline_model():
	model = Sequential()
	model.add(Dense(1, input_dim=31, kernel_initializer='glorot_normal'))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(64, init='normal'))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))
	model.add(Dense(16, init='normal'))
	model.add(Activation('relu'))
	model.add(Dense(1, init='normal'))
	model.add(Activation('softmax'))
	model.compile(loss='mean_absolute_error', optimizer='adam')
	return model

def findBestMLModel(xTrain, yTrain):
	allModels = {} # Dictionary of models and their respective losses

	# All of the traditional regression models
	print 'Running Linear Regression'
	model = linear_model.LinearRegression()
	predicted = cross_val_score(model, xTrain, yTrain, scoring='neg_mean_absolute_error', cv=10)
	allModels[model] = predicted.mean()


	# fit model no training data
	# print "Xgbboost"
	# model = XGBClassifier()
	# predicted = cross_val_score(model, xTrain, yTrain, scoring='neg_mean_absolute_error', cv=10)
	# allModels[model] = predicted.mean()
	#print 'Keras Regressor'
	#model = KerasRegressor(build_fn=baseline_model, epochs=30, batch_size=50, verbose=True)
	#predicted = cross_val_score(model, xTrain, yTrain, scoring='neg_mean_absolute_error', cv=10)
	#allModels[model] = predicted.mean()

	print 'Running Bayesian Ridge Regression'
	model = linear_model.BayesianRidge()
	predicted = cross_val_score(model, xTrain, yTrain, scoring='neg_mean_absolute_error', cv=10)
	allModels[model] = predicted.mean()

	print 'Running Ridge Regression'
	model = linear_model.Ridge()
	predicted = cross_val_score(model, xTrain, yTrain, scoring='neg_mean_absolute_error', cv=10)
	allModels[model] = predicted.mean()

	print 'Running Lasso Regression'
	model = linear_model.Lasso()
	predicted = cross_val_score(model, xTrain, yTrain, scoring='neg_mean_absolute_error', cv=10)
	allModels[model] = predicted.mean()

	# SVM 
	print 'Running SVM'
	model = svm.SVR()
	predicted = cross_val_score(model, xTrain, yTrain, scoring='neg_mean_absolute_error', cv=10)
	allModels[model] = predicted.mean()

	# Decision Trees
	print 'Running Decision Trees'
	model = tree.DecisionTreeRegressor()
	predicted = cross_val_score(model, xTrain, yTrain, scoring='neg_mean_absolute_error', cv=10)
	allModels[model] = predicted.mean()

	# Random Forests
	print 'Running Random Forest'
	model = RandomForestRegressor()
	predicted = cross_val_score(model, xTrain, yTrain, scoring='neg_mean_absolute_error', cv=10)
	allModels[model] = predicted.mean()

	# K Nearest Neighbors
	print 'Running KNN'
	model = KNeighborsRegressor()
	predicted = cross_val_score(model, xTrain, yTrain, scoring='neg_mean_absolute_error', cv=10)
	allModels[model] = predicted.mean()

	# Gradient Boosted Methods
	print 'Running Gradient Boosted Regressor'
	model = GradientBoostingRegressor()
	predicted = cross_val_score(model, xTrain, yTrain, scoring='neg_mean_absolute_error', cv=10)
	allModels[model] = predicted.mean()

	# Neural network

	print 'Running Neural Network'
	allModels[model] = neuralNetwork()

	# Return the best model
	sortedModels = sorted(allModels.items(), key=operator.itemgetter(1), reverse=True)
	for model in sortedModels:
	    print 'Model:', model[0]
	    print 'Loss:', model[1]

	return sortedModels[0][0]



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

