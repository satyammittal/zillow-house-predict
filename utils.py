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
	airConditionMode = properties['airconditioningtypeid'].value_counts().argmax()
	properties['airconditioningtypeid'] = properties['airconditioningtypeid'].fillna(airConditionMode)

	architectureMode = properties['architecturalstyletypeid'].value_counts().argmax()
	properties['architecturalstyletypeid'] = properties['architecturalstyletypeid'].fillna(architectureMode)

	basementSqFeetAverage = properties['basementsqft'].mean()
	properties['basementsqft'] = properties['basementsqft'].fillna(basementSqFeetAverage)

	bathroomCntMode = properties['bathroomcnt'].value_counts().argmax()
	properties['bathroomcnt'] = properties['bathroomcnt'].fillna(bathroomCntMode)

	bedroomCntMode = properties['bedroomcnt'].value_counts().argmax()
	properties['bedroomcnt'] = properties['bedroomcnt'].fillna(bedroomCntMode)

	buildingClassType = properties['buildingclasstypeid'].value_counts().argmax()
	properties['buildingclasstypeid'] = properties['buildingclasstypeid'].fillna(buildingClassType)

	buildingQualityType = properties['buildingqualitytypeid'].value_counts().argmax()
	properties['buildingqualitytypeid'] = properties['buildingqualitytypeid'].fillna(buildingQualityType)

	calculatedBathnBedroom = properties['calculatedbathnbr'].value_counts().argmax()
	properties['calculatedbathnbr'] = properties['calculatedbathnbr'].fillna(calculatedBathnBedroom)

	# Making deck type a binary label
	properties['decktypeid'] = properties['decktypeid'].fillna(0) 
	properties['decktypeid'] = properties['decktypeid'].replace(66,1)

	floor1SqFeetAverage = properties['finishedfloor1squarefeet'].mean()
	properties['finishedfloor1squarefeet'] = properties['finishedfloor1squarefeet'].fillna(floor1SqFeetAverage)

	calculatedSqFeetAverage = properties['calculatedfinishedsquarefeet'].mean()
	properties['calculatedfinishedsquarefeet'] = properties['calculatedfinishedsquarefeet'].fillna(calculatedSqFeetAverage)

	finishedSqFeet6 = properties['finishedsquarefeet6'].mean()
	properties['finishedsquarefeet6'] = properties['finishedsquarefeet6'].fillna(finishedSqFeet6)

	finishedSqFeet12 = properties['finishedsquarefeet12'].mean()
	properties['finishedsquarefeet12'] = properties['finishedsquarefeet12'].fillna(finishedSqFeet12)

	finishedSqFeet13 = properties['finishedsquarefeet13'].mean()
	properties['finishedsquarefeet13'] = properties['finishedsquarefeet13'].fillna(finishedSqFeet13)

	finishedSqFeet15 = properties['finishedsquarefeet15'].mean()
	properties['finishedsquarefeet15'] = properties['finishedsquarefeet15'].fillna(finishedSqFeet15)

	finishedSqFeet50 = properties['finishedsquarefeet50'].mean()
	properties['finishedsquarefeet50'] = properties['finishedsquarefeet50'].fillna(finishedSqFeet50)

	fips = properties['fips'].value_counts().argmax()
	properties['fips'] = properties['fips'].fillna(fips)

	# Making fireplace count\ a binary label
	properties['fireplacecnt'] = properties['fireplacecnt'].replace([2,3,4,5,6,7,8,9],1)
	properties['fireplacecnt'] = properties['fireplacecnt'].fillna(0) 

	fullCntBathMode = properties['fullbathcnt'].value_counts().argmax()
	properties['fullbathcnt'] = properties['fullbathcnt'].fillna(fullCntBathMode)

	garageCntMode = properties['garagecarcnt'].value_counts().argmax()
	properties['garagecarcnt'] = properties['garagecarcnt'].fillna(garageCntMode)

	garageSqFeetMode = properties['garagetotalsqft'].value_counts().argmax()
	properties['garagetotalsqft'] = properties['garagetotalsqft'].fillna(garageSqFeetMode)

	# Making hot tub a binary label
	properties['hashottuborspa'] = properties['hashottuborspa'].replace(True,1)
	properties['hashottuborspa'] = properties['hashottuborspa'].fillna(0)

	heatingMode = properties['heatingorsystemtypeid'].value_counts().argmax()
	properties['heatingorsystemtypeid'] = properties['heatingorsystemtypeid'].fillna(heatingMode)

	latitudeMax = properties['latitude'].value_counts().argmax()
	properties['latitude'] = properties['latitude'].fillna(latitudeMax)

	longitudeMax = properties['longitude'].value_counts().argmax()
	properties['longitude'] = properties['longitude'].fillna(longitudeMax)

	lotSizeMode = properties['lotsizesquarefeet'].value_counts().argmax()
	properties['lotsizesquarefeet'] = properties['lotsizesquarefeet'].fillna(lotSizeMode)

	# Making pool a binary label
	properties['poolcnt'] = properties['poolcnt'].fillna(0)

	properties['poolsizesum'] = properties['poolsizesum'].fillna(0)

	# These properties show through with the previous features
	#properties = properties.drop('pooltypeid10', axis=1)
	#properties = properties.drop('pooltypeid2', axis=1)
	properties = properties.drop('pooltypeid7', axis=1)

	# Why would these even impact the price (but idk, maybe they're important)?
	properties = properties.drop('propertycountylandusecode', axis=1)	
	properties = properties.drop('propertylandusetypeid', axis=1)
	properties = properties.drop('propertyzoningdesc', axis=1)
	properties = properties.drop('rawcensustractandblock', axis=1)
	properties = properties.drop('censustractandblock', axis=1)

	properties['regionidcounty'] = properties['regionidcounty'].replace([3101, 1286, 2061],[0,1,2])
	properties['regionidcounty'] = properties['regionidcounty'].fillna(0) 

	# No idea how to handle these features
	properties = properties.drop('regionidcity', axis=1)
	properties = properties.drop('regionidzip', axis=1)
	properties = properties.drop('regionidneighborhood', axis=1)

	# Don't bedroom and bathroom counts already do this?
	properties = properties.drop('roomcnt', axis=1)
	properties = properties.drop('threequarterbathnbr', axis=1)

	# Making story type a binary label
	properties['storytypeid'] = properties['storytypeid'].fillna(0) 
	properties['storytypeid'] = properties['storytypeid'].replace(7,1)

	# Only has like a couple thousand non NA values, so not worth
	properties = properties.drop('typeconstructiontypeid', axis=1)

	unitMode = properties['unitcnt'].value_counts().argmax()
	properties['unitcnt'] = properties['unitcnt'].fillna(unitMode)

	yardSqFt17 = properties['yardbuildingsqft17'].mean()
	properties['yardbuildingsqft17'] = properties['yardbuildingsqft17'].fillna(yardSqFt17)

	yardSqFt26 = properties['yardbuildingsqft26'].mean()
	properties['yardbuildingsqft26'] = properties['yardbuildingsqft26'].fillna(yardSqFt26)

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
	

	# Idk, I don't wanna deal with these
	properties = properties.drop('assessmentyear', axis=1)
	properties = properties.drop('taxdelinquencyflag', axis=1)
	properties = properties.drop('taxdelinquencyyear', axis=1)

	#Have to normalize the data now

	properties = properties.drop(['yardbuildingsqft26', 'yardbuildingsqft17',
		'storytypeid','pooltypeid2','pooltypeid10','poolsizesum','hashottuborspa','finishedsquarefeet6','finishedsquarefeet13',
		'decktypeid','buildingclasstypeid','basementsqft','architecturalstyletypeid'],
		axis=1)
	# #living area proportions 
	# properties['living_area_prop'] = properties['calculatedfinishedsquarefeet'] / properties['lotsizesquarefeet']

	# #tax value ratio
	# properties['value_ratio'] = properties['taxvaluedollarcnt'] / properties['taxamount']
	# Tax amount already does this
	properties = properties.drop('taxvaluedollarcnt', axis=1)

	# #tax value proportions
	# properties['value_prop'] = properties['structuretaxvaluedollarcnt'] / properties['landtaxvaluedollarcnt']

	# #combination of longitude and latitude 
	# properties['location'] = properties['latitude'] + properties['longitude']

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
	#print 'Running SVM'
	#model = svm.SVR()
	#predicted = cross_val_score(model, xTrain, yTrain, scoring='neg_mean_absolute_error', cv=10)
	#allModels[model] = predicted.mean()

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

	#print 'Running Neural Network'
	#allModels[model] = neuralNetwork()

	# Return the best model
	sortedModels = sorted(allModels.items(), key=operator.itemgetter(1), reverse=True)
	for model in sortedModels:
	    print 'Model:', model[0]
	    print 'Loss:', model[1]

	return sortedModels[0][0]
