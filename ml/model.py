from ml.data import process_train_data

import pickle
import json
from sklearn.metrics import median_absolute_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor


def train_model(X_train, y_train):
	"""
	Function to train HGB model
	
	ACCEPTS:
		X_train - Training dataset predictors
		y_train - Training dataset values
	
	RETURNS:
		model - Trained HGB regressor model
	"""
	print('Training model')
	
	# Create regressor
	regressor = HistGradientBoostingRegressor(
		loss='absolute_error', 
		random_state=42, 
		learning_rate=0.5,
		max_iter=500,
		max_leaf_nodes=100,
		max_depth=100,
		min_samples_leaf=100)
	
	# Fit regressor with training data
	model = regressor.fit(X_train, y_train)
	
	return model


def compute_model_metrics(X, y, preds, model):
	"""
	Compute r-squared score as well as median and mean absolute error
	
	ACCEPTS:
		X - X_test for r2_score
		y - y_test array
		preds - prediction array from model output
		model - Regression model to calculate metrics
	
	RETURNS:
		mdae - Median absolute error
		mae - Mean absolute error
		r2_score - R-squared score
	"""
	print('Computing model metrics')
	
	# Calculate metrics
	mdae = median_absolute_error(y, preds)
	mae = mean_absolute_error(y, preds)
	r2_score = model.score(X, y)
	
	return mdae, mae, r2_score
	

def inference(model, X, test_set=True):
	"""
	Run dataset through model for predictions
	
	ACCEPTS:
		model - Trained ML model to run predictions
		X - Dataset to run through model
		test_set - Boolean; Used for terminal output to tell user which dataset the model is predicting values for
		
	RETURNS:
		preds - array of predictions on X through model
	"""
	
	# Tell user if the model is in test mode to compute metrics or predicting values for user dataset
	if test_set:
		print('Predicting prices of testing dataset')
	else:
		print('Predicting prices of user dataset')
	
	# Predict values
	preds = model.predict(X)
	
	return preds
	

def save_model(model, path):
	"""
	Saves model at specified path using pickle
	
	ACCEPTS:
		model - Model (or encoder) to save
		path - path to save model
	
	RETURNS:
		Nothing
	"""
	# print('Saving model')
	pickle.dump(model, open(path, 'wb'))
	

def load_model(path):
	"""
	Loads model (or encoder) from path
	
	ACCEPTS:
		path - Filepath to load from
	
	RETURNS:
		model - Item found at path
	"""
	# print('Loading model')
	model = pickle.load(open(path, 'rb'))
	
	return model
	

def save_metrics(metrics, path):
	"""
	Saves JSON file of metrics at path
	
	ACCEPTS:
		metrics - JSON serializable variable to save at path
		path - Filepath to save metrics at
		
	RETURNS:
		Nothing
	"""
	# print('Saving metrics')
	json.dump(metrics, open(path, 'w'))

	
def load_metrics(path):
	"""
	Loads JSON file from path
	
	ACCEPTS:
		path - Filepath to load from
	
	RETURNS:
		metrics - JSON file found at path
	"""
	# print('Loading metrics')
	metrics = json.load(open(path, 'r'))
	return metrics
	


