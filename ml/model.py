from ml.data import process_train_data

import pickle
import json
from sklearn.metrics import median_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor


def train_model(X_train, y_train):

	print('Training model')
	regressor = HistGradientBoostingRegressor(
		loss='absolute_error', 
		random_state=42, 
		categorical_features=['state', 'status'])
	model = regressor.fit(X_train, y_train)
	
	return model


def compute_model_metrics(y, preds, model, X):

	print('Computing model metrics')
	mdae = median_absolute_error(y, preds)
	r2_score = model.score(X, y)
	
	return mdae, r2_score
	

def inference(model, X, test_set=True):
	
	if test_set:
		print('Predicting prices of testing dataset')
	else:
		print('Predicting prices of user dataset')
	preds = model.predict(X)
	return preds
	

def save_model(model, path):
	
	print('Saving model')
	pickle.dump(model, open(path, 'wb'))
	

def load_model(path):

	print('Loading model')
	model = pickle.load(open(path, 'rb'))
	return model
	

def save_metrics(metrics, path):
	
	print('Saving metrics')
	json.dump(metrics, open(path, 'w'))

	
def load_metrics(path):
	print('Loading metrics')
	metrics = json.load(open(path, 'r'))
	return metrics
	


