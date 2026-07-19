import os

import pandas as pd

from ml.data import process_train_data, load_metadata
from ml.model import train_model, compute_model_metrics, inference, save_model, load_model, save_metrics
import argparse

import kagglehub


def create_model(args):
	"""
	Creates and saves ml model, model metrics, and column transformer
	
	ACCEPTS:
		args - Argparse arguments
	
	RETURNS:
		Nothing
	"""
	
	# Create path variables
	project_path = os.path.dirname(os.path.abspath(__file__))
	os.makedirs(os.path.join(project_path, 'data'), exist_ok=True)
	md_path = os.path.join(project_path, 'data', 'metadata.json')
	model_path = os.path.join(project_path, 'model', 'model.pkl')
	cf_path = os.path.join(project_path, 'model', 'cf.pkl')
	metrics_path = os.path.join(project_path, 'model', 'metrics.json')
	
	# If skip_clean argument is not made in cli, clean the data
	if not args.skip_clean:
		# Download the training dataset from Kaggle
		if not args.t:
			print('Downloading training data...')
			path = kagglehub.dataset_download("ahmedshahriarsakib/usa-real-estate-dataset")
			data_path = os.path.join(path, 'realtor-data.zip.csv')
		
		# Add way to use different datasets to train model
		else:
			data_path = args.training_data
		
		# Used in process_training_data function
		clean = True
		
	else:
		print('Loading clean data')
		# If skip_clean, load clean_data dataset and set clean to false
		data_path = os.path.join(project_path, 'data', 'clean_data.csv')
		clean = False
	
	print(data_path)
	
	# Load data as pandas DataFrame
	data = pd.read_csv(data_path)
	
	# Split data, get encoder out
	X_train, y_train, X_test, y_test, cf = process_train_data(data, clean)
	
	# Save column transformer
	save_model(cf, cf_path)
	
	# Train model on X_train, y_train
	model = train_model(X_train, y_train)
	
	# Save model
	print('Saving model')
	save_model(model, model_path)
	print(f'Model saved to {model_path}')

	# Load model
	model = load_model(model_path)
	print(f'Model loaded from {model_path}')

	# Predict price on X_test
	print('Making predictions')
	preds = inference(model, X_test)

	# Compute model metrics
	print('Computing model metrics')
	mdae, mae, r2_score = compute_model_metrics(X_test, y_test, preds, model)
	
	# Save metrics and print to terminal
	metrics_dict = {'mae': mae, 'mdae': mdae, 'r2_score': r2_score}
	save_metrics(metrics_dict, metrics_path)
	print('\n' + f'Mean absolute error: {mae:.2f} | Median absolute error: {mdae:.2f} | R-Squared: {r2_score:.2f}\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Training model')
	
	parser.add_argument(
		'--skip_clean',
		help='Skips the initial cleaning step',
		action='store_true'
	)
	
	parser.add_argument(
		'-t',
		help='Tells the code to use --training_data argument.',
		required=False,
		action='store_true'
	)
	
	parser.add_argument(
		'--training_data',
		help='The training dataset to use if -t. If -t is not used, this argument is ignored.',
		type=str,
		required=False
	)
	
		
	args = parser.parse_args()
	create_model(args)
