import os

import pandas as pd

from ml.data import process_train_data
from ml.model import train_model, compute_model_metrics, inference, save_model, load_model, save_metrics
import argparse

import kagglehub


def create_model(args):
	
	project_path = os.path.dirname(os.path.abspath(__file__))
	os.makedirs(os.path.dirname('data/'), exist_ok=True)
	md_path = os.path.join(project_path, 'data', 'metadata.json')
	model_path = os.path.join(project_path, 'model', 'model.pkl')
	metrics_path = os.path.join(project_path, 'model', 'metrics.json')
	
		
	if args.do_clean == 1:
		if args.download_data == 'yes':
			print('Downloading training data...')
			path = kagglehub.dataset_download("ahmedshahriarsakib/usa-real-estate-dataset")
			data_path = f'{path}/realtor-data.zip.csv'
		else:
			data_path = args.training_data
		
		clean = True
	else:
		print('Loading clean data')
		data_path = os.path.join(project_path, 'data', 'clean_data.csv')
		clean = False
	
	print(data_path)
	
	data = pd.read_csv(data_path)
	
	X_train, y_train, X_test, y_test, _ = process_train_data(data, clean)

	model = train_model(X_train, y_train)
	
	save_model(model, model_path)
	print(f'Model saved to {model_path}')

	model = load_model(model_path)
	print(f'Model loaded from {model_path}')

	preds = inference(model, X_test)

	mdae, r2_score = compute_model_metrics(y_test, preds, model, X_test)
	
	
	metrics_dict = {'mdae': mdae, 'r2_score': r2_score}
	save_metrics(metrics_dict, metrics_path)
	

	print('\n' + f'Median absolute error: {mdae:.2f} | R-Squared: {r2_score:.2f}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Training model')
	
	parser.add_argument(
		'--do_clean',
		help='0 to skip cleaning, 1 to perform cleaning. If data/clean_data does not exist, do not change this.',
		type=int,
		default=1
	)
	
	parser.add_argument(
		'--download_data',
		type=str,
		required=False,
		default='yes'
	)
	
	parser.add_argument(
		'--training_data',
		type=str,
		required=False
	)
	
		
	args = parser.parse_args()
	create_model(args)
