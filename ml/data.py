import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype
import json


 
def save_metadata(md, path):
	json.dump(md, open(path, 'w+'))
	
def load_metadata(path):
	md = json.load(open(path, 'r'))
	
	return md


project_path = os.path.dirname(os.path.abspath(__file__))
md_path = os.path.join(project_path[:-2], 'data', 'metadata.json')


def process_train_data(data, do_clean=True):
	
	df = data
	
	if do_clean:
		print('Cleaning data... (this step may take a while)')
		df[['year', 'month', 'day']] = df['prev_sold_date'].str.split(pat='-').apply(pd.Series)
		df = df.drop(columns=['prev_sold_date', 'city'])

		df = df.dropna(subset='price')
		
		print('Saving clean dataset as csv')
		df.to_csv('./data/clean_data.csv', index=False)
	
	metadata = {'columns': [], 'medians': {}, 'cat_columns': []}
	metadata['columns'] = list(df.columns)
	
	desc = df.describe()
	for col in df.columns:
		if is_numeric_dtype(df[col]):
			metadata['medians'][col] = desc[col]['50%']
		else:
			metadata['cat_columns'].append(col)
	
	save_metadata(metadata, md_path)
	
	
	print('Splitting data into train and test')
	X_train, X_test = train_test_split(df, random_state=42)
	y_train = X_train.pop('price')
	y_test = X_test.pop('price')
	
	return X_train, y_train, X_test, y_test, df
	
	
def process_user_data(user_file):
	
	user_df = pd.read_csv(user_file)
	
	md = load_metadata(md_path)
	columns = md['columns']
	cat_columns = md['cat_columns']
	medians = md['medians']
	
	for col in columns:
		if col not in user_df.columns:
			if col in cat_columns:
				user_df[col] = None
			elif col == 'status':
				user_df['status'] = 'for_sale'
			else:
				user_df[col] = medians[col]
				
	return user_df
