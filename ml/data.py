import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from pandas.api.types import is_numeric_dtype
import json


# JSON save and load functions to avoid circular importing
def save_metadata(md, path):
	"""
	Saves JSON file of metadata at path
	
	ACCEPTS:
		md - JSON serializable variable to save at path
		path - Filepath to save md at
		
	RETURNS:
		Nothing
	"""
	json.dump(md, open(path, 'w+'))
	
	
def load_metadata(path):
	"""
	Loads JSON file from path
	
	ACCEPTS:
		path - Filepath to load from
	
	RETURNS:
		md - JSON file found at path
	"""
	md = json.load(open(path, 'r'))
	
	return md


# Create variable for parent directory containing all project files
# The [:-2] at the end removes the last two items in the string. In this case,
# the filepath ends in '/ml', so the variable is set to the parent file of /ml.
# If the name of the 'ml' folder is changed to have a different number of
# charaters, this number also needs to be changed
project_path = (os.path.dirname(os.path.abspath(__file__)))[:-2]



def process_train_data(df, do_clean=True):
	"""
	Cleans and splits the dataset to train and test. Also saves metadata and fitted encoder to be used on user data
	
	ACCEPTS:
		df - pandas DataFrame to clean
		do_clean - Boolean; If False, skips the initial cleaning step
	
	RETURNS:
		X_train - Predictor training data
		y_train - Train prices to train model
		X_test - Predictor testing data
		y_test - Test prices to test model
		encoder - Fitted column transformer with OrdinalEncoder
	"""
	# If do_clean, clean data by dropping unused columns and removing nulls 
	# and outliers in price column
	if do_clean:
		print('Cleaning data...')
		# Ensure columns exist in df before dropping
		for col in ['prev_sold_date', 'brokered_by', 'street']:
			if col in df.columns:
				df = df.drop(columns=col)

		df = df.dropna(subset='price')
		df = df.query('price < 1000000')
		
		# Save clean dataset
		print('Saving clean dataset as csv')
		save_path = os.path.join(project_path, 'data', 'clean_data.csv')
		df.to_csv(save_path, index=False)
	
	
	# Create metadata JSON file; this is used to process user data
	metadata = {'columns': [], 'medians': {}, 'cat_columns': []}
	metadata['columns'] = list(df.columns)
	
	desc = df.describe()
	for col in df.columns:
		if is_numeric_dtype(df[col]):
			metadata['medians'][col] = desc[col]['50%']
		else:
			metadata['cat_columns'].append(col)
	
	# Save metadata in 'data' folder
	md_path = os.path.join(project_path, 'data', 'metadata.json')
	save_metadata(metadata, md_path)
	
	# Get categorical columns from metadata
	cat_cols = metadata['cat_columns']
	# Create ColumnTransformer object with OrdinalEncoder
	enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
	
	# Fit transformer to dataset
	enc = enc.fit(df[cat_cols])
	# Encode categorical columns
	df[cat_cols] = pd.DataFrame(
		enc.transform(df[cat_cols]), 
		columns=enc.get_feature_names_out()
	)
	
	print('Splitting data into train and test')
	# Split dataset to train and test
	X_train, X_test = train_test_split(df, random_state=42)
	y_train = X_train.pop('price')
	y_test = X_test.pop('price')
	
	
	
	
	return X_train, y_train, X_test, y_test, enc
	
	
def process_user_data(user_df, enc, md):
	"""
	Ensure columns in user data match what the model expects and encodes categorical columns
	
	ACCEPTS:
		user_df - pandas DataFrame for users to pass through model
		enc - OrdinalEncoder to encode categorical columns in user_file
		md - Metadata of training data
	
	RETURNS:
		user_df - Processed user dataset
	"""
	
	# Get column names, categorical columns, and medians from md
	columns = md['columns']
	cat_columns = md['cat_columns']
	medians = md['medians']
	
	columns.remove('price')
	del medians['price']
	
	# Iterate over columns
	for col in columns:
		# If a column name does not exist in user_df
		if col not in user_df.columns:
			# If the column is categorical, set column value to None
			if col in cat_columns or col == 'zip_code':
				user_df[col] = None
			# If numerical, use training dataset's median value
			else:
				user_df[col] = medians[col]
	for col in user_df.columns:
		if col == 'price':
			raise ValueError('"Price" column should not be your dataset.')
		if col not in columns:
			raise ValueError(f'Column {col} does not exist in training data.')
	
	# Encode user_df categorical columns
	user_df = user_df.reindex(columns=columns)
	
	user_df[cat_columns] = pd.DataFrame(
		enc.transform(user_df[cat_columns]),
		columns=enc.get_feature_names_out()
	)
				
	return user_df
