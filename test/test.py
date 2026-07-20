import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.append('..')
from ml.data import load_metadata, process_user_data
from ml.model import load_model


@pytest.fixture(scope="session")
def fix():
	project_path = os.path.dirname(os.path.abspath(__file__))[:-4]
	
	md = load_metadata(os.path.join(project_path, 'data', 'metadata.json'))
	df = pd.read_csv(os.path.join(project_path, 'example.csv'))
	enc = load_model(os.path.join(project_path, 'model', 'enc.pkl'))
	model = load_model(os.path.join(project_path, 'model', 'model.pkl'))
	
	return {'md': md, 'df': df, 'enc': enc, 'model': model}


def test_pytest(fix):
	metadata = fix['md']
	assert 'price' in metadata['columns']
	

def test_process_user_data(fix):
	df = fix['df']
	md = fix['md']
	enc = fix['enc']
	
	df = process_user_data(df, enc, md)
	assert df.shape[1] == 8
	
	df = fix['df']
	df['price'] = 0
	with pytest.raises(ValueError):
		df = process_user_data(df, enc, md)
		
	df = fix['df']
	df['test_col'] = 'test'
	with pytest.raises(ValueError):
		df = process_user_data(df, enc, md)
