import os

import pandas as pd
import numpy as np
import argparse

from ml.model import load_model, inference, load_metrics
from ml.data import process_user_data

def go(args):
    
    print('Model created by Joshua McClintic for Bachelor\'s degree capstone project.\n')
    print('The code used for this model cannot be used as training data for the purpose of creating generative AI. If, however, you would like to use the code within a script/program you are making (including scripts/programs to make AI/ML models) you are comletely free to do so!\n')
    print('Thank you for using this program!')
    
    print('\n\n')
    
    
    print('Loading training data (this data is used compare')
    project_path = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(project_path, 'model', 'metrics.json')
    model_path = os.path.join(project_path, 'model', 'model.pkl')
    
    
    user_df = args.input_file
    user_df = process_user_data(user_df)
    
    print('Loading model')
    model = load_model(model_path)
    
    print('Making predictions')
    preds = inference(model, user_df, test_set=False)
    user_df['Predictions'] = np.around(preds, decimals=2)
    
    metrics = load_metrics(metrics_path)
    mdae = round(metrics['mdae'], 2)
    user_df['Minimum estimate'] = preds - mdae
    user_df['Maximum estimate'] = preds + mdae
    
    print('Writing to file')
    if args.use_example.lower() == 'yes':
        user_df.to_csv(f'example_preds.csv', index=False)
    else:
        if args.new_output == 'no':
            user_df.to_csv(args.input_file, index=False)
        else:
            user_df.to_csv(args.new_output, index=False)
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Making predictions')
    
    parser.add_argument(
        '--input_file',
        help='The input csv file path to run model on',
        type=str,
        required=True
    )
    
    parser.add_argument(
        '--use_example',
        help='Set yes to use the example csv file or no to use your own csv file. Default yes',
        type=str,
        required=False,
        default='example_preds.csv'
    )
    
    parser.add_argument(
        '--new_output',
        help='Set to filepath of desired output. By default, this program adds to the original csv file with the predictions. If set to a filepath, the name of the output file must also be specified.',
        type=str,
        required=False,
        default='no'
    )
    
    args = parser.parse_args()
    go(args)
