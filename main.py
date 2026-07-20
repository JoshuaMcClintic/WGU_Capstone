import os

import pandas as pd
import numpy as np
import argparse

from ml.model import load_model, inference, load_metrics
from ml.data import process_user_data, load_metadata

def go(args):
    # Print welcome message
    print('Model created by Joshua McClintic for Bachelor\'s degree capstone project.\n')
    print('The code used for this model cannot be used as training data for the purpose of creating generative AI. If, however, you would like to use the code within a script/program you are making (including scripts/programs to make AI/ML models) you are comletely free to do so!\n')
    print('Thank you for using this program!')
    
    print('\n\n')
    
    # Create path variables
    project_path = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(project_path, 'model', 'metrics.json')
    model_path = os.path.join(project_path, 'model', 'model.pkl')
    enc_path = os.path.join(project_path, 'model', 'enc.pkl')
    md_path = os.path.join(project_path, 'data', 'metadata.json')
    
    # Load model, column transformer, and metadata
    print('Loading model')
    model = load_model(model_path)
    enc = load_model(enc_path)
    md = load_metadata(md_path)
    
    # Process user data from input_file CL argument
    if args.u:
        user_df = args.input_file
        user_df = pd.read_csv(user_df)
        user_df = process_user_data(user_df, enc, md)
    else:
        user_df = 'example.csv'
        user_df = pd.read_csv(user_df)
        user_df = process_user_data(user_df, enc, md)
    
    
    
    # Predict user_df houes prices and write predictions to user_df
    print('Making predictions')
    preds = inference(model, user_df, test_set=False)
    user_df['Predictions'] = np.around(preds, decimals=2)
    
    # Load metrics and add prediction range to user_df
    metrics = load_metrics(metrics_path)
    mdae = round(metrics['mdae'], 2)
    user_df['Minimum estimate'] = np.around(preds - mdae, decimals=2)
    user_df['Maximum estimate'] = np.around(preds + mdae, decimals=2)
    
    # Inverse transform categorical columns
    cat_cols = md['cat_columns']
    user_df[cat_cols] = pd.DataFrame(
        enc.inverse_transform(user_df[cat_cols]),
        columns=cat_cols
    )
    
    # If if -n is used, write output to new output_file, else add to input_file
    print('Writing to file')
    if args.n:
        user_df.to_csv(args.output_file, index=False)
    else:
        user_df.to_csv(args.input_file)
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Making predictions')
    
    parser.add_argument(
        '-u',
        help='Use user file rather than example file. If not set, example.csv is automatically used.',
        action='store_true'
    )
    
    parser.add_argument(
        '--input_file',
        help='The input csv file path to run model on',
        type=str,
        required=False
    )
    
    parser.add_argument(
        '-n',
        help='If set, uses --output as file output. If not, overwrites the input file by adding predicion columns.',
        action='store_true'
    )
    
    parser.add_argument(
        '--output_file',
        help='If -n is used, the output path of the new file, including filename.',
        type=str
    )
    
    args = parser.parse_args()
    go(args)
