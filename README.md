# NOTICE

This project is currently in beta while I wait for a fully optimized model. Its current median absolute error is just over 
50,000 and its mean absolute error is just over 80,000 with an $$R^2$$ score of .68. For more information, check out the 
model_card.md file.

---


## Data

### Source

This dataset is publicly available on kaggle, and can be found at https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset.


### Dataset

The training data for this project consists of 2 million rows of real estate data. The columns are as follows:

* `brokered_by` (categorically encoded agency/broker)
* `status` (Housing status - a. ready for sale or b. ready to build)
* `price` (Housing price, it is either the current listing price or recently sold price if the house is sold recently)
* `bed` (# of beds)
* `bath` (# of bathrooms)
* `acre_lot` (Property / Land size in acres)
* `street` (categorically encoded street address)
* `city` (city name)
* `state` (state name)
* `zip_code` (postal code of the area)
* `house_size` (house area/size/living space in square feet)
* `prev_sold_date` (Previously sold date)

The columns `prev_sold_date`, `brokered_by`, and `street` are removed before training. `prev_sold_date` has very little 
correlation with the price of a house, and is unlikely to be a known variable when users run the model. `brokered_by` and 
`street` are both encoded, and there is realistically no way for me to encode a user's dataset and encode these variables 
in the same way, so they cannot be passed into the model as input, and are thus taken out of the training data.


## Running the model

### First steps

To run the model, first download the repository (ideally through releases page). Click on the latest release, then the 
"Source code (zip)" to download. Extract the files to your desired location. Next, navigate to the extracted folder, which 
will have the name of the repository followed by the version number, just as the zip file. Open the folder, right click in 
the folder, then click "Open in terminal". 


### Build virtual env

This step isn't strictly necessary, but it is recommended. While in the terminal, type (or copy-paste, note that ctrl+v 
to paste doesn't work in the terminal, so you'll have to right-click to paste) the following code, one by one: 
* `python -m venv model_venv`
* `source model_venv/bin/activate`

Once the virtual environment is active (you can tell it's active if, on the start of the terminal line, you see `(model_venv)`), 
you can run `pip install -r requirements.txt`. This install the necessary python libraries to the venv. The libraries in this 
file are pandas, scikit-learn, and kagglehub.


### Run the model

When running the model, you can use the "example.csv" file to see how the input/output works. I would recommend opening the 
"example_preds.csv" file at least once to see what columns can be used in your input csv file. If a column does not exist 
in the "example_preds.csv" file, it should not exist in your csv file. Also, I should mention that your csv file *needs* to 
have column names at the top, like how it is in the "example.csv" file. For more detailed information on how to format the 
csv file, check the "Input" section.

To run the model on the "example.csv" file, type or copy-paste `python3 main.py -n --output_file example_preds.csv` into the 
terminal. More information on the arguments will be given later, but for now, `-n` tells the code to not overwrite the original 
csv file, and `--output_file [file_path]` tells the code where to write the new output file. This `[file_path]` should contain 
path and name of the desired output file, like `[path/filename.csv]`. Here, a path is not specified, but a name is, so the new 
output is written directly into the folder "main.py" is located. It is important to note that, while the filname does not need 
to exist, the path or folder you are writing the file to does need to exist.

By default, the code runs the "example.csv" file through the model. To run a different file through, type `python3 main.py -u 
--input_file [path/filename.csv]` where `[path/filename.csv]` is the location of your csv file. Just like in the above example, 
you can add an output location by typing `python3 main.py -u --input_file[path/filename.csv] -n --output_file [path/output.csv]`. 
For context, if an output filepath is not specified, the original file will be overwritten. This can save on space if you are 
running a very large csv file through the model.


### Input

The user input can, but does not necessarily have to, contain the following columns:

* `status` (can be one of the following values: "for_sale", "ready_to_build", "sold") - Categorical
* `bed` (# of beds) - Numerical
* `bath` (# of baths) - Numerical
* `acre_lot` (Property / Land size in acres)  - Numerical
* `city` (city name) - Categorical
* `state` (state name) - Categorical
* `zip_code` (postal code of the area) - Categorical
* `house_size` (house area/size/living space in square feet) - Numerical

The input csv file must have column headers, and those headers, if used, must match spelling and capitalization of these 
column names. However, you do not have to add every column. For example, you can leave out the `acre_lot` column entirely. 
In such cases, one of the two will happen:

* 1: In the case of numerical columns, the median value of that column from the training dataset will be used. These medians can be found in the "data/metadata.json" file. 
* 2: In the case of categorical columns, the value will be null, or empty

Note that "zip_code" is not listed in the categorical columns and contains a median value in the metadata. This is fine. 
The code interperets this column as numeric and calculates its median. When "zip_code" is empty in the user csv file, it 
gets treated as categorical. Keeping it numeric when training increases model performance, as setting it as categorical 
in the metadata encodes the "zip_code" column when training. This is unnecessary.

There are two final things to mention. The first is that in your input file, the order of the columns does not matter. The 
second is that the more columns you use, the more accurate the model will be. This fact is not represented in the output, but 
you should know that more data input means more reliable output.


### Output

Within the output file, any column that existed in the input file will exist here as well, though maybe in a different order. 
Along with those columns, any column that was not included will be added to the output using the above ruling. This needs to 
be done to ensure the model can actually predict the price, as it needs the same columns as the training data. You do not need 
to add these columns yourself as the script will do that for you.

The final three columns in the output are "Predictions", "Minimum estimate", and "Maximum estimate". The prediction is what the 
model predicts a hose with those features will be. The minimum and maximum estimates take the prediction and add or subtract 
the median absolute error, which can be found in "model/metrics.json".

As stated earlier, if no output is specified, the original csv file will be overwritten.


## Code

### main.py

This is the main script to run csv files through a trained model.

**Command Line Arguments**

* -u: 
* * Flag, boolean
* * If -u is written in the command line, --input_file must also be specified. If not, "example.csv" is run through the model

* --input_file [path/to/file/filename.csv]:
* * Filepath, string
* * Path to the input file, including file name. Rather than type out the full filepath, you can click and drag a file to the command line

* -n:
* * Flag, boolean
* * If -n is written in the command line, --output_file must also be specified. If not, the input_file will be overwritten

* --output_file [existing/path/filename.csv]:
* * Filepath, string
* * Path to write the output to, including file name. If just a filename is given without a path, the output will be written to the folder main.py is located. Otherwise, the given path must already exist.


### train_model.py

This script trains the ml model from scratch. This is an unnecessary step in most situations, and should only be used if 
something goes wrong, such as a corrupted or missing file.

**Command Line Arguments**

* --skip_clean:
* * Flag, boolean
* * If written to the command line, skips a basic preprocessing step. By using this argument, -t and --training_data are ignored. If "clean_data.csv" does not exist in the "data/" folder, calling this argument will cause an error.

* -t:
* * Flag, boolean
* * If used, --training_data must be specified. Otherwise, the Kaggle dataset will be used.

* --training_data [path/to/file/filename.csv]
* * Filepath, string
* * If -t is not used, this argument will be ignored. Otherwise, path to the training dataset (as a csv file) to be used to train the model.


