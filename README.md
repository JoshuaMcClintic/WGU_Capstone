# NOTICE

This project exists as the capstone project for my BSDA degree for WGU. The goal of the project was to create a data analytics 
problem to solve, then solve it. Included in this GitHub repository are as follows:

* Python scripts to train a machine learning model to predict house prices
* A python script to run csv files through the model to predict the prices of houses based on the features included in the files
* EDA jupyter notebook in ipynb and html format which chronicles the process of exploring the data and optimizing the model
* Error visualization pdf which shows the model's performance using charts made in Tableau
* The trained model
* Extra files for related items

This project should not be used to *determine* the price of a house. Its intended use is to compare sets of features and 
locations to streamline the process of searching for a house.


## Python

This project requires python 3.10 or higher, as it uses a scikit-learn version of 1.7.2, which is unavailable for python 3.9. 
This is not a hard requirement, and the code will still probably work, but I cannot ensure that it will work as intended.


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

* `python3 -m venv model_venv`

* **For Linux/MacOS systems**: `source model_venv/bin/activate`
* **For Windows systems**: `./model_venv/Scripts/activate`


Once the virtual environment is active (you can tell it's active if, on the start of the terminal line, you see `(model_venv)`), 
you can run `pip install -r requirements.txt`. This install the necessary python libraries to the venv. The libraries in this 
file are pandas, scikit-learn, and kagglehub.


### Run the model

When running the model, you can use the "example.csv" file to see how the input/output works. I would recommend opening the 
"example_preds.csv" file at least once to see what columns can be used in your input csv file. If a column does not exist 
in the "example_preds.csv" file, it should not exist in your csv file. Also, I should mention that your csv file *needs* to 
have column names at the top, like how it is in the "example.csv" file. For more detailed information on how to format the 
csv file, check the "Input" section. Both the "example.csv" and the "example_preds.csv" files can be found in the "docs" 
folder.

To run the model on the "example.csv" file, type or copy-paste `python3 main.py -e` into the 
terminal. More information on the arguments will be given later, but for now, `-e` just tells the script to use the 
"example.csv" file. When running the script on other files, don't include the `-e`. For more instructions, check the "main.py" 
sub-section in the "Code" section of this readme. This will write the predictions to a new csv file called "example_preds.csv". 
You can delete the existing "example_preds.csv" to see this. When running your own files, you can choose to overwrite the 
original file or create a new file.

To run the model on your own csv file (the code does not currently accept anything but csv files, which can be created in any 
spreadsheet program), type `python3 main.py --input_file "[path/to/file/filename.csv]"`, using your own filepath/filename.csv, 
also without the square brackets. If you want to create a new csv file with the outputs, add `--output_file 
"[existing/path/new_filename.csv]"` to the end. As the command implies, the path to the file you are writing to *must* exist. 
However, the file itself does not have to exist. You can expirment with the "example.csv" file by specifying it as the input 
file, rather than using `-e`.


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

When specifying the input file, there are two main ways to write the file to the terminal. The first is the manual way, by 
actually writing the path to the file. The second is far easier, as you can simply click and drag the file into the terminal 
window, which will automatically write the entire absolute path of that file to the terminal. 

You can also add the file to the folder containing the "main.py" file, then for the `--input_file` argument, simply write out 
the filename, like `python3 main.py --input_file example.csv --output_file example_output.csv`. This will also write the output 
file to the same folder. That is, by not specifying a path and only specifying the filename, the code will attempt to read from 
or write to a file in the same folder as the "main.py" file.

The input csv file must have column headers, and those headers, if used, must match spelling and capitalization of these 
column names. However, you do not have to add every column. For example, you can leave out the `acre_lot` column entirely. 
In such cases, one of the two will happen:

* 1: In the case of numerical columns, the median value of that column from the training dataset will be used. These medians can be found in the "data/metadata.json" file. 
* 2: In the case of categorical columns, the value will be null, or empty

Note that "zip_code" is not listed in the categorical columns and contains a median value in the metadata. This is fine. 
The code interperets this column as numeric and calculates its median. When "zip_code" is empty in the user csv file, it 
gets treated as categorical. Keeping it numeric when training increases model performance, as setting it as categorical 
in the metadata encodes the "zip_code" column when training. This is unnecessary.

There are two final things to mention. The first is that, in your input file, the order of the columns does not matter. The 
second is that the more columns you use, the more accurate the model will be. This fact is not represented in the output, but 
you should know that more data input means more reliable output, but don't feel pressured to using every column if you, for 
example, don't want to compare between specific vales of "acre_lot".


### Output

Within the output file, any column that existed in the input file will exist here as well, though maybe in a different order. 
Along with those columns, any column that was not included will be added to the output using the above ruling. This needs to 
be done to ensure the model can actually predict the price, as it needs the same columns as the training data. You do not need 
to add these columns yourself as the script will do that for you.

The final three columns in the output are "Predictions", "Minimum_estimate", and "Maximum_estimate". The prediction is what the 
model predicts a hose with those features will be. The minimum and maximum estimates take the prediction and add or subtract 
the median absolute error, which can be found in "model/metrics.json".

As stated earlier, if no output is specified, the original csv file will be overwritten. If you plan on running the file 
through the model multiple times, you should specify a new output.


## Code

### main.py

This is the main script to run csv files through a trained model.

**Command Line Arguments**

* -e: 
  * Flag, boolean
  * If -e is written, the "example.csv" file will be passed through the model. If used, the other arguments will be ignored.

* --input_file [path/to/file/filename.csv]:
  * Filepath, string
  * Path to the input file, including file name. Rather than type out the full filepath, you can click and drag a file to the command line

* --output_file [existing/path/filename.csv]:
  * Filepath, string
  * Path to write the output to, including file name. If just a filename is given without a path, the output will be written to the folder main.py is located. Otherwise, the given path must already exist.

* --budget [int]
  * Integer number
  * Maximum price to include in the output. If this value is specified, after the model is run, any row where the prediction value is above this number will be removed and not included in the final output. It should be a whole number, so no decimals.


Note that the `--budget` argument does not get used on the min or max estimates, only the prediction. As such, if used, some 
values in the "Maximum_estimate" column may be above this budget. If you don't want this, subtract the median absolute error 
found above or in the model_card.md file from your budget, and use that value.


### train_model.py

This script trains the ml model from scratch. This is an unnecessary step in most situations, and should only be used if 
something goes wrong, such as a corrupted or missing file.

**Command Line Arguments**

* --skip_clean:
  * Flag, boolean
  * If written to the command line, skips a basic preprocessing step. By using this argument, --training_data is ignored. If "clean_data.csv" does not exist in the "data/" folder, calling this argument will cause an error.

* --training_data [path/to/file/filename.csv]
  * Filepath, string
  * Path to the training data. If "--skip_clean" is used, this argument will be ignored. This argument should only be used if you know what you are doing, and is only here in case the original data file is updated and someone wants to update the model.


## Error Visualizations.

There is a file in the "docs/" folder called "Visualizations of Model Error.pdf", which contains links to a Tableau page 
(https://public.tableau.com/app/profile/joshua.mcclintic/viz/Wgu_project2/Dashboard1). This document explains the 
visualizations in the Tableau page more thoroughly, but in general, there are 5 visulizations. The first shows the median 
predicted price and the median actual price of homes in each state. For the rest, there is a black line showing the median 
actual price of a house per each numeric value in the dataset. Along with this line, there is a light blue inner bar, which 
shows the median predicted price per X value +/- the median absolute error. The red outer bars show the same price +/- the 
mean absolute error.

What we see is that generally, the model does well to predict the price of homes when these values are relatively small, but 
when they start to get quite large, the model begins to struggle. Note that the values at which the model significantly loses 
performance tend to be at points where model performance likely does not matter anymore. For example, the model tends to 
perform very well until around 5 baths, and still performs decently well until around 9 baths. While these numbers are not 
absurdly high, at least for a single family home, 5 baths feels somewhat excessive. A similar trend can be seen with the 
others.

One final thing about the graphs is that the actual data continues beyond what the graphs show. The cutoff points you see 
were determined by me, and are intended to either show how the model's performance differs between small and large values, 
or increasing the range makes it harder to read the plots. 


