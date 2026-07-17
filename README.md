# NOTICE
This project is currntly in beta as I wait for model optimization. As such, its predictions can be fairly inaccurate. Also, 
in current release, the project does not support running on Windows systems. In its current release, it is also not 
documented, as I often forget to document my code while writing it. In future releases, it will be.

---

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
you can run `pip install -r requirements.txt`. This install the necessary python libraries to the venv.


### Run the model

When running the model, you can use the "example.csv" file to see how the input/output works. I would recommend opening the 
"example_preds.csv" file at least once to see what columns can be used in your input csv file. To run the "example.csv" file 
through the model, in the command line (make sure the venv is active, if using), type (or copy-paste) `python3 main.py 
--input_file example.csv --use_example yes`. For more detailed information on each option, a later section titled "main.py 
Options" will explain. 

For now, `--input_file` is the filepath of the csv file to pass through the model, and `--use_example` tells the code to 
specifically use the example.csv file. This will be removed later, as it is redundant, but I didn't realize that at the 
time. In any case, to use your own file, replace example.csv with your desired csv file, which you can make via any 
spreadsheet software through Download or Export > csv. The easiest way to do this is to type `python main.py --input_file`, 
then click and drag your csv file from the file explorer to the terminal. 

Before pressing enter to run, you can choose to add the predictions to the original csv file or write a new csv file by 
adding `--new_output` followed by the desired path to the output file. This path *does* need to also have the filenam, 
so "path/to/file" will not work, but "path/to/file/file.csv" will, assuming that "path/to/file" actually exists. Later 
versions will create this path, but this one does not. Doing this will have the same effect as running the "example.csv" 
file through the model, leaving the original file untouched, instead creating a new one.
