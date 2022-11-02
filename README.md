# big_data_project1

## Running Code

To run code, make sure to install the neccessary libraries (Using Anaconda may be best). 

Next, The first file to run is process.py. this file collects the training data and parses it into two .csv files which will be used for the next file that will be run.

The next file to run is train_classify.py. This file will parse the training data and train a linear SVC model for classifications. There are two ways to run the file:

1. The first option is to leave it as is and it will train on a training and testing batch based on just the training data; this will output two files: Classification_Report.txt and Conf_Matrix.png.

2. The second option is to uncomment the code specified in train_classify.py to run on the real testing data. This will produce a csv file called pred.csv. This is what is used to make submissions on Kaggle.
