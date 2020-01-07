
Open your terminal and change directory to your folder

then write :
    Doker-compose build 
    (this step take a long time)
    Docker-compose up -d
then open your browser and go to http://127.0.0.1:8888/tree this is your new local environement, ready to work on Machine Learning, this docker file was made by Aurelien Géron.

if you have your own environement, for run this IA you need to import module Xgboost, pandas, joblib for Python (pip install). 

IA production : 

    For reproduce the Kaggle score donwload the file test cs-test.csv and place it on the root of folder :
    https://www.kaggle.com/c/GiveMeSomeCredit/data
   
    Set your source file name and Execute NoteBook Production IA.
    When it's done, you will have a new file forkaggFinal.csv (name by default) on the root of folder, ready to send for kaggle if you want your Score will be = 0.86221.

    Or change data to make prediction you need

    the columns probability of csv file give the probability for prediction = 1.
    
Travaux préparatoire : 

    This all Notebooks and scripts use for build the Ia. Reproduce them was not recommanded, Bigml was use for some of predicions analyse,
    so you must have your logs & api keys. those notebook's are just for information and observation. The number 5 is the notebook where model was built.

 