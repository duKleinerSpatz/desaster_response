# Disaster Response Pipeline Project

A repository for the second project of the Udacity Nanodegree Programm "Data Scientist", in which a training set of disaster response messages will be transformed through data pipelines and classified for further uses.

### Description:
1. The two csv-files containing the disaster messages data were loaded, merged, cleaned and saved to an sqlite database. (see "process_data.py" or "ETL_Pipeline_Preparation.ipynb")

2. I used the data from the sqlite database to build a machine learning model through a pipeline that tokenizes, vectorizes and classifies the messages. ("train_classifier.py")
    - In order to find the "best" classifier, i iterated through MLPClassifier, KNeighborsClassifier, GaussianNB, RandomForestClassifier, AdaBoostClassifier
    - since the computing took a lot of time, i only used a 10% fraction of the database
    - it appeared that the KNeighborsClassifier was the most efficient one
    - after tuning the model with gridsearch, I saved it as a pickle file (see "classifier.pkl")

3. Finally i embedded the database, classifier as well as two further visualizations in the given web application ("run.py")

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
