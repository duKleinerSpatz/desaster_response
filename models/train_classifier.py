import sys
import pandas as pd
import time
from datetime import datetime
from IPython.core.display import display, HTML
from sqlalchemy import create_engine
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score

nltk.download(["wordnet", "punkt"])


def load_data(database_filepath):
    engine = create_engine(database_filepath)
    df = pd.read_sql_table("disaster_messages", database_filepath)
    X = df["message"].values
    Y = df.drop(labels=["id", "message", "original", "genre"], axis=1).values
    features = df.drop(labels=["id", "message", "original", "genre"], axis=1).columns

    return X, Y, features


def tokenize(text):
    """
    function:
        separates messages into root form of lower case word tokens without punctuation or stopwords
    args:
        text(str): message to be later classified
    returns:
        lemmed(list of str): list of root forms of lower case word tokens without punctuation or stopwords of the message
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]

    # lemmatization: Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]

    return lemmed


def build_model(X, Y):
    pipeline = Pipeline([
        ('tfidfvect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, features):
    """
    function: prints out statistics of the results after fitting and predicting a model
    args:
        Y_test(numpy.ndarray): test data from train_test_split
        Y_pred(numpy.ndarray): predicted data
        features(list of str): list of column names of the features to be predicted
    return:
        df_res(DataFrame): DataFrame containing the classification report data for each feature
    """

    # predict on test data
    Y_pred = model.predict(X_test)

    # subset_accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in Y_test.
    subset_accuracy = accuracy_score(Y_test, Y_pred)
    overall_accuracy = (Y_pred == Y_test).mean()
    print("subset_accuracy: {:.3f}\noverall_accuracy: {:.3f}".format(subset_accuracy, overall_accuracy))

    # create a results dataframe containing the classification reports for all columns (multiindexed)
    df_res = pd.DataFrame()
    i = 0
    for y_test, y_pred in zip(Y_test.transpose(), Y_pred.transpose()):
        df_temp = pd.DataFrame.from_dict(classification_report(y_test, y_pred, output_dict=True, zero_division=1))
        df_temp = pd.concat([df_temp], axis=1, keys=[features[i]])  # add column name as additional level
        df_res = pd.concat([df_res, df_temp], axis=1)
        i += 1

    # overall mean classification report values:
    display(df_res.transpose().mean())
    # display(df_res.transpose())


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, features = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, features)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()