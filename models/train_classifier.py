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
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score

nltk.download(["wordnet", "punkt"])


def load_data(database_filepath):
    """
    args:
        database_filepath(str): filepath to database
    returns:
        X(numpy.ndarray): array containing the messages
        Y(numpy.ndarray): array containing the suitable disaster categories for each message
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("disaster_messages", engine)  # .sample(frac=0.1)
    X = df["message"].values
    Y = df.drop(labels=["id", "message", "original", "genre"], axis=1).values
    features = df.drop(labels=["id", "message", "original", "genre"], axis=1).columns

    return X, Y, features


def tokenize(text):
    """
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


def build_model():
    """
    args:
        -
    returns:
        cv(model): with gridsearch optimized model
    """
    pipeline = Pipeline([
        ('tfidfvect', TfidfVectorizer(tokenizer=tokenize)),
        #('to_dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    params_pick = {
        'tfidfvect__use_idf': (True, False),
        'clf__estimator__n_neighbors': [5, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=params_pick)

    return cv


def evaluate_model(model, X_test, Y_test, categories):
    """
    args:
        Y_test(numpy.ndarray): test data from train_test_split
        Y_pred(numpy.ndarray): predicted data
        categories(list of str): list of column names of the categories to be predicted
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
        df_temp = pd.concat([df_temp], axis=1, keys=[categories[i]])  # add column name as additional level
        df_res = pd.concat([df_res, df_temp], axis=1)
        i += 1

    # overall mean classification report values:
    display(df_res.transpose().mean())
    display(df_res.transpose())

    # display best params
    print("best parameters found by gridsearch:\n", model.best_params_)

def save_model(model, model_filepath):
    """
    args:
        model(model): model to be saved
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """
    load data from sqlite database, build a machine learning model to categorize disaster messages,
    evaluate and save model as pickle file.
    """
    if len(sys.argv) == 3:
        print("Starting train_classifier.py on {}".format(datetime.now()))
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, features = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        # calculate time to fit
        start_time = time.time()
        print('Training model...')
        model.fit(X_train, Y_train)
        print("--- {:.0f}s seconds to fit model ---\n".format((time.time() - start_time)))

        # calculate time to evaluate
        start_time = time.time()
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, features)
        print("--- {:.0f}s seconds to evaluate model ---\n".format((time.time() - start_time)))

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
