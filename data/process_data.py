import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    args:
        messages_filepath(str): path to file containing messages
        categories_filepath(str): path to file containing categories
    returns:
        df(pd.DataFrame): DataFrame containing messages and matching categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how="outer", on="id")

    return df

def clean_data(df):
    """
    args:
        df(pd.DataFrame): dataframe containing messages and matching categories
    regurns:
        df(pd.DataFrame): cleaned dataframe containing messages and matching categories
    """
    categories = df.categories.str.split(pat=";", n=-1, expand=True)
    category_colnames = list(categories.iloc[0].str[:-2].values)
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
    categories.replace(2, 1, inplace=True)
    df.drop(labels=["categories"], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    args:
        df(pd.DataFrame): cleaned dataframe to be saved as an sql data base
        database_filename(str): filename of the saved sql data base
    """
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql('disaster_messages', engine, index=False)


def main():
    """
    loads from csv, cleans and saves data of disaster messages to an sqlite database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()