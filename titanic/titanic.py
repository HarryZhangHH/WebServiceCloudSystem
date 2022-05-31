import warnings
warnings.filterwarnings('ignore')

import re
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from random import randint

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# from sklearn.preprocessing import PolynomialFeatures

# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score

# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Playground')
parser.add_argument('--model', default='knn', type=str,
                    help='select a model from knn, mlp, sgd, svc')
parser.add_argument('--max_iter', default=1500, type=int,
                    help='max_iter for mlp')
# parser.add_argument('--batch_size', default=32, type=int,
#                     help='per-agent batch size')
# parser.add_argument('--lr', default=0.1, type=float,
#                     help='reference learning rate (for 256 sample batch-size)')
# parser.add_argument('--num_dataloader_workers', default=10, type=int,
#                     help='number of dataloader workers to fork from main')
# parser.add_argument('--num_epochs', default=90, type=int,
#                     help='number of epochs to train')

def get_social_staus(train, test):
    names = train['Name']
    social_status = []
    family_name = []
    for name in names:
        f_name =  name.split(',')[0]
        splitted = name.split(',')[1]
        splitted = splitted.split('.')[0]
        #print(splitted)
        social_status.append(splitted)
        family_name.append(f_name)
    files = pd.Series(social_status)
    #print(Counter(social_status))
    train['Title'] = files
    files = pd.Series(family_name)
    #print(Counter(family_name))
    train['FamilyName'] = files
    titles = []
    for title in test:
        test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)
    for item in test['Title']:
        if item not in titles:
            titles.append(item)
    for item in train['Title']:
        if item not in titles:
            titles.append(item)
    encoded_titles = []
    for i, item in enumerate(titles):
        encoded_titles.append(i)

    def titles_encoder(title):
        titles = [' Mr', ' Mrs', ' Miss', ' Master', ' Ms', ' Col', ' Rev', ' Dr', ' Dona', ' Don', ' Mme', ' Major', ' Lady', ' Sir', ' Mlle', ' Capt', ' Countess', ' Jonkheer']
        encoded_titles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        title = str(title)
        if title in titles:
            text_index = titles.index(title)
            encoded = encoded_titles[text_index]
            return int(encoded)
    train['TitleEncoded'] = train['Title'].apply(titles_encoder)
    return train
    
def get_familysize(train):
    train['Parch'].replace([0], [1], inplace = True)
    familysize = train['SibSp'] + train['Parch']
    train['FamilySize'] = familysize
    return train

def get_embark_encoded(train):
    def embark_encoded(code):
        if str(code) == 'S':
            return 0
        elif str(code) == 'Q':
            return 1
        elif str(code) == 'C':
            return 2
    train.Embarked.fillna('S', inplace=True)
    train['EmbarkedNum'] = train['Embarked'].apply(embark_encoded)
    return train

def clean_ticket_text(train):
    def get_ticket_number(text):
        text = text.split(' ')
        text = text[-1]
        if str(text) == 'LINE':
            text = 0
        return int(text)
    def get_ticket_header(text):
        text = text.split(' ')
        text = text[0]
        return text
    train['TicketNumber'] = train['Ticket'].apply(get_ticket_number)
    train['TicketHeader'] = train['Ticket'].apply(get_ticket_header)
    return train

def clean_cabin_number(train):
    def letter_cleaner(text):
        text = str(text)
        text = text.split()[-1]
        text = re.sub(r'[0-9]', '', text)
        return text
    def get_cabin_number(text):
        text = str(text)
        splitted = text.split(' ')
        if text == 'nan':
            text = 0
        else:
            text = splitted[0]
            text = text[1:]
            if str(text).isdigit()==False:
                text = splitted[-1]
                text = text[1:]
                if str(text).isdigit()==False:
                    text = 0
        return int(text)
    train = train.drop(['Name', 'PassengerId',], axis = 1)
    train['Sex'].replace(['male', 'female'], [0, 1], inplace = True)
    # cleaning the CabinNumber
    train['CabinLetter'] = train['Cabin'].apply(letter_cleaner)
    train['CabinNumber'] = train['Cabin'].apply(get_cabin_number)
    # Normalization the Fare
    train['Fare'] = train['Fare'] / train['Fare'].max()
    return train

def get_bridge(train):
    def cabin_filler(value):
        if value > 0.06 and value < 0.07:
            text = 'A'
        elif value > 0.19:
            text = 'B'
        elif value > 0.11 and value < 0.19:
            text = 'C'
        elif value > 0.08 and value < 0.11:
            text = 'D'
        elif value > 0.06 and value < 0.08:
            text = 'E'
        elif value > 0.04 and value < 0.06:
            text = 'F'
        elif value < 0.04:
            text = 'G'
        elif value > 0.04 and value < 0.07:
            text = 'T'
        else:
            text = 'G'
            print(f'Fare is {value}')
        return text
    def passenger_by_bridge(bridge):
        if bridge == 'A':
            return 42
        elif bridge == 'B':
            return 123
        elif bridge == 'C':
            return 310
        elif bridge == 'D':
            return 285
        elif bridge == 'E':
            return 583
        elif bridge == 'F':
            return 684
        elif bridge == 'G':
            return 362
        else:
            print(bridge)
    def passenger_by_class(passenger_class):
        if passenger_class == 3:
            return 1026
        elif passenger_class == 2:
            return 674
        elif passenger_class == 1:
            return 689
        else:
            print(passenger_class)
    class_letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
    cabin_class = []
    class_meanings = []
    for i in class_letter:
        cabin = train[(train['CabinLetter']== i)]
        #print(f'Class {i}:\t\t{cabin["Fare"].mean()}\t:meanFare')
        cabin_class.append(i)
        class_meanings.append(cabin["Fare"].mean())
    print(train.count())
    train['Bridge'] = train['Fare'].apply(cabin_filler)
    train['PassengerByBridge'] = train['Bridge'].apply(passenger_by_bridge)
    train['PassengerByClass'] = train['Pclass'].apply(passenger_by_class)
    staff_index = []
    for i, item in enumerate(train['Title']):
        if item in [' Major',' Col',' Capt',]:
            #print(item)
            train.at[i,'Bridge']= 'T'
            staff_index.append(i)
    nb_staff = len(staff_index)+2# + 2 Col in test set
    for i, item in enumerate(train['Title']):
        if item in [' Major',' Col',' Capt',]:
            train.at[i,'PassengerByBridge']= nb_staff
            train.at[i,'PassengerByClass']= nb_staff
    return train

def fillna_age(train):
    training_data = pd.read_csv('train.csv')
    child_age = randint(6, 12)
    for i, item in enumerate(train['Title']):
        if item in [' Miss', ' Mrs', ' Ms',]:
            if item == ' Miss':
                age = randint(16, 23)
                if str(train.at[i,'Age'])=='nan':
                    if str(training_data.at[i,'Parch'])!= 0:
                        train.at[i,'Age']= age
                    else:
                        train.at[i,'Age']= child_age
            elif item == ' Ms':
                age = randint(18, 26)
                if str(train.at[i,'Age'])=='nan':
                    if str(training_data.at[i,'Parch'])!= 0:
                        train.at[i,'Age']= age
                    else:
                        train.at[i,'Age']= child_age
            elif item == ' Mrs':
                age = randint(24, 34)
                if str(train.at[i,'Age'])=='nan':
                    if str(training_data.at[i,'Parch'])!= 0:
                        train.at[i,'Age']= age
                    else:
                        train.at[i,'Age']= child_age
    mean_age = int(train['Age'].mean())
    train['Age'] =  train['Age'].fillna(randint(mean_age-4, mean_age+4))
    return train

def classify_social_status(train):
    def class_affilier(title):
        title = str(title)
        if title in [' Mr']:
            text = 'RegularGuy'
        elif title in [' Master', ' Don', ' Jonkheer', ' Sir']:
            text = 'RichGuy'
        elif title in [' Dr',' Rev',' Major',' Col',' Capt',' Major']:
            text = 'SmartGuy'
        elif title in [' Miss', ' Mrs', ' Ms', ' Mlle']:
            text = 'RegularMiss'
        elif title in [' the Countess',' Lady',' Mme',' Dona']:
            text = 'RichMiss'
        else:
            print(title)
        return text
    train['SocialClass'] = train['Title'].apply(class_affilier)
    return train

def normalize(train):
    train['Bridge'].replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'], [0, 1, 2, 3, 4, 5, 6, 7], inplace = True)
    train['SocialClass'].replace(['RichMiss', 'RegularMiss', 'SmartGuy', 'RichGuy', 'RegularGuy'], [0, 1, 2, 3, 4], inplace = True)
    train['PassengerByClass'] = train['PassengerByClass'] / train['PassengerByClass'].max()
    train['PassengerByBridge'] = train['PassengerByBridge'] / train['PassengerByBridge'].max()
    train['Age'] = train['Age'] / train['Age'].max()
    return train

def find_correlation(train):
    corr_matrix = train.select_dtypes(np.number).corr()
    corr_matrix[(corr_matrix < 0.1) & (corr_matrix > -0.1)] = 0
    corr = corr_matrix["Survived"].sort_values(ascending = False)
    print(f'Correlation: {corr}')
    indexNames = corr[abs(corr.values) < 0.4].index.values
    indexNames = np.setdiff1d(indexNames, ['Id','MSSubClass'])

def train_model(X_train, y_train):
    numerical_features = make_column_selector(dtype_include = np.number)
    categorical_features = make_column_selector(dtype_exclude = np.number)
    numerical_pipeline = make_pipeline(SimpleImputer(),
                                    StandardScaler())
    categorical_pipeline = make_pipeline(SimpleImputer(strategy=('most_frequent')),
                                        OneHotEncoder())
    preprocessor = make_column_transformer((numerical_pipeline, numerical_features),
                                        (categorical_pipeline, categorical_features))
    if args.model == 'knn':
        model = make_pipeline(preprocessor,
                            KNeighborsClassifier())
    elif args.model == 'sgd':
        model = make_pipeline(preprocessor,
                            SGDClassifier())
    elif args.model == 'svc':
        model = make_pipeline(preprocessor,
                            SVC())
    elif args.model == 'mlp':
        model = make_pipeline(preprocessor,
                            MLPClassifier(max_iter=args.max_iter))
    
    model.fit(X_train, y_train)
    return model

def preprocessing(train):
    test = pd.read_csv('test.csv')
    train = get_social_staus(train, test)
    train = get_familysize(train)
    train = get_embark_encoded(train)
    train = clean_ticket_text(train)
    train = clean_cabin_number(train)
    train = get_bridge(train)
    train = fillna_age(train)
    train = classify_social_status(train)
    return train

def main():
    start = time.time()
    global args
    args = parser.parse_args()
    print(f'--- Selected model: {args.model} ---')
    print('--- Load data ---')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    print('--- Preprocessing ---')
    train = preprocessing(train)
    test = preprocessing(test)
    train_norm = normalize(train.copy())
    test_norm = normalize(test.copy())
    # find_correlation(train)
    # print(train.describe())
    # print(train.head())

    print('--- Train/Test split, split ratio:6:4 ---')
    y = train_norm['Survived']
    X = train_norm.drop(['Survived', 'Cabin', 'FamilyName', 'Ticket', 'SibSp',
                   'FamilySize', 'Parch', 'PassengerByBridge', 'TicketHeader',
                  'CabinLetter', 'Title', 'Embarked', 'TicketNumber', 'Age'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    print('--- Train ---')
    model = train_model(X_train, y_train)
    print(f'{args.model} -- training score: {model.score(X_train, y_train)}')
    print(f'{args.model} -- test score: {model.score(X_test, y_test)}')
    # print(model.steps[-1][1].feature_importances_)
    y_pred = model.predict(X_test.copy())
    print(f'Precission_score: {precision_score(y_test,y_pred)}')
    print(f'Recall_score: {recall_score(y_test,y_pred)}')
    print(f'F1-score: {f1_score(y_test,y_pred)}')

    test_x = test_norm.drop(['Cabin', 'FamilyName', 'Ticket', 'SibSp',
                   'FamilySize', 'Parch', 'PassengerByBridge', 'TicketHeader',
                  'CabinLetter', 'Title', 'Embarked', 'TicketNumber', 'Age'], axis = 1)
    prediction = model.predict(test_x.copy())
    test['Survived'] = prediction
    result = pd.concat([train, test], ignore_index=True)
    result.to_csv('result.csv')

    end = time.time()
    print(f'--- Finish! Time:{end-start} ---')
if __name__ == '__main__':
    main()
