#!/usr/bin/env python3
# Imports
import warnings
warnings.filterwarnings("ignore")
import re
import os
import sys
import numpy as np
import pandas as pd
from random import randint
import yaml

from sklearn.neighbors import KNeighborsClassifier

from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
import pickle

# The functions
def get_social_staus(train):
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
    def passenger_by_class(passenger_class):
        if passenger_class == 3:
            return 1026
        elif passenger_class == 2:
            return 674
        elif passenger_class == 1:
            return 689
    class_letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
    cabin_class = []
    class_meanings = []
    for i in class_letter:
        cabin = train[(train['CabinLetter']== i)]
        #print(f'Class {i}:\t\t{cabin["Fare"].mean()}\t:meanFare')
        cabin_class.append(i)
        class_meanings.append(cabin["Fare"].mean())
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
#    print(f'Correlation: {corr}')
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
    model = make_pipeline(preprocessor,
                            KNeighborsClassifier())
    
    model.fit(X_train, y_train)
    return model

def preprocessing(train):
    train = get_social_staus(train)
    train = get_familysize(train)
    train = get_embark_encoded(train)
    train = clean_ticket_text(train)
    train = clean_cabin_number(train)
    train = get_bridge(train)
    train = fillna_age(train)
    train = classify_social_status(train)
    return train

def load_data():
    if os.path.exists("./train.csv") and os.path.exists("./test.csv"):
        train = pd.read_csv("./train.csv")
        test = pd.read_csv("./test.csv")
        train.to_csv("/data/train.csv",index=False)
        test.to_csv("/data/test.csv",index=False)
        return "Data Loaded"
    else:
        return "train.csv and test.csv not exists in package"

def prep():
    if os.path.exists("/data/train.csv") and os.path.exists("/data/test.csv"):
        train = pd.read_csv("/data/train.csv")
        test = pd.read_csv("/data/test.csv")
        train_pro = preprocessing(train)
        test_pro = preprocessing(test)
        train_norm = normalize(train_pro.copy())
        test_norm = normalize(test_pro.copy())
        train_pro.to_csv("/data/train_pro.csv",index=False)
        test_pro.to_csv("/data/test_pro.csv",index=False)
        train_norm.to_csv("/data/train_norm.csv",index=False)
        test_norm.to_csv("/data/test_norm.csv",index=False)
        return "Data preprocessed"
    else:
        return "Data not loaded, please Load data first"

def split():
    if os.path.exists("/data/train_norm.csv") and os.path.exists("/data/test_norm.csv"):
        train_norm = pd.read_csv("/data/train_norm.csv")
        test_norm = pd.read_csv("/data/test_norm.csv")
        y = train_norm['Survived'].to_frame('Survived')
        X = train_norm.drop(['Survived', 'Cabin', 'FamilyName', 'Ticket', 'SibSp','FamilySize', 'Parch', 'PassengerByBridge', 'TicketHeader','CabinLetter', 'Title', 'Embarked', 'TicketNumber', 'Age'], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        X_train.to_csv("/data/X_train.csv",index=False)
        X_test.to_csv("/data/X_test.csv",index=False)
        y_train.to_csv("/data/y_train.csv",index=False)
        y_test.to_csv("/data/y_test.csv",index=False)
        return "Train test set splitted"
    else:
        return "No preprocessed data present, please preprocess it first"

def train():
    if os.path.exists("/data/X_train.csv") and os.path.exists("/data/y_train.csv"):
        X_train = pd.read_csv("/data/X_train.csv")
        y_train = pd.read_csv("/data/y_train.csv")
        model = train_model(X_train, y_train)
        pickle.dump(model, open('/data/model.pkl', 'wb'))
        return f" -- training score: {model.score(X_train, y_train)}"
    else:
        return "No training data present, please perform train test split first"

def evaluate():
    if os.path.exists("/data/X_test.csv") and os.path.exists("/data/y_test.csv") and os.path.exists("/data/model.pkl"):
        model = pickle.load(open('/data/model.pkl', 'rb'))
        X_test = pd.read_csv("/data/X_test.csv")
        y_test = pd.read_csv("/data/y_test.csv")
        y_pred = model.predict(X_test.copy())
        return f"Precission_score: {precision_score(y_test,y_pred)}\nRecall_score: {recall_score(y_test,y_pred)}\nF1-score: {f1_score(y_test,y_pred)}"
    else:
        return "Model not trained, please train it first"

def predict():
    if os.path.exists("/data/train_pro.csv") and os.path.exists("/data/test_pro.csv") and os.path.exists("/data/test_norm.csv") and os.path.exists("/data/model.pkl"):
        train = pd.read_csv("/data/train_pro.csv")
        test = pd.read_csv("/data/test_pro.csv")
        test_norm = pd.read_csv("/data/test_norm.csv")
        model = pickle.load(open('/data/model.pkl', 'rb'))
        test_x = test_norm.drop(['Cabin', 'FamilyName', 'Ticket', 'SibSp','FamilySize', 'Parch', 'PassengerByBridge', 'TicketHeader','CabinLetter', 'Title', 'Embarked', 'TicketNumber', 'Age'], axis = 1)
        prediction = model.predict(test_x.copy())
        test['Survived'] = prediction
        result = pd.concat([train, test], ignore_index=True)
        result.to_csv('/data/result.csv',index=False)
        return "Result predicted"
    else:
        return "Required files not found, did you finish all the previous process"

    
if __name__ == '__main__':
    if len(sys.argv) != 2 or (sys.argv[1] != "load_data" and sys.argv[1] != "preprocess" and sys.argv[1] != "split" and sys.argv[1] != "train" and sys.argv[1] != "evaluate" and sys.argv[1] != "predict"):
        print(f"Usage: {sys.argv[0]} load_data|preprocess|split|train|evaluate/predict")
        exit(1)

    command = sys.argv[1]
    if command == "load_data":
        print(yaml.dump({"output":load_data()}))
    elif command == "preprocess":
        print(yaml.dump({"output":prep()}))
    elif command == "split":
        print(yaml.dump({"output":split()}))
    elif command == "train":
        print(yaml.dump({"output":train()}))
    elif command == "evaluate":
        print(yaml.dump({"output":evaluate()}))
    elif command == "predict":
        print(yaml.dump({"output":predict()}))
    # Done
